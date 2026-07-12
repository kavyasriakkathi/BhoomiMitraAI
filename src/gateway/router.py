"""
BhoomiMitra AI — WhatsApp Webhook Router

Handles:
  GET  /webhook/whatsapp  — Meta verification challenge
  POST /webhook/whatsapp  — Incoming message processing
"""

import json
from fastapi import APIRouter, Query, Request, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.core.database import get_db
from src.core.logging import logger
from src.gateway.security import verify_webhook_signature
from src.gateway.schemas import (
    WhatsAppWebhookPayload,
    ParsedIncomingMessage,
)
from src.gateway.service import (
    get_or_create_farmer,
    is_duplicate_message,
    store_incoming_message,
)
from src.ai.service import process_text_message
from src.gateway.whatsapp_client import send_text_message, mark_message_as_read

router = APIRouter()


# ──────────────────────────────────────────────
# GET — Webhook Verification Challenge
# ──────────────────────────────────────────────

@router.get("/whatsapp")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """
    Meta sends a GET request with a challenge token when you first
    register the webhook URL. We must return the challenge string
    if the verify_token matches ours.
    """
    settings = get_settings()

    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verification successful.")
        return int(hub_challenge)

    logger.warning(
        f"Webhook verification FAILED. "
        f"Mode={hub_mode}, Token mismatch."
    )
    raise HTTPException(status_code=403, detail="Verification failed.")


# ──────────────────────────────────────────────
# POST — Incoming Message Handler
# ──────────────────────────────────────────────

@router.post("/whatsapp")
async def receive_message(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receives incoming WhatsApp messages from Meta's Cloud API.

    Full pipeline:
      1. Verify HMAC-SHA256 signature.
      2. Parse the nested JSON payload.
      3. Extract message fields into a flat ParsedIncomingMessage.
      4. Deduplicate by message_id.
      5. Upsert the farmer record.
      6. Store the conversation entry.
      7. Generate AI response (text messages only).
      8. Send AI response back to farmer via WhatsApp.
      9. Mark incoming message as read (blue ticks).
    """

    # Step 1: Verify signature (returns raw body bytes)
    body_bytes = await verify_webhook_signature(request)

    # Step 2: Parse payload
    try:
        payload_dict = json.loads(body_bytes)
        payload = WhatsAppWebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        return {"status": "ignored", "reason": "parse_error"}

    # Step 3: Extract messages from the nested structure
    messages_processed = 0

    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                continue

            value = change.value
            if not value.messages:
                continue

            sender_name = None
            if value.contacts:
                sender_name = value.contacts[0].profile.name

            for msg in value.messages:
                parsed = _extract_message(msg, sender_name)

                if parsed is None:
                    logger.info(f"Unsupported message type: {msg.type}. Skipping.")
                    continue

                # Step 4: Deduplicate
                if await is_duplicate_message(db, parsed.message_id):
                    logger.warning(f"Duplicate message {parsed.message_id}. Skipping.")
                    continue

                # Step 5: Upsert farmer
                farmer = await get_or_create_farmer(
                    db, parsed.phone_number, sender_name
                )

                # Step 6: Store conversation
                conversation = await store_incoming_message(db, farmer, parsed)
                messages_processed += 1

                # Step 7: Generate AI response (text messages only for MVP)
                if parsed.message_type == "text" and parsed.text_content:
                    ai_response = await process_text_message(db, farmer, conversation)

                    # Step 8: Send response back to farmer
                    outbound_id = await send_text_message(
                        to_phone=parsed.phone_number,
                        message_text=ai_response,
                    )

                    # Update delivery status in DB
                    conversation.outbound_message_id = outbound_id
                    conversation.delivery_status = "sent" if outbound_id else "failed"
                    db.add(conversation)
                    await db.commit()

                    logger.info(
                        f"Reply {'sent' if outbound_id else 'FAILED'} to "
                        f"{parsed.phone_number} (outbound_id={outbound_id})"
                    )

                # Step 9: Mark incoming message as read (best-effort)
                await mark_message_as_read(parsed.message_id)

                logger.info(
                    f"Processed message {parsed.message_id} "
                    f"from {parsed.phone_number} (type={parsed.message_type})"
                )

    # Step 7: Always return 200 OK to Meta
    return {"status": "ok", "messages_processed": messages_processed}


# ──────────────────────────────────────────────
# Helper: Extract flat message from nested Meta format
# ──────────────────────────────────────────────

def _extract_message(msg, sender_name: str = None) -> ParsedIncomingMessage | None:
    """
    Convert a nested WhatsAppMessage into a flat ParsedIncomingMessage.
    Returns None for unsupported message types.
    """
    base = {
        "phone_number": msg.from_,
        "message_id": msg.id,
        "timestamp": msg.timestamp,
        "message_type": msg.type,
        "sender_name": sender_name,
    }

    if msg.type == "text" and msg.text:
        return ParsedIncomingMessage(**base, text_content=msg.text.body)

    elif msg.type == "audio" and msg.audio:
        return ParsedIncomingMessage(
            **base,
            media_id=msg.audio.id,
            media_mime_type=msg.audio.mime_type,
        )

    elif msg.type == "image" and msg.image:
        return ParsedIncomingMessage(
            **base,
            text_content=msg.image.caption,
            media_id=msg.image.id,
            media_mime_type=msg.image.mime_type,
        )

    return None  # Unsupported types (location, sticker, etc.)
