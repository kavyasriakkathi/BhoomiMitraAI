"""
BhoomiMitra AI — WhatsApp Webhook Router

Handles:
  GET  /webhook/whatsapp  — Meta verification challenge
  POST /webhook/whatsapp  — Incoming message processing
"""

import json
from fastapi import APIRouter, Query, Request, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.core.database import get_db
from src.core.logging import logger
from src.gateway.security import verify_webhook_signature
from src.gateway.schemas import (
    WhatsAppWebhookPayload,
    ParsedIncomingMessage,
)
from src.gateway.service import process_message_pipeline

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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Receives incoming WhatsApp messages from Meta's Cloud API.

    Full pipeline:
      1. Verify HMAC-SHA256 signature.
      2. Parse the nested JSON payload.
      3. Extract message fields into a flat ParsedIncomingMessage.
      4. Queue the background processing pipeline.
      5. Immediately return HTTP 200 OK to Meta to prevent timeouts.
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

    # Step 3: Extract messages and queue background tasks
    messages_queued = 0

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

                # Step 4: Queue the background processing pipeline
                background_tasks.add_task(
                    process_message_pipeline,
                    db=db,
                    parsed=parsed,
                    sender_name=sender_name
                )
                messages_queued += 1

                logger.info(
                    f"Queued background processing for message {parsed.message_id} "
                    f"from {parsed.phone_number} (type={parsed.message_type})"
                )

    # Step 5: Always return 200 OK to Meta immediately
    return {"status": "ok", "messages_queued": messages_queued}


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
