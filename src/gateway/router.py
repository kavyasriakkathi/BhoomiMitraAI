"""
BhoomiMitra AI — WhatsApp Webhook Router

Handles:
  GET  /webhook/whatsapp  — Meta verification challenge
  POST /webhook/whatsapp  — Incoming message processing
"""

import json
from fastapi import APIRouter, Query, Request, Depends, HTTPException, BackgroundTasks, Response
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
    Meta sends a GET request with a challenge token when registering the webhook URL.
    Returns plain text challenge string per Meta specification.
    """
    settings = get_settings()

    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verification successful.")
        return Response(content=str(hub_challenge or ""), media_type="text/plain")

    logger.warning(
        f"Webhook verification FAILED. "
        f"Mode={hub_mode}, Token mismatch."
    )
    raise HTTPException(status_code=403, detail="Verification failed.")


# ──────────────────────────────────────────────
# GET — Diagnostic Health Check (Non-secret)
# ──────────────────────────────────────────────

@router.get("/whatsapp/health")
async def whatsapp_health_check():
    """
    Production-safe status check for WhatsApp & AI service configuration.
    Exposes booleans and public IDs only. No secrets.
    """
    settings = get_settings()
    return {
        "success": True,
        "data": {
            "whatsapp_configured": bool(settings.whatsapp_api_token),
            "phone_number_id_configured": bool(settings.whatsapp_phone_number_id),
            "phone_number_id": settings.whatsapp_phone_number_id or "not_set",
            "verify_token_configured": bool(settings.whatsapp_verify_token),
            "gemini_configured": bool(settings.google_gemini_api_key),
            "database_configured": bool(settings.database_url),
            "app_env": settings.app_env,
        }
    }


# ──────────────────────────────────────────────
# POST — Incoming Message Handler
# ──────────────────────────────────────────────

@router.post("/whatsapp")
async def receive_message(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receives incoming WhatsApp messages from Meta's Cloud API.
    """
    logger.info("========== WHATSAPP WEBHOOK HIT ==========")

    # Step 1: Verify signature (returns raw body bytes)
    body_bytes = await verify_webhook_signature(request)

    try:
        raw_str = body_bytes.decode("utf-8", errors="ignore")
        logger.info(f"Raw webhook payload received ({len(body_bytes)} bytes)")
    except Exception:
        raw_str = ""

    # Step 2: Parse payload
    try:
        payload_dict = json.loads(body_bytes)
        payload = WhatsAppWebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        return {"status": "ignored", "reason": "parse_error"}

    # Step 3: Extract messages and queue background tasks
    messages_queued = 0

    if not payload.entry:
        logger.info("Webhook payload contained no 'entry' items.")
        return {"status": "ok", "messages_queued": 0}

    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                logger.info(f"Ignoring non-messages field change: {change.field}")
                continue

            value = change.value
            if not value or not value.messages:
                logger.info("Webhook change event contained no 'messages' array (likely a status/read update).")
                continue

            logger.info(f"WEBHOOK MESSAGE COUNT: {len(value.messages)}")

            sender_name = None
            if value.contacts and value.contacts[0].profile:
                sender_name = value.contacts[0].profile.name

            for msg in value.messages:
                parsed = _extract_message(msg, sender_name)

                if parsed is None:
                    logger.info(f"Unsupported message type: {msg.type}. Skipping.")
                    continue

                logger.info(f"MESSAGE ID: {parsed.message_id}")
                logger.info(f"MESSAGE TYPE: {parsed.message_type}")
                logger.info(f"SENDER PHONE: {parsed.phone_number}")
                if parsed.text_content:
                    logger.info(f"MESSAGE TEXT RECEIVED: {parsed.text_content[:100]}")

                # Step 4: Queue the background processing pipeline
                background_tasks.add_task(
                    process_message_pipeline,
                    parsed=parsed,
                    sender_name=sender_name
                )
                messages_queued += 1

                logger.info(
                    f"BACKGROUND PIPELINE QUEUED for message {parsed.message_id} "
                    f"from {parsed.phone_number}"
                )

    # Step 5: Return HTTP 200 OK immediately to Meta
    return {"status": "ok", "messages_queued": messages_queued}


# ──────────────────────────────────────────────
# Helper: Extract flat message from nested Meta format
# ──────────────────────────────────────────────

def _extract_message(msg, sender_name: str = None) -> ParsedIncomingMessage | None:
    """
    Convert a nested WhatsAppMessage into a flat ParsedIncomingMessage.
    Returns None for unsupported message types.
    """
    if not msg or not msg.from_ or not msg.id:
        return None

    base = {
        "phone_number": str(msg.from_),
        "message_id": str(msg.id),
        "timestamp": str(msg.timestamp or ""),
        "message_type": str(msg.type or "text"),
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

    return None

