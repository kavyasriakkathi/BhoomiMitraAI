from __future__ import annotations

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
    Logs raw request at line 1, verifies signature, parses payload, and queues background processing.
    """
    # VERY FIRST LINE: Read body bytes and log raw request + headers before any parsing or validation
    body_bytes = await request.body()
    client_ip = request.client.host if request.client else "unknown"
    headers_dict = dict(request.headers)

    try:
        raw_body_str = body_bytes.decode("utf-8", errors="replace")
    except Exception:
        raw_body_str = "<binary or decode error>"

    logger.info("=" * 80)
    logger.info("========== WHATSAPP WEBHOOK POST HIT ==========")
    logger.info(f"Client IP   : {client_ip}")
    logger.info(f"Headers     : {json.dumps(headers_dict, default=str)}")
    logger.info(f"Raw Body ({len(body_bytes)} bytes): {raw_body_str}")
    logger.info("=" * 80)

    # Step 1: Verify HMAC-SHA256 signature if app secret is configured
    is_signature_valid = await verify_webhook_signature(request, body_bytes)
    if not is_signature_valid:
        logger.error(f"[WEBHOOK REJECTED] Signature validation failed for POST request from IP {client_ip}.")
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"status": "error", "detail": "Invalid signature"})

    # Step 2: Parse raw JSON payload
    try:
        payload_dict = json.loads(body_bytes)
    except Exception as parse_err:
        logger.error(
            f"[WEBHOOK JSON PARSE ERROR] Invalid JSON payload received from Meta: {parse_err}\n"
            f"Raw body string: {raw_body_str}"
        )
        return {"status": "ignored", "reason": "invalid_json"}

    # Step 3: Validate against Pydantic schema
    try:
        payload = WhatsAppWebhookPayload(**payload_dict)
    except Exception as schema_err:
        logger.error(
            f"[WEBHOOK SCHEMA ERROR] Failed to parse payload into WhatsAppWebhookPayload model: {schema_err}\n"
            f"Payload dict: {json.dumps(payload_dict, default=str)}"
        )
        return {"status": "ignored", "reason": "schema_validation_error"}

    # Step 4: Extract messages and queue background tasks
    messages_queued = 0
    queued_message_ids: set[str] = set()

    if not payload.entry:
        logger.info("Webhook payload contained no 'entry' items.")
        return {"status": "ok", "messages_queued": 0}

    for entry_idx, entry in enumerate(payload.entry):
        for change_idx, change in enumerate(entry.changes):
            logger.info(f"Processing payload entry[{entry_idx}] change[{change_idx}] field: '{change.field}'")

            if change.field != "messages":
                logger.info(f"Ignoring non-messages field change: '{change.field}'")
                continue

            value = change.value
            if not value:
                logger.info("Webhook change event contained empty 'value' object.")
                continue

            # Check if payload contains status updates (sent, delivered, read receipts) instead of user messages
            raw_change_value = payload_dict.get("entry", [{}])[entry_idx].get("changes", [{}])[change_idx].get("value", {})
            statuses = raw_change_value.get("statuses") if isinstance(raw_change_value, dict) else None

            if statuses:
                for status_item in statuses:
                    msg_id = status_item.get("id", "unknown")
                    status_type = status_item.get("status", "unknown")
                    recipient_id = status_item.get("recipient_id", "unknown")
                    logger.info(f"[STATUS RECEIPT] Message {msg_id} to {recipient_id} status updated to: '{status_type}'")

            if not value.messages:
                logger.info("Webhook change event contained no incoming 'messages' array (likely a status/read receipt update).")
                continue

            logger.info(f"WEBHOOK USER MESSAGES COUNT: {len(value.messages)}")

            sender_name = None
            if value.contacts and value.contacts[0].profile:
                sender_name = value.contacts[0].profile.name

            for msg in value.messages:
                parsed = _extract_message(msg, sender_name)

                if parsed is None:
                    logger.info(f"Unsupported message type: '{msg.type}'. Skipping message ID {msg.id}.")
                    continue

                if parsed.message_id in queued_message_ids:
                    logger.warning(
                        f"Duplicate message ID '{parsed.message_id}' detected within same webhook payload batch. Skipping."
                    )
                    continue

                queued_message_ids.add(parsed.message_id)

                logger.info(f"INCOMING MESSAGE PARSED SUCCESSFULLY:")
                logger.info(f"  Message ID  : {parsed.message_id}")
                logger.info(f"  Message Type: {parsed.message_type}")
                logger.info(f"  Sender Phone: {parsed.phone_number}")
                logger.info(f"  Sender Name : {parsed.sender_name or 'Unknown'}")
                if parsed.text_content:
                    logger.info(f"  Message Text: {parsed.text_content[:150]}")

                # Step 5: Queue the background processing pipeline
                try:
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
                except Exception as queue_err:
                    logger.exception(f"Failed to queue background task for message {parsed.message_id}: {queue_err}")

    # Step 6: Return HTTP 200 OK immediately to Meta
    logger.info(f"Webhook processing complete. Responding HTTP 200 OK to Meta ({messages_queued} messages queued).")
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

