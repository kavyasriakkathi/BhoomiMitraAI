"""
BhoomiMitra AI — WhatsApp Webhook Security

Implements HMAC-SHA256 signature verification for Meta WhatsApp Cloud API webhooks.
"""

import hmac
import hashlib
from fastapi import Request
from src.config import get_settings
from src.core.logging import logger


async def verify_webhook_signature(request: Request, body_bytes: bytes = None) -> bool:
    """
    Verifies the X-Hub-Signature-256 header sent by Meta using WHATSAPP_APP_SECRET.
    
    Returns:
        True if signature is valid or if WHATSAPP_APP_SECRET is not configured in development/test.
        False if signature verification fails or if WHATSAPP_APP_SECRET is missing in production.
    """
    settings = get_settings()

    if not settings.whatsapp_app_secret:
        if settings.is_production:
            logger.error(
                "[SECURITY CRITICAL] WHATSAPP_APP_SECRET is not configured in production. "
                "Webhook signature verification rejected."
            )
            return False
        logger.warning(
            "[SECURITY WARNING] WHATSAPP_APP_SECRET is not configured in environment variables. "
            "Webhook signature verification is SKIPPED."
        )
        return True

    if body_bytes is None:
        body_bytes = await request.body()

    signature_header = request.headers.get("x-hub-signature-256") or request.headers.get("X-Hub-Signature-256")

    client_ip = request.client.host if request.client else "unknown"

    if not signature_header:
        logger.error(
            f"[SECURITY ERROR] Signature verification failed! "
            f"Missing 'X-Hub-Signature-256' header in incoming POST request from IP {client_ip}."
        )
        return False

    expected_hash = hmac.new(
        settings.whatsapp_app_secret.encode("utf-8"),
        body_bytes,
        hashlib.sha256
    ).hexdigest()
    expected_signature = f"sha256={expected_hash}"

    if hmac.compare_digest(signature_header, expected_signature):
        logger.info("Webhook X-Hub-Signature-256 verified successfully.")
        return True
    else:
        logger.error(
            f"[SECURITY ERROR] Webhook signature verification FAILED!\n"
            f"  Client IP         : {client_ip}\n"
            f"  Received Header   : {signature_header}\n"
            f"  Expected Signature: {expected_signature}"
        )
        return False

    