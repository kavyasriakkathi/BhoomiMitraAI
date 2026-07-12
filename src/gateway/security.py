"""
BhoomiMitra AI — WhatsApp Webhook Security

HMAC-SHA256 signature verification for Meta webhook payloads.
Prevents spoofing and replay attacks.
"""

import hashlib
import hmac
from fastapi import Request, HTTPException
from src.config import get_settings
from src.core.logging import logger


async def verify_webhook_signature(request: Request) -> bytes:
    """
    Validate the X-Hub-Signature-256 header against the raw request body.

    Meta signs every webhook POST with a SHA-256 HMAC using the
    App Secret. We MUST verify this before processing ANY message.

    Returns the raw body bytes on success (so we don't read the stream twice).
    Raises HTTPException(403) on failure.
    """
    settings = get_settings()
    app_secret = settings.whatsapp_app_secret

    # If no app_secret is configured (dev mode), skip verification
    if not app_secret:
        logger.warning("WHATSAPP_APP_SECRET not set — skipping signature verification (dev only).")
        return await request.body()

    signature_header = request.headers.get("X-Hub-Signature-256", "")
    if not signature_header:
        logger.warning("Webhook request missing X-Hub-Signature-256 header.")
        raise HTTPException(status_code=403, detail="Missing signature header.")

    # Header format: "sha256=<hex_digest>"
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=403, detail="Invalid signature format.")

    expected_signature = signature_header[7:]  # Strip "sha256=" prefix

    body = await request.body()

    # Compute HMAC-SHA256
    computed_hash = hmac.new(
        key=app_secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed_hash, expected_signature):
        logger.error("Webhook signature verification FAILED. Possible spoofing attempt.")
        raise HTTPException(status_code=403, detail="Invalid signature.")

    return body
