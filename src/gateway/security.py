"""
BhoomiMitra AI — WhatsApp Webhook Security

TEMPORARY DEBUG VERSION
"""

from fastapi import Request
from src.core.logging import logger


async def verify_webhook_signature(request: Request) -> bytes:
    """
    TEMPORARY DEBUG VERSION
    Skips signature verification completely.
    """
    logger.warning("⚠️ Signature verification temporarily disabled")
    return await request.body()
    