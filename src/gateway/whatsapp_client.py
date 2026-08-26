"""
BhoomiMitra AI — WhatsApp Outbound Client

Sends messages back to farmers via Meta's WhatsApp Cloud API.
Handles text messages, retries on failure, and delivery logging.
"""

import time
import httpx
from typing import Optional
from src.config import get_settings
from src.core.logging import logger


# Meta Cloud API base URL
META_API_VERSION = "v20.0"
META_BASE_URL = f"https://graph.facebook.com/{META_API_VERSION}"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


async def send_text_message(
    to_phone: str,
    message_text: str,
) -> Optional[str]:
    """
    Send a text message to a farmer via WhatsApp Cloud API.

    Args:
        to_phone: Farmer's phone number (with country code, e.g. "919876543210").
        message_text: The AI-generated response text.

    Returns:
        The Meta message ID on success, or None on failure.
    """
    settings = get_settings()

    if not settings.whatsapp_api_token or not settings.whatsapp_phone_number_id:
        logger.error(
            "[WHATSAPP OUTBOUND ERROR] Credentials not configured. "
            "Set WHATSAPP_API_TOKEN and WHATSAPP_PHONE_NUMBER_ID."
        )
        return None

    url = f"{META_BASE_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_text,
        },
    }

    logger.info(
        f"[WHATSAPP OUTBOUND START]\n"
        f"  URL             : {url}\n"
        f"  Phone Number ID : {settings.whatsapp_phone_number_id}\n"
        f"  Recipient Phone : {to_phone}\n"
        f"  Message Length  : {len(message_text)} chars\n"
        f"  Message Preview : '{message_text[:100]}...'"
    )

    total_wa_start = time.time()

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        attempt_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, json=payload)

            duration = time.time() - attempt_start
            logger.info(
                f"[WHATSAPP OUTBOUND RESPONSE] (Attempt {attempt}/{MAX_RETRIES}, took {duration:.2f}s):\n"
                f"  HTTP Status Code: {response.status_code}\n"
                f"  Response Body   : {response.text}"
            )

            if response.status_code == 200:
                data = response.json()
                wa_message_id = data.get("messages", [{}])[0].get("id")
                total_duration = time.time() - total_wa_start
                logger.info(
                    f"[WHATSAPP OUTBOUND SUCCESS] Delivered to {to_phone} in {total_duration:.2f}s "
                    f"(wa_id={wa_message_id})"
                )
                return wa_message_id

            if response.status_code == 401:
                logger.error(
                    f"[WHATSAPP OUTBOUND ERROR] HTTP 401 Unauthorized for {to_phone} — "
                    f"The Meta WHATSAPP_API_TOKEN is invalid or expired.\n"
                    f"Meta Response: {response.text}"
                )
                try:
                    from src.core.alerting import dispatch_founder_alert, AlertCategory, AlertSeverity
                    import asyncio
                    asyncio.create_task(dispatch_founder_alert(
                        category=AlertCategory.AUTH_FAILURE,
                        severity=AlertSeverity.CRITICAL,
                        component="whatsapp_gateway",
                        summary="Meta WHATSAPP_API_TOKEN rejected with HTTP 401 Unauthorized.",
                        recommended_action="Update WHATSAPP_API_TOKEN in Render environment variables immediately.",
                        details={"status_code": 401}
                    ))
                except Exception as alert_err:
                    logger.debug(f"Founder alert dispatch skipped: {alert_err}")
                return None

            if response.status_code == 403:
                logger.error(
                    f"[WHATSAPP OUTBOUND ERROR] HTTP 403 Forbidden for {to_phone} — "
                    f"Check phone number ID ({settings.whatsapp_phone_number_id}) permissions.\n"
                    f"Meta Response: {response.text}"
                )
                return None

            if response.status_code == 400:
                logger.error(
                    f"[WHATSAPP OUTBOUND ERROR] HTTP 400 Bad Request for {to_phone} — "
                    f"Meta Response: {response.text}"
                )
                return None

            # Rate limited by Meta — wait and retry
            if response.status_code == 429:
                logger.warning(
                    f"[WHATSAPP OUTBOUND RATE LIMIT] Rate limited by Meta (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {RETRY_DELAY_SECONDS * attempt}s...\n"
                    f"Meta Response: {response.text}"
                )
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue

            # Non-retryable error
            logger.error(
                f"[WHATSAPP OUTBOUND ERROR] Failed to send message to {to_phone}: "
                f"HTTP {response.status_code} — Meta Response: {response.text}"
            )
            return None

        except httpx.TimeoutException:
            duration = time.time() - attempt_start
            logger.warning(
                f"[WHATSAPP OUTBOUND TIMEOUT] Timeout after {duration:.2f}s sending to {to_phone} (attempt {attempt}/{MAX_RETRIES})."
            )
            if attempt < MAX_RETRIES:
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            return None

        except Exception as e:
            duration = time.time() - attempt_start
            logger.exception(f"[WHATSAPP OUTBOUND UNEXPECTED ERROR] Failed sending message to {to_phone} after {duration:.2f}s: {e}")
            return None

    total_duration = time.time() - total_wa_start
    logger.error(f"[WHATSAPP OUTBOUND EXHAUSTED] All {MAX_RETRIES} retries exhausted for {to_phone} after {total_duration:.2f}s.")
    return None


async def mark_message_as_read(message_id: str) -> None:
    """
    Send a 'read' receipt back to Meta so the farmer sees blue ticks.
    This is a best-effort call — failures are logged but not retried.
    """
    settings = get_settings()

    if not settings.whatsapp_api_token or not settings.whatsapp_phone_number_id:
        return

    url = f"{META_BASE_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        logger.info(
            f"WHATSAPP READ RECEIPT RESPONSE for message {message_id}: "
            f"HTTP {response.status_code} | Body: {response.text}"
        )
    except Exception as e:
        logger.warning(f"Failed to send read receipt for {message_id}: {e}")


async def download_media_bytes(media_id: str) -> Optional[tuple[bytes, str]]:
    """
    Downloads media from Meta Cloud API using a two-step process:
    1. Resolve the media URL via the media_id.
    2. Download the actual binary data from the resolved URL.
    
    Returns:
        A tuple of (raw_bytes, mime_type) on success, or None on failure.
    """
    settings = get_settings()

    if not settings.whatsapp_api_token:
        logger.error("WhatsApp API credentials not configured.")
        return None

    # Step 1: Resolve Media URL
    resolve_url = f"{META_BASE_URL}/{media_id}"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_api_token}",
    }

    media_url = None
    mime_type = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(resolve_url, headers=headers)
                
            if response.status_code == 200:
                data = response.json()
                media_url = data.get("url")
                mime_type = data.get("mime_type")
                break
                
            if response.status_code == 404:
                logger.error(f"Media {media_id} not found or expired.")
                return None
                
            if response.status_code == 429:
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
                
            logger.error(f"Failed to resolve media {media_id}: HTTP {response.status_code} - {response.text}")
            return None
            
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            return None
        except Exception as e:
            logger.exception(f"Unexpected error resolving media {media_id}: {e}")
            return None

    if not media_url:
        logger.error(f"Failed to resolve media URL for {media_id} after {MAX_RETRIES} attempts.")
        return None

    # Step 2: Download Binary Data
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Meta requires the Bearer token even for the direct media URL download
                media_response = await client.get(media_url, headers=headers)
                
            if media_response.status_code == 200:
                logger.info(f"Successfully downloaded media {media_id} ({len(media_response.content)} bytes)")
                return media_response.content, mime_type
                
            if media_response.status_code == 429:
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
                
            logger.error(f"Failed to download media bytes for {media_id}: HTTP {media_response.status_code}")
            return None
            
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                import asyncio
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            return None
        except Exception as e:
            logger.exception(f"Unexpected error downloading media {media_id}: {e}")
            return None

    logger.error(f"Failed to download media bytes for {media_id} after {MAX_RETRIES} attempts.")
    return None
