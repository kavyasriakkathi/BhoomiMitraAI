"""
Tests verifying production error fixes:
1. Meta WhatsApp error 131056 (pair rate limit) detection & cooldown suppression.
2. Local process-level in-flight deduplication preventing DB unique constraint collisions.
3. 4xx BhoomiMitraException client notice logging without false application errors.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import Request

from src.gateway.whatsapp_client import (
    send_text_message,
    send_audio_message,
    record_pair_rate_limit,
    is_pair_rate_limited,
    _PAIR_RATE_LIMITED_NUMBERS,
)
from src.gateway.service import (
    acquire_redis_inflight_lock,
    release_redis_inflight_lock,
    _LOCAL_INFLIGHT_CACHE,
    _LOCAL_INFLIGHT_LOCK,
)
from src.core.exceptions import (
    BhoomiMitraException,
    bhoomimitra_exception_handler,
)


@pytest.fixture(autouse=True)
def clean_state():
    _PAIR_RATE_LIMITED_NUMBERS.clear()
    with _LOCAL_INFLIGHT_LOCK:
        _LOCAL_INFLIGHT_CACHE.clear()
    yield
    _PAIR_RATE_LIMITED_NUMBERS.clear()
    with _LOCAL_INFLIGHT_LOCK:
        _LOCAL_INFLIGHT_CACHE.clear()


@pytest.mark.asyncio
async def test_meta_131056_activates_cooldown_and_suppresses_subsequent_sends():
    """Verify that hitting Meta error 131056 marks the recipient in cooldown and prevents subsequent outbound spam."""
    test_phone = "918977112090"

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error":{"message":"(#131056) Pair rate limit hit","code":131056,"type":"OAuthException"}}'
    mock_resp.json.return_value = {
        "error": {
            "message": "(#131056) Pair rate limit hit",
            "code": 131056,
            "type": "OAuthException",
        }
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        res = await send_text_message(to_phone=test_phone, message_text="Hello farmer")
        assert res is None
        assert is_pair_rate_limited(test_phone) is True
        assert mock_post.call_count == 1

        # Second attempt should be immediately suppressed without making another HTTP call
        res2 = await send_text_message(to_phone=test_phone, message_text="Second message")
        assert res2 is None
        assert mock_post.call_count == 1  # Unchanged! Did not call Meta again

        # Audio send should also be immediately suppressed
        res_audio = await send_audio_message(to_phone=test_phone, media_id="media_12345")
        assert res_audio is None
        assert mock_post.call_count == 1  # Still unchanged!


@pytest.mark.asyncio
async def test_process_level_inflight_lock_drops_concurrent_duplicates():
    """Verify that multiple simultaneous requests with the same message_id are deduplicated in-memory."""
    msg_id = "wamid.HBgMOTE3Nzk0OTI2NzgyFQIAEhgUM0I3QUQ1NzBDRDc3NzFBNjhGNjQA"

    # First request acquires lock
    acquired1, token1, redis1 = await acquire_redis_inflight_lock(msg_id)
    assert acquired1 is True

    # Immediate second duplicate request within milliseconds MUST be rejected
    acquired2, token2, redis2 = await acquire_redis_inflight_lock(msg_id)
    assert acquired2 is False
    assert token2 is None

    # Releasing the lock frees it
    await release_redis_inflight_lock(msg_id, token1, redis1)

    acquired3, token3, redis3 = await acquire_redis_inflight_lock(msg_id)
    assert acquired3 is True
    await release_redis_inflight_lock(msg_id, token3, redis3)


@pytest.mark.asyncio
async def test_401_exception_does_not_log_application_error():
    """Verify that 401 on /auth/me logs as client notice, not logger.error."""
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/auth/me"

    exc = BhoomiMitraException(
        message="Authentication required. Please provide a valid Bearer token or login cookie.",
        status_code=401,
    )

    with patch("src.core.exceptions.logger.error") as mock_err, \
         patch("src.core.exceptions.logger.info") as mock_info:
        response = await bhoomimitra_exception_handler(mock_request, exc)
        assert response.status_code == 401
        assert mock_err.call_count == 0  # No logger.error called!
        assert mock_info.call_count == 1
