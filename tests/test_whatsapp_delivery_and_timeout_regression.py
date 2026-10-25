"""
Regression tests for WhatsApp outbound timeouts, delivery status callbacks,
automated bot/service message filtering, and analytics metric protection.
"""

import pytest
import time
import httpx
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import Conversation, Farmer
from src.ai.decision_engine import (
    AIDecisionEngine,
    is_automated_service_message,
)
from src.gateway.service import (
    update_conversation_delivery_status,
    process_message_pipeline,
)
from src.gateway.whatsapp_client import send_text_message
from src.gateway.router import _is_duplicate_or_record_message
from src.gateway.schemas import ParsedIncomingMessage
from src.analytics.service import AnalyticsService, classify_intent


def test_classify_intent_automated_service_notification():
    """Verify automated_service_notification is classified under 'other'."""
    assert classify_intent("automated_service_notification") == "other"


def test_automated_service_detection():
    """Verify various third-party bot and service messages are detected as automated."""
    lic_sample = "Thank you for using LIC WhatsApp Services. Please choose from the menu below."
    assert is_automated_service_message(lic_sample) is True
    assert AIDecisionEngine.is_greeting_only(lic_sample) is False

    otp_sample = "Your verification code OTP is 492810. Do not share this with anyone."
    assert is_automated_service_message(otp_sample) is True
    assert AIDecisionEngine.is_greeting_only(otp_sample) is False

    # Genuine greeting should NOT be detected as automated service
    greeting_sample = "నమస్తే భూమిమిత్ర"
    assert is_automated_service_message(greeting_sample) is False
    assert AIDecisionEngine.is_greeting_only(greeting_sample) is True

    english_greeting = "Hello BhoomiMitra"
    assert is_automated_service_message(english_greeting) is False
    assert AIDecisionEngine.is_greeting_only(english_greeting) is True


@pytest.mark.asyncio
async def test_process_message_handles_lic_automated_message():
    """Verify incoming LIC message is routed to automated_service_notification without LLM call."""
    engine = AIDecisionEngine()
    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    mock_farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="te",
    )
    mock_conv = Conversation(
        id=uuid4(),
        farmer_id=mock_farmer.id,
        message_id="test_lic_msg_101",
        user_message="Hi! Thank you for using LIC WhatsApp Services.",
        delivery_status="pending",
    )

    with patch.object(engine, "detect_all_intents") as mock_detect:
        reply = await engine.process_message(mock_db, mock_farmer, mock_conv)

    # Should not call domain detection
    mock_detect.assert_not_called()
    assert mock_conv.intent == "automated_service_notification"
    assert "BhoomiMitra" in reply or "భూమిమిత్ర" in reply
    assert mock_db.commit.called


@pytest.mark.asyncio
async def test_update_conversation_delivery_status_progression():
    """Verify monotonic status progression: pending -> sent -> delivered -> read."""
    outbound_wamid = f"wamid_test_{uuid4().hex[:8]}"
    mock_conv = Conversation(
        id=uuid4(),
        farmer_id=uuid4(),
        message_id="inbound_1",
        outbound_message_id=outbound_wamid,
        delivery_status="sent",
    )

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conv
    mock_session.execute = AsyncMock(return_value=mock_result)

    class MockAsyncSessionContext:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("src.gateway.service.AsyncSessionLocal", return_value=MockAsyncSessionContext()):
        # 1. Update from 'sent' to 'delivered' -> succeeds
        status_payload_delivered = {"id": outbound_wamid, "status": "delivered"}
        res1 = await update_conversation_delivery_status(status_payload_delivered)
        assert res1 is True
        assert mock_conv.delivery_status == "delivered"

        # 2. Cannot revert from 'delivered' back to 'sent'
        status_payload_sent = {"id": outbound_wamid, "status": "sent"}
        res2 = await update_conversation_delivery_status(status_payload_sent)
        assert res2 is False
        assert mock_conv.delivery_status == "delivered"

        # 3. Update from 'delivered' to 'read' -> succeeds
        status_payload_read = {"id": outbound_wamid, "status": "read"}
        res3 = await update_conversation_delivery_status(status_payload_read)
        assert res3 is True
        assert mock_conv.delivery_status == "read"


@pytest.mark.asyncio
async def test_send_text_message_exhausts_retries_on_timeout():
    """Verify send_text_message handles 3 consecutive timeouts gracefully and returns None."""
    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings, \
         patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Connection timed out")), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_config = MagicMock()
        mock_config.whatsapp_api_token = "fake_token"
        mock_config.whatsapp_phone_number_id = "fake_phone_id"
        mock_config.whatsapp_api_timeout_seconds = 0.1
        mock_settings.return_value = mock_config

        result = await send_text_message(
            to_phone="919876543210",
            message_text="Test timeout handling",
        )

        assert result is None
        # Should have slept between attempt 1 and 2, and attempt 2 and 3 (2 sleeps total)
        assert mock_sleep.await_count == 2


@pytest.mark.asyncio
async def test_analytics_metrics_exclude_failed_timeouts_from_response_time():
    """Verify conversations with delivery_status='failed' do not inflate average response time."""
    mock_db = AsyncMock()

    # Mock delivery status breakdown query
    del_res = MagicMock()
    del_res.all.return_value = [("sent", 8), ("delivered", 2), ("failed", 5)]

    # Mock response time query - only sent/delivered/read should be queried
    rt_res = MagicMock()
    # Average response time of 2.15s, 0 slow responses
    rt_res.first.return_value = (2.15, 0)

    # Empty mock responses for other queries
    generic_res = MagicMock()
    generic_res.scalar.return_value = 0
    generic_res.all.return_value = []

    async def mock_execute(stmt):
        stmt_str = str(stmt).lower()
        if "delivery_status" in stmt_str and "group by" in stmt_str:
            return del_res
        elif "avg" in stmt_str and "response_time_seconds" in stmt_str:
            # Verify the query filters by delivery_status in sent/delivered/read
            assert "delivery_status in" in stmt_str or "delivery_status =" in stmt_str
            return rt_res
        return generic_res

    mock_db.execute = AsyncMock(side_effect=mock_execute)

    svc = AnalyticsService(mock_db)
    summary = await svc.get_summary()

    # Delivered count (8 sent + 2 delivered = 10)
    assert summary.delivery.sent == 10
    assert summary.delivery.failed == 5
    assert summary.delivery.success_rate_pct == round(10 / 15 * 100.0, 2)

    # Average response time accurately reflects successful queries, not 48s timeouts
    assert summary.avg_response_time_seconds == 2.15
    assert summary.slow_responses_count == 0


@pytest.mark.asyncio
async def test_webhook_deduplication():
    """Verify _is_duplicate_or_record_message correctly detects duplicates."""
    test_id = f"wamid_dedup_{uuid4().hex[:10]}"
    # First arrival should NOT be duplicate
    is_dup1 = await _is_duplicate_or_record_message(test_id)
    assert is_dup1 is False

    # Second arrival of identical message_id SHOULD be detected as duplicate
    is_dup2 = await _is_duplicate_or_record_message(test_id)
    assert is_dup2 is True
