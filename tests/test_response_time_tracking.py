"""
Focused regression tests for persistent AI response-time tracking.

Verifies:
1. Conversation model has replied_at and response_time_seconds attributes.
2. ConversationUpdate and ConversationResponse schemas serialize and validate response_time_seconds and replied_at.
3. Pipeline Stage 7 records non-null response_time_seconds (> 0) and replied_at timestamp on completed conversations.
4. No real WhatsApp messages are dispatched during testing.
"""

import pytest
import time
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import Conversation, Farmer
from src.conversation.schemas import ConversationResponse, ConversationUpdate
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline


def test_conversation_model_has_response_time_fields():
    conv = Conversation(
        id=uuid4(),
        farmer_id=uuid4(),
        message_id="test_msg_001",
        user_message="Test question",
        user_message_type="text",
        ai_response="Test answer",
        replied_at=datetime.utcnow(),
        response_time_seconds=1.45,
    )
    assert conv.replied_at is not None
    assert conv.response_time_seconds == 1.45


def test_conversation_schemas_support_response_time_fields():
    now = datetime.utcnow()
    # Response schema
    resp = ConversationResponse(
        id=uuid4(),
        farmer_id=uuid4(),
        message_id="msg_schema_002",
        user_message="What is the price of urea?",
        user_message_type="text",
        ai_response="Urea is ₹295 per bag.",
        delivery_status="sent",
        created_at=now,
        replied_at=now,
        response_time_seconds=2.34,
    )
    dumped = resp.model_dump()
    assert dumped["response_time_seconds"] == 2.34
    assert dumped["replied_at"] is not None

    # Update schema
    update_data = ConversationUpdate(
        ai_response="Updated answer",
        delivery_status="sent",
        replied_at=now,
        response_time_seconds=0.85,
    )
    update_dump = update_data.model_dump(exclude_unset=True)
    assert update_dump["response_time_seconds"] == 0.85
    assert update_dump["replied_at"] == now


@pytest.mark.asyncio
async def test_pipeline_stage_7_records_response_time_and_replied_at():
    test_msg_id = f"wamid_test_rt_{uuid4().hex[:8]}"
    parsed = ParsedIncomingMessage(
        message_id=test_msg_id,
        phone_number="+919876543299",
        message_type="text",
        text_content="Korutla lo urea undha?",
        timestamp=str(int(time.time())),
        sender_name="Pilot Test Farmer",
    )

    mock_farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543299",
        preferred_language="te",
    )

    mock_conv = Conversation(
        id=uuid4(),
        farmer_id=mock_farmer.id,
        message_id=test_msg_id,
        user_message="Korutla lo urea undha?",
        user_message_type="text",
        created_at=datetime.utcnow(),
    )

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute = AsyncMock()

    class MockAsyncSessionContext:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with (
        patch("src.gateway.service.AsyncSessionLocal", return_value=MockAsyncSessionContext()),
        patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock) as mock_dup,
        patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_get_farmer,
        patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock) as mock_store,
        patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_process_text,
        patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send_text,
        patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock) as mock_mark_read,
    ):
        mock_dup.return_value = False
        mock_get_farmer.return_value = mock_farmer
        mock_store.return_value = mock_conv
        mock_process_text.return_value = "ఇక్కడ యూరియా అందుబాటులో లేదు."
        mock_send_text.return_value = "meta_outbound_id_12345"
        mock_mark_read.return_value = True

        # Run pipeline
        await process_message_pipeline(parsed=parsed, sender_name="Pilot Test Farmer")

        # Verify that mock_conv had replied_at and response_time_seconds populated
        assert mock_conv.outbound_message_id == "meta_outbound_id_12345"
        assert mock_conv.delivery_status == "sent"
        assert mock_conv.ai_response == "ఇక్కడ యూరియా అందుబాటులో లేదు."
        assert mock_conv.replied_at is not None
        assert isinstance(mock_conv.replied_at, datetime)
        assert mock_conv.response_time_seconds is not None
        assert mock_conv.response_time_seconds >= 0.0
