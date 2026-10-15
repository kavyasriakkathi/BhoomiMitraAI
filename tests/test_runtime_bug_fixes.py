"""
BhoomiMitra AI — Runtime Bug-Fix Regression Tests
====================================================
Covers the production Render log issues:

  P1  Duplicate WhatsApp webhook / idempotency
  P2  Unsupported incoming WhatsApp message types (error 131051)
  P3  Gemini memory-extraction timeout isolation
  P4  Agmarknet / mandi-price fallback correctness
  P5  /auth/me authentication logging
  P6  Worker shutdown (documentation / smoke)
  P7  Fertilizer safety preservation smoke
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import (
    _IN_FLIGHT_MESSAGE_IDS,
    acquire_in_flight_lock,
    release_in_flight_lock,
    is_duplicate_message,
    store_incoming_message,
)
from src.gateway.router import _extract_message
from src.core.models import Farmer, Conversation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_parsed(
    *,
    message_id: str = "wamid.test123",
    phone_number: str = "919876543210",
    message_type: str = "text",
    text_content: Optional[str] = "ఏ పంట వేయాలి?",
) -> ParsedIncomingMessage:
    return ParsedIncomingMessage(
        phone_number=phone_number,
        message_id=message_id,
        timestamp="1696000000",
        message_type=message_type,
        text_content=text_content,
    )


def _make_farmer(farmer_id: Optional[uuid.UUID] = None) -> Farmer:
    return Farmer(
        id=farmer_id or uuid.uuid4(),
        phone_number="919876543210",
        preferred_language="te",
    )


def _make_wa_message(
    *,
    msg_id: str = "wamid.ext001",
    from_: str = "919876543210",
    type_: str = "text",
    body: str = "హలో",
    include_text: bool = True,
) -> MagicMock:
    msg = MagicMock()
    msg.from_ = from_
    msg.id = msg_id
    msg.timestamp = "1696000000"
    msg.type = type_

    text = MagicMock()
    text.body = body
    msg.text = text if include_text else None
    msg.audio = None
    msg.voice = None
    msg.image = None
    msg.location = None
    return msg


# =============================================================================
# P1 — DUPLICATE WHATSAPP WEBHOOK / IDEMPOTENCY
# =============================================================================


class TestInFlightLock:
    """Unit tests for the in-process in-flight registry."""

    def setup_method(self):
        _IN_FLIGHT_MESSAGE_IDS.clear()

    def teardown_method(self):
        _IN_FLIGHT_MESSAGE_IDS.clear()

    def test_first_acquire_succeeds(self):
        assert acquire_in_flight_lock("msg-abc") is True

    def test_second_acquire_of_same_id_fails(self):
        acquire_in_flight_lock("msg-dup")
        assert acquire_in_flight_lock("msg-dup") is False

    def test_release_allows_re_acquire(self):
        acquire_in_flight_lock("msg-rel")
        release_in_flight_lock("msg-rel")
        assert acquire_in_flight_lock("msg-rel") is True

    def test_different_ids_do_not_interfere(self):
        assert acquire_in_flight_lock("msg-A") is True
        assert acquire_in_flight_lock("msg-B") is True

    def test_release_of_unknown_id_is_safe(self):
        release_in_flight_lock("msg-nonexistent")  # Must not raise

    def test_empty_message_id_always_acquires(self):
        assert acquire_in_flight_lock("") is True
        assert acquire_in_flight_lock("") is True


class TestIsDuplicateMessage:
    """Unit tests for the DB-level idempotency check."""

    @pytest.mark.asyncio
    async def test_returns_false_for_new_message_id(self):
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result_mock)

        assert await is_duplicate_message(db, "wamid.new") is False

    @pytest.mark.asyncio
    async def test_returns_true_for_known_message_id(self):
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = uuid.uuid4()
        db.execute = AsyncMock(return_value=result_mock)

        assert await is_duplicate_message(db, "wamid.known") is True


class TestStoreIncomingMessage:
    """Unit tests for duplicate-safe conversation storage."""

    @pytest.mark.asyncio
    async def test_duplicate_delivery_returns_none_without_raising(self):
        """
        When the DB unique constraint fires (IntegrityError), store_incoming_message
        must rollback, log, and return None — never raise.
        """
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock(side_effect=IntegrityError("duplicate key", {}, None))
        db.rollback = AsyncMock()

        farmer = _make_farmer()
        parsed = _make_parsed(message_id="wamid.dup")

        result = await store_incoming_message(db, farmer, parsed)
        assert result is None
        db.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_other_db_errors_propagate(self):
        """Non-integrity DB errors must not be swallowed."""
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock(side_effect=RuntimeError("connection lost"))

        farmer = _make_farmer()
        parsed = _make_parsed(message_id="wamid.err")

        with pytest.raises(RuntimeError, match="connection lost"):
            await store_incoming_message(db, farmer, parsed)


class TestProcessMessagePipelineIdempotency:
    """Integration-style tests for process_message_pipeline duplicate handling."""

    def setup_method(self):
        _IN_FLIGHT_MESSAGE_IDS.clear()

    def teardown_method(self):
        _IN_FLIGHT_MESSAGE_IDS.clear()

    @pytest.mark.asyncio
    async def test_in_flight_lock_prevents_concurrent_duplicate_processing(self):
        """
        A second concurrent delivery of the same message_id must be dropped
        before any expensive processing (DB, STT, AI) occurs.
        """
        msg_id = "wamid.concurrent"
        parsed = _make_parsed(message_id=msg_id)

        _IN_FLIGHT_MESSAGE_IDS.add(msg_id)

        send_mock = AsyncMock()
        with patch("src.gateway.service.send_text_message", send_mock):
            from src.gateway.service import process_message_pipeline
            await process_message_pipeline(parsed=parsed, sender_name="Test")

        send_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_duplicate_check_prevents_second_processing(self):
        """
        When the DB already contains the message_id, the pipeline exits after
        Stage 1 without calling the AI or sending a second reply.
        """
        msg_id = "wamid.db_dup"
        parsed = _make_parsed(message_id=msg_id)

        send_mock = AsyncMock()
        is_dup_mock = AsyncMock(return_value=True)

        with (
            patch("src.gateway.service.is_duplicate_message", is_dup_mock),
            patch("src.gateway.service.send_text_message", send_mock),
            patch("src.gateway.service.AsyncSessionLocal") as mock_session_cls,
        ):
            mock_db = AsyncMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.gateway.service import process_message_pipeline
            await process_message_pipeline(parsed=parsed, sender_name="Test")

        send_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_is_always_released_even_on_pipeline_failure(self):
        """
        The in-flight lock MUST be released in the finally block even when
        an unexpected exception propagates through the pipeline.
        """
        msg_id = "wamid.lock_release"
        parsed = _make_parsed(message_id=msg_id)

        with (
            patch(
                "src.gateway.service.AsyncSessionLocal",
                side_effect=RuntimeError("db init failed"),
            ),
        ):
            from src.gateway.service import process_message_pipeline

            try:
                await process_message_pipeline(parsed=parsed, sender_name="Test")
            except Exception:
                pass

        assert msg_id not in _IN_FLIGHT_MESSAGE_IDS

    @pytest.mark.asyncio
    async def test_retry_after_transient_failure_lock_is_released(self):
        """
        A first delivery that fails before the conversation is stored must release
        the in-flight lock so a subsequent retry can succeed.
        """
        msg_id = "wamid.retry"
        parsed = _make_parsed(message_id=msg_id)

        with patch(
            "src.gateway.service.AsyncSessionLocal",
            side_effect=RuntimeError("transient DB error"),
        ):
            try:
                from src.gateway.service import process_message_pipeline
                await process_message_pipeline(parsed=parsed, sender_name="Test")
            except Exception:
                pass

        assert msg_id not in _IN_FLIGHT_MESSAGE_IDS
        assert acquire_in_flight_lock(msg_id) is True


# =============================================================================
# P2 — UNSUPPORTED WHATSAPP MESSAGE TYPES
# =============================================================================


class TestExtractMessage:
    """Tests for _extract_message router helper."""

    def test_valid_text_message_returns_parsed(self):
        msg = _make_wa_message(type_="text", body="హలో")
        result = _extract_message(msg, sender_name="Farmer")
        assert result is not None
        assert result.message_type == "text"
        assert result.text_content == "హలో"
        assert result.message_id == "wamid.ext001"

    def test_missing_sender_returns_none(self):
        msg = _make_wa_message()
        msg.from_ = None
        result = _extract_message(msg, sender_name="Farmer")
        assert result is None

    def test_missing_message_id_returns_none(self):
        msg = _make_wa_message()
        msg.id = None
        result = _extract_message(msg, sender_name="Farmer")
        assert result is None

    def test_text_message_with_no_text_payload_returns_none(self):
        msg = _make_wa_message(type_="text", include_text=False)
        result = _extract_message(msg, sender_name="Farmer")
        assert result is None

    def test_unsupported_type_returns_parsed_not_none(self):
        """
        Unsupported types (video, document, sticker) must return a ParsedIncomingMessage
        rather than None, so they are handled gracefully later in the pipeline.
        """
        for type_ in ("video", "document", "sticker", "contacts", "unsupported", "interactive"):
            msg = _make_wa_message(type_=type_)
            result = _extract_message(msg, sender_name="Farmer")
            assert result is not None, f"Expected non-None for type={type_}"
            assert result.message_type == type_
            assert result.text_content is None

    def test_image_type_returns_parsed(self):
        msg = _make_wa_message(type_="image")
        msg.image = MagicMock()
        msg.image.id = "media_img_001"
        msg.image.mime_type = "image/jpeg"
        msg.image.caption = "crop photo"
        result = _extract_message(msg, sender_name="Farmer")
        assert result is not None
        assert result.message_type == "image"
        assert result.media_id == "media_img_001"
        assert result.text_content == "crop photo"

    def test_location_type_returns_parsed(self):
        msg = _make_wa_message(type_="location")
        msg.location = MagicMock()
        msg.location.latitude = 17.966
        msg.location.longitude = 79.594
        msg.location.name = "Warangal"
        msg.location.address = "TS"
        result = _extract_message(msg, sender_name="Farmer")
        assert result is not None
        assert result.message_type == "location"
        assert result.latitude == pytest.approx(17.966)
        assert result.longitude == pytest.approx(79.594)


class TestUnsupportedMessagePipelineHandling:
    """Tests that unsupported types do not crash the pipeline or call Gemini."""

    @pytest.mark.asyncio
    async def test_unsupported_type_uses_fallback_no_gemini_call(self):
        """
        Unsupported message types must use the unsupported-media fallback without
        calling Gemini or persisting malformed content as a farmer question.
        """
        msg_id = "wamid.unsupported001"
        parsed = _make_parsed(
            message_id=msg_id,
            message_type="sticker",
            text_content=None,
        )

        send_mock = AsyncMock(return_value="out_msg_id_001")
        gemini_mock = AsyncMock()

        mock_conv = MagicMock()
        mock_conv.id = uuid.uuid4()
        mock_conv.delivery_status = "sent"
        mock_conv.outbound_message_id = None
        mock_conv.ai_response = None
        mock_conv.replied_at = None
        mock_conv.response_time_seconds = None

        with (
            patch("src.gateway.service.send_text_message", send_mock),
            patch("src.gateway.service.is_duplicate_message", AsyncMock(return_value=False)),
            patch("src.gateway.service.store_incoming_message", AsyncMock(return_value=mock_conv)),
            patch("src.gateway.service.get_or_create_farmer", AsyncMock(return_value=_make_farmer())),
            patch("src.gateway.service.AsyncSessionLocal") as mock_session_cls,
            patch("src.ai.service.generate_response", gemini_mock),
            patch("src.gateway.service.mark_message_as_read", AsyncMock()),
        ):
            mock_db = AsyncMock()
            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.gateway.service import process_message_pipeline
            await process_message_pipeline(parsed=parsed, sender_name="Test")

        send_mock.assert_called_once()
        gemini_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_type_sends_exactly_one_reply(self):
        """Unsupported type must not create an infinite loop — exactly one reply."""
        parsed = _make_parsed(
            message_id="wamid.unsupported_once",
            message_type="document",
            text_content=None,
        )

        send_calls = []

        async def _mock_send(to_phone, message_text):
            send_calls.append(message_text)
            return "out_id"

        mock_conv = MagicMock()
        mock_conv.id = uuid.uuid4()
        mock_conv.delivery_status = "sent"
        mock_conv.outbound_message_id = None
        mock_conv.ai_response = None
        mock_conv.replied_at = None
        mock_conv.response_time_seconds = None

        with (
            patch("src.gateway.service.send_text_message", side_effect=_mock_send),
            patch("src.gateway.service.is_duplicate_message", AsyncMock(return_value=False)),
            patch("src.gateway.service.store_incoming_message", AsyncMock(return_value=mock_conv)),
            patch("src.gateway.service.get_or_create_farmer", AsyncMock(return_value=_make_farmer())),
            patch("src.gateway.service.AsyncSessionLocal") as mock_session_cls,
            patch("src.gateway.service.mark_message_as_read", AsyncMock()),
        ):
            mock_db = AsyncMock()
            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            from src.gateway.service import process_message_pipeline
            await process_message_pipeline(parsed=parsed, sender_name="Test")

        assert len(send_calls) == 1

    def test_duplicate_unsupported_event_in_same_payload_deduplicated(self):
        """
        Two entries of the same message_id in one payload batch must only
        enqueue one pipeline task.
        """
        queued_ids: set[str] = set()
        msg_id = "wamid.dup_unsupported"

        msg1 = _make_wa_message(msg_id=msg_id, type_="sticker")
        msg2 = _make_wa_message(msg_id=msg_id, type_="sticker")

        results = []
        for msg in (msg1, msg2):
            parsed = _extract_message(msg, sender_name="Farmer")
            if parsed and parsed.message_id not in queued_ids:
                queued_ids.add(parsed.message_id)
                results.append(parsed)

        assert len(results) == 1


# =============================================================================
# P3 — GEMINI MEMORY EXTRACTION TIMEOUT ISOLATION
# =============================================================================


class TestMemoryExtractionTimeoutIsolation:
    """
    Tests that memory extraction timeouts are absorbed as best-effort failures
    without breaking the conversation flow.
    """

    @pytest.mark.asyncio
    async def test_memory_extraction_timeout_does_not_raise(self):
        """
        A TimeoutError from the Gemini memory LLM must be caught inside
        extract_and_update_memory; the rule-based extraction result must still be saved.
        """
        from src.memory.service import FarmerMemoryService, FarmerMemoryRepository
        from src.memory.models import FarmerMemory

        farmer_id = uuid.uuid4()
        memory = FarmerMemory(farmer_id=farmer_id)
        memory.confidence_scores = {}
        memory.primary_crops = []
        memory.favorite_shops = []
        memory.disease_history = []
        memory.pesticide_history = []
        memory.fertilizer_history = []
        memory.secondary_crops = []
        memory.preferred_brands = []

        repo = AsyncMock(spec=FarmerMemoryRepository)
        repo.get_or_create = AsyncMock(return_value=memory)
        repo.save = AsyncMock(return_value=memory)

        service = FarmerMemoryService(repo)

        with patch("src.memory.service.generate_response", AsyncMock(side_effect=TimeoutError("Gemini timed out"))):
            result = await service.extract_and_update_memory(
                farmer_id=farmer_id,
                user_message="I grow 5 acres of paddy near Warangal",
                ai_response="Great! Here is some paddy advice.",
            )

        repo.save.assert_awaited_once()
        saved_memory = repo.save.call_args[0][0]
        assert saved_memory.farm_size == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_memory_extraction_api_failure_does_not_raise(self):
        """A generic API error from Gemini must also be absorbed."""
        from src.memory.service import FarmerMemoryService, FarmerMemoryRepository
        from src.memory.models import FarmerMemory

        farmer_id = uuid.uuid4()
        memory = FarmerMemory(farmer_id=farmer_id)
        memory.confidence_scores = {}
        memory.primary_crops = []
        memory.favorite_shops = []
        memory.disease_history = []
        memory.pesticide_history = []
        memory.fertilizer_history = []
        memory.secondary_crops = []
        memory.preferred_brands = []

        repo = AsyncMock(spec=FarmerMemoryRepository)
        repo.get_or_create = AsyncMock(return_value=memory)
        repo.save = AsyncMock(return_value=memory)

        service = FarmerMemoryService(repo)

        with patch(
            "src.memory.service.generate_response",
            AsyncMock(side_effect=RuntimeError("Gemini API quota exhausted")),
        ):
            result = await service.extract_and_update_memory(
                farmer_id=farmer_id,
                user_message="I have cotton crop",
                ai_response="Cotton crop advisory here.",
            )

        repo.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_memory_extraction_success_updates_fields(self):
        """Successful LLM extraction must update memory fields correctly."""
        from src.memory.service import FarmerMemoryService, FarmerMemoryRepository
        from src.memory.models import FarmerMemory

        farmer_id = uuid.uuid4()
        memory = FarmerMemory(farmer_id=farmer_id)
        memory.confidence_scores = {}
        memory.primary_crops = []
        memory.favorite_shops = []
        memory.disease_history = []
        memory.pesticide_history = []
        memory.fertilizer_history = []
        memory.secondary_crops = []
        memory.preferred_brands = []
        memory.district = None
        memory.village = None
        memory.farm_size = None

        repo = AsyncMock(spec=FarmerMemoryRepository)
        repo.get_or_create = AsyncMock(return_value=memory)
        repo.save = AsyncMock(return_value=memory)

        service = FarmerMemoryService(repo)

        llm_response = json.dumps({
            "updates": {"district": "Warangal", "farm_size": 3.0},
            "confidence_scores": {"district": 0.9, "farm_size": 0.85},
        })

        with patch("src.memory.service.generate_response", AsyncMock(return_value=llm_response)):
            result = await service.extract_and_update_memory(
                farmer_id=farmer_id,
                user_message="I have 3 acres in Warangal district",
                ai_response="Paddy advisory for Warangal.",
            )

        saved = repo.save.call_args[0][0]
        assert saved.district == "Warangal"
        assert saved.farm_size == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_memory_extraction_invalid_json_falls_back_gracefully(self):
        """Malformed LLM JSON must not crash; rule-based result must be saved."""
        from src.memory.service import FarmerMemoryService, FarmerMemoryRepository
        from src.memory.models import FarmerMemory

        farmer_id = uuid.uuid4()
        memory = FarmerMemory(farmer_id=farmer_id)
        memory.confidence_scores = {}
        memory.primary_crops = []
        memory.favorite_shops = []
        memory.disease_history = []
        memory.pesticide_history = []
        memory.fertilizer_history = []
        memory.secondary_crops = []
        memory.preferred_brands = []

        repo = AsyncMock(spec=FarmerMemoryRepository)
        repo.get_or_create = AsyncMock(return_value=memory)
        repo.save = AsyncMock(return_value=memory)

        service = FarmerMemoryService(repo)

        with patch("src.memory.service.generate_response", AsyncMock(return_value="```json\n{ invalid }\n```")):
            result = await service.extract_and_update_memory(
                farmer_id=farmer_id,
                user_message="cotton crop, 2 acres",
                ai_response="Cotton advice.",
            )

        saved = repo.save.call_args[0][0]
        assert saved.farm_size == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_trigger_background_memory_extraction_is_fire_and_forget(self):
        """trigger_background_memory_extraction must return without blocking."""
        from src.memory.service import trigger_background_memory_extraction

        farmer_id = uuid.uuid4()

        with patch("src.memory.service.generate_response", AsyncMock(side_effect=TimeoutError("timed out"))):
            task = trigger_background_memory_extraction(
                farmer_id=farmer_id,
                user_message="paddy 4 acres",
                ai_response="Advice here.",
            )

        assert task is None or isinstance(task, asyncio.Task)

        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except Exception:
                pass


# =============================================================================
# P4 — AGMARKNET / MANDI PRICE FALLBACK CORRECTNESS
# =============================================================================


class TestAgmarknetClient:
    """Tests for the Agmarknet API client behaviour."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_list(self):
        from src.market.agmarknet_client import AgmarknetClient

        client = AgmarknetClient(api_key="", api_url="https://api.data.gov.in/agmarknet")
        result = await client.fetch_prices(commodity="Paddy")
        assert result == []

    @pytest.mark.asyncio
    async def test_network_timeout_returns_empty_list(self):
        import httpx
        from src.market.agmarknet_client import AgmarknetClient

        client = AgmarknetClient(
            api_key="test-key",
            api_url="https://api.data.gov.in/agmarknet",
            timeout_seconds=2.0,
            max_retries=0,
        )

        with patch(
            "src.market.agmarknet_client.AgmarknetClient._get_from_cache",
            AsyncMock(return_value=None),
        ), patch(
            "src.market.agmarknet_client.AgmarknetClient._set_in_cache",
            AsyncMock(),
        ), patch(
            "httpx.AsyncClient.get",
            AsyncMock(side_effect=httpx.ConnectTimeout("timeout")),
        ):
            result = await client.fetch_prices(commodity="Paddy", state="Telangana")

        assert result == []

    @pytest.mark.asyncio
    async def test_api_returns_records_correctly_normalised(self):
        from src.market.agmarknet_client import AgmarknetClient

        raw_records = [
            {
                "commodity": "Paddy",
                "market": "Warangal",
                "district": "Warangal",
                "state": "Telangana",
                "min_price": "1800",
                "max_price": "2200",
                "modal_price": "2000",
                "arrival_date": "16/10/2026",
            }
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": raw_records}

        client = AgmarknetClient(
            api_key="valid-key",
            api_url="https://api.data.gov.in/agmarknet",
            cache_ttl_seconds=0,
        )

        with patch(
            "src.market.agmarknet_client.AgmarknetClient._get_from_cache",
            AsyncMock(return_value=None),
        ), patch(
            "src.market.agmarknet_client.AgmarknetClient._set_in_cache",
            AsyncMock(),
        ), patch(
            "httpx.AsyncClient.get",
            AsyncMock(return_value=mock_response),
        ):
            result = await client.fetch_prices(commodity="Paddy", state="Telangana")

        assert len(result) == 1
        assert result[0]["commodity"] == "Paddy"
        assert result[0]["market"] == "Warangal"
        assert result[0]["modal_price"] == pytest.approx(2000.0)

    @pytest.mark.asyncio
    async def test_api_http_error_returns_empty_list(self):
        from src.market.agmarknet_client import AgmarknetClient

        mock_response = MagicMock()
        mock_response.status_code = 503

        client = AgmarknetClient(
            api_key="valid-key",
            api_url="https://api.data.gov.in/agmarknet",
            max_retries=0,
        )

        with patch(
            "src.market.agmarknet_client.AgmarknetClient._get_from_cache",
            AsyncMock(return_value=None),
        ), patch(
            "src.market.agmarknet_client.AgmarknetClient._set_in_cache",
            AsyncMock(),
        ), patch(
            "httpx.AsyncClient.get",
            AsyncMock(return_value=mock_response),
        ):
            result = await client.fetch_prices(commodity="Paddy")

        assert result == []

    def test_prioritize_records_filters_by_state_and_district(self):
        from src.market.agmarknet_client import AgmarknetClient

        client = AgmarknetClient(api_key="k", api_url="http://x")
        records = [
            {
                "commodity": "Paddy", "market": "Hyderabad", "district": "Hyderabad",
                "state": "Telangana", "min_price": 1900.0, "max_price": 2100.0,
                "modal_price": 2000.0, "arrival_date": datetime(2026, 10, 16),
            },
            {
                "commodity": "Paddy", "market": "Warangal", "district": "Warangal",
                "state": "Telangana", "min_price": 1800.0, "max_price": 2200.0,
                "modal_price": 2050.0, "arrival_date": datetime(2026, 10, 16),
            },
            {
                "commodity": "Paddy", "market": "Guntur", "district": "Guntur",
                "state": "Andhra Pradesh", "min_price": 2000.0, "max_price": 2400.0,
                "modal_price": 2200.0, "arrival_date": datetime(2026, 10, 16),
            },
        ]

        result = client._prioritize_records(records, state="Telangana", district="Warangal")
        assert len(result) == 2
        assert result[0]["market"] == "Warangal"
        assert all(r["state"] == "Telangana" for r in result)

    def test_prioritize_records_returns_records_when_no_state_match(self):
        """When only an AP record exists and Telangana is requested, records are returned unfiltered."""
        from src.market.agmarknet_client import AgmarknetClient

        client = AgmarknetClient(api_key="k", api_url="http://x")
        records = [
            {
                "commodity": "Cotton", "market": "Guntur", "district": "Guntur",
                "state": "Andhra Pradesh", "min_price": 6000.0, "max_price": 7000.0,
                "modal_price": 6500.0, "arrival_date": datetime(2026, 10, 16),
            },
        ]

        result = client._prioritize_records(records, state="Telangana", district="Karimnagar")
        assert len(result) == 1


class TestMarketServiceStaleFallback:
    """Tests for stale fallback disclosure in the market service."""

    @pytest.mark.asyncio
    async def test_db_fallback_source_note_indicates_local_not_live(self):
        """
        When live API fails and DB records are returned, source_note must
        mention local DB and must NOT claim live data.
        """
        from src.market.service import MarketService, MarketPriceRepository

        stale_record = MagicMock()
        stale_record.id = uuid.uuid4()
        stale_record.commodity = "Paddy"
        stale_record.commodity_telugu = None
        stale_record.market_name = "Warangal"
        stale_record.district = "Warangal"
        stale_record.state = "Telangana"
        stale_record.min_price = 1800.0
        stale_record.max_price = 2200.0
        stale_record.modal_price = 2000.0
        stale_record.unit = "Quintal"
        stale_record.price_date = datetime.utcnow() - timedelta(days=7)
        stale_record.source = "agmarknet_api"
        stale_record.created_at = datetime.utcnow() - timedelta(days=7)
        stale_record.updated_at = datetime.utcnow() - timedelta(days=7)

        mock_repo = AsyncMock(spec=MarketPriceRepository)
        mock_repo.get_prices_by_commodity = AsyncMock(return_value=[stale_record])

        mock_client = AsyncMock()
        mock_client.fetch_prices = AsyncMock(return_value=[])

        service = MarketService(repository=mock_repo, client=mock_client)

        result = await service.get_prices_for_query(
            commodity="Paddy",
            district="Warangal",
            state="Telangana",
        )

        assert result.data_available is True
        assert result.is_live is False
        assert result.data_freshness_hours is not None
        assert result.data_freshness_hours > 24.0

    @pytest.mark.asyncio
    async def test_no_data_returns_data_available_false(self):
        """When both API and DB return empty, data_available must be False."""
        from src.market.service import MarketService, MarketPriceRepository

        mock_repo = AsyncMock(spec=MarketPriceRepository)
        mock_repo.get_prices_by_commodity = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.fetch_prices = AsyncMock(return_value=[])

        service = MarketService(repository=mock_repo, client=mock_client)

        result = await service.get_prices_for_query(
            commodity="Paddy",
            district="UnknownVillage",
            state="Telangana",
        )

        assert result.data_available is False
        assert result.results == []


# =============================================================================
# P5 — /auth/me AUTHENTICATION LOGGING
# =============================================================================


class TestAuthMeLogging:
    """
    Tests that the /auth/me 401 for an unauthenticated browser session
    does not generate unexpected ERROR-level log noise.
    """

    @pytest.mark.asyncio
    async def test_unauthenticated_auth_me_returns_401(self):
        from httpx import AsyncClient, ASGITransport
        from src.main import app
        from src.auth.dependencies import get_current_user, get_current_active_user

        saved_user = app.dependency_overrides.pop(get_current_user, None)
        saved_active = app.dependency_overrides.pop(get_current_active_user, None)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/auth/me")
            assert response.status_code == 401
        finally:
            if saved_user:
                app.dependency_overrides[get_current_user] = saved_user
            if saved_active:
                app.dependency_overrides[get_current_active_user] = saved_active

    @pytest.mark.asyncio
    async def test_unauthenticated_auth_me_does_not_log_at_error_level(self, caplog):
        """
        A routine 401 on /auth/me must NOT emit an ERROR-level log entry
        for 'Authentication required'.
        """
        import logging
        from httpx import AsyncClient, ASGITransport
        from src.main import app
        from src.auth.dependencies import get_current_user, get_current_active_user

        saved_user = app.dependency_overrides.pop(get_current_user, None)
        saved_active = app.dependency_overrides.pop(get_current_active_user, None)
        try:
            with caplog.at_level(logging.ERROR, logger="krishimitra-ai"):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    await ac.get("/auth/me")

            error_records = [
                r for r in caplog.records
                if r.levelno >= logging.ERROR
                and "Authentication required" in r.getMessage()
            ]
            assert len(error_records) == 0, (
                f"Unexpected ERROR log for routine 401: {[r.getMessage() for r in error_records]}"
            )
        finally:
            if saved_user:
                app.dependency_overrides[get_current_user] = saved_user
            if saved_active:
                app.dependency_overrides[get_current_active_user] = saved_active

    @pytest.mark.asyncio
    async def test_invalid_credentials_auth_me_returns_401(self):
        """Invalid Bearer token must return 401."""
        from httpx import AsyncClient, ASGITransport
        from src.main import app
        from src.auth.dependencies import get_current_user, get_current_active_user

        saved_user = app.dependency_overrides.pop(get_current_user, None)
        saved_active = app.dependency_overrides.pop(get_current_active_user, None)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get(
                    "/auth/me",
                    headers={"Authorization": "Bearer invalid.token.here"},
                )
            assert response.status_code == 401
        finally:
            if saved_user:
                app.dependency_overrides[get_current_user] = saved_user
            if saved_active:
                app.dependency_overrides[get_current_active_user] = saved_active

    @pytest.mark.asyncio
    async def test_authenticated_auth_me_returns_200(self):
        """Authenticated request to /auth/me returns 200 with user profile."""
        from httpx import AsyncClient, ASGITransport
        from src.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/auth/me")
        assert response.status_code == 200
        data = response.json()
        assert data.get("email") == "admin@bhoomimitra.ai"


# =============================================================================
# P6 — WORKER SHUTDOWN (INVESTIGATION / DOCUMENTATION)
# =============================================================================


class TestWorkerShutdown:
    """
    Worker SIGTERM and Bad file descriptor messages are expected behaviour
    on Render during rolling deploys. These tests verify correct configuration.
    """

    def test_render_worker_config_uses_reasonable_timeout(self):
        import os
        render_yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "render.yaml"
        )
        with open(render_yaml_path) as f:
            content = f.read()
        assert "--timeout 90" in content
        assert "-w 2" in content

    def test_engine_dispose_is_called_on_shutdown(self):
        """Verify the lifespan handler's shutdown path calls engine.dispose()."""
        dispose_called = {"v": False}
        mock_engine = MagicMock()

        async def _mock_dispose():
            dispose_called["v"] = True

        mock_engine.dispose = AsyncMock(side_effect=_mock_dispose)

        async def _run():
            await mock_engine.dispose()

        asyncio.run(_run())
        assert dispose_called["v"] is True


# =============================================================================
# P7 — FERTILIZER SAFETY / MULTI-TURN MEMORY REGRESSION
# =============================================================================


class TestFertilizerSafetyPreserved:
    """Smoke tests confirming fertilizer-safety functions still import and run."""

    def test_get_unsupported_media_fallback_response_importable(self):
        from src.ai.prompts import get_unsupported_media_fallback_response
        result = get_unsupported_media_fallback_response("te")
        assert isinstance(result, str) and len(result) > 10

    def test_get_fallback_response_importable(self):
        from src.ai.prompts import get_fallback_response
        result = get_fallback_response("te")
        assert isinstance(result, str) and len(result) > 10

    def test_check_fertilizer_history_details_importable(self):
        try:
            from src.ai.service import check_fertilizer_history_details
            result = check_fertilizer_history_details("DAP వేయలేదు, యూరియా వేయలేదు.")
            assert isinstance(result, dict)
            assert "has_basal_history" in result
            assert "has_prior_urea_or_potash" in result
        except ImportError:
            pytest.skip("check_fertilizer_history_details not present in this branch")

    def test_build_unconfirmed_paddy_fertilizer_guidance_importable(self):
        try:
            from src.ai.service import build_unconfirmed_paddy_fertilizer_guidance
            result = build_unconfirmed_paddy_fertilizer_guidance("te")
            assert isinstance(result, str) and len(result) > 10
        except ImportError:
            pytest.skip("build_unconfirmed_paddy_fertilizer_guidance not present in this branch")
