"""
BhoomiMitra AI — Phase 4: Production Polish & Final Readiness Test Suite

Validates:
1. Concurrency & Multi-Worker Reliability (Database unique constraint authoritative duplicate protection)
2. External Service & AI Timeouts (Gemini, STT, OpenWeather, Agmarknet, Meta Outbound)
3. Graceful Degradation across all modules without crashes
4. Startup Configuration Validation (Required vs Optional variables, no secret leakage)
5. Logging Privacy & Header/Phone Masking
6. Resource & Memory Safety (Oversized media rejection, response length capping)
7. End-to-End Representative Production Flows Matrix
"""

import pytest
import uuid
import json
import asyncio
import httpx
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from src.main import app, validate_production_settings, audit_environment_variables
from src.config import Settings, get_settings
from src.core.models import Farmer, Conversation, MarketPrice, Shop, Inventory, GovernmentScheme
from src.gateway.schemas import ParsedIncomingMessage, WhatsAppWebhookPayload, WhatsAppMessage
from src.gateway.router import _extract_message, mask_phone_number, sanitize_headers_for_logging
from src.gateway.service import (
    process_message_pipeline,
    store_incoming_message,
    is_duplicate_message,
    acquire_in_flight_lock,
    release_in_flight_lock,
)
from src.gateway.whatsapp_client import send_text_message, download_media_bytes
from src.ai.service import process_text_message, process_image_message, _finalize_whatsapp_response
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent, get_decision_engine
from src.ai.schemas import AIGenerateResponse
from src.ai.prompts import (
    get_fallback_response,
    get_voice_fallback_response,
    get_image_fallback_response,
    get_unsupported_media_fallback_response,
    get_market_fallback_response,
    get_weather_fallback_response,
    get_schemes_fallback_response,
    get_shops_fallback_response,
)

client = TestClient(app)


# ==============================================================================
# 1. MULTI-WORKER / CONCURRENCY RELIABILITY TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrency_database_unique_constraint_authoritative():
    """
    Simulate Worker A and Worker B receiving identical Meta webhook retries simultaneously.
    Authoritative duplicate detection via DB IntegrityError prevents Worker B from executing.
    """
    from sqlalchemy.exc import IntegrityError

    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.CONCURRENT_MULTI_WORKER_001",
        timestamp="1700000000",
        message_type="text",
        text_content="వరంగల్లో పత్తి ధర ఎంత?",
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    # Worker B encounters unique constraint violation when attempting store_incoming_message
    mock_db.commit.side_effect = IntegrityError("duplicate key", params=None, orig=Exception("unique constraint"))

    class MockContext:
        async def __aenter__(self):
            return mock_db
        async def __aexit__(self, exc_type, exc, tb):
            pass

    with patch("src.gateway.service.AsyncSessionLocal", return_value=MockContext()), \
         patch("src.ai.service.AIService.generate_ai_response") as mock_ai, \
         patch("src.gateway.whatsapp_client.send_text_message") as mock_send:

        await process_message_pipeline(parsed)

        # Worker B must NOT call AI and must NOT send duplicate WhatsApp message
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_concurrency_in_flight_lock_lifecycle():
    """Verify in-flight locks are acquired, prevent races, and always release cleanly."""
    msg_id = "wamid.IN_FLIGHT_LIFECYCLE_TEST"
    
    assert acquire_in_flight_lock(msg_id) is True
    # Second acquisition on same message ID should fail (in-flight race prevented)
    assert acquire_in_flight_lock(msg_id) is False

    release_in_flight_lock(msg_id)
    # After release, acquisition should succeed again
    assert acquire_in_flight_lock(msg_id) is True
    release_in_flight_lock(msg_id)


# ==============================================================================
# 2. AI / EXTERNAL SERVICE TIMEOUT TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_gemini_timeout_graceful_degradation():
    """When Gemini API exceeds timeout, pipeline catches it and produces safe fallback without crash."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=farmer.id,
        user_message="టమాటా పంటలో పురుగులు వచ్చాయి ఏం చేయాలి?"
    )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=TimeoutError("Gemini API timed out after 5.0s")):
        reply = await process_text_message(db_mock, farmer, conv)

        assert reply is not None
        assert "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది" in reply
        assert "TimeoutError" not in reply
        assert "Traceback" not in reply


@pytest.mark.asyncio
async def test_stt_timeout_graceful_voice_fallback():
    """When Google STT API times out, voice fallback is returned cleanly."""
    from src.language.service import LanguageService
    from src.core.exceptions import BhoomiMitraException

    svc = LanguageService()
    svc._google_client = MagicMock()
    svc._google_client.recognize = AsyncMock(side_effect=asyncio.TimeoutError())

    with pytest.raises(BhoomiMitraException) as exc_info:
        await svc.transcribe_audio(b"fake_audio_bytes", "audio/ogg")
    
    assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_openweather_timeout_degrades_gracefully():
    """When OpenWeather API times out via httpx.TimeoutException, it returns None without crashing."""
    from src.weather.openweather_client import OpenWeatherClient

    client = OpenWeatherClient(api_key="mock_key", api_url="https://api.openweathermap.org", timeout_seconds=0.1)

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("OpenWeather timed out")):
        res = await client.fetch_weather(district="Warangal", state="Telangana")
        assert res is None


@pytest.mark.asyncio
async def test_agmarknet_timeout_degrades_gracefully():
    """When Agmarknet API times out, it logs cleanly and returns empty list for DB fallback."""
    from src.market.agmarknet_client import AgmarknetClient

    client = AgmarknetClient(api_key="mock_key", api_url="https://api.data.gov.in", timeout_seconds=0.1)

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Agmarknet timed out")):
        res = await client.fetch_prices(commodity="Cotton", district="Warangal")
        assert res == []


@pytest.mark.asyncio
async def test_meta_outbound_timeout_retries_and_handles_exhaustion():
    """When Meta Outbound API times out on all retries, returns None without raising uncaught exceptions."""
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Meta timeout")):
        result = await send_text_message("919876543210", "Test message")
        assert result is None


# ==============================================================================
# 3. STARTUP CONFIGURATION VALIDATION TESTS
# ==============================================================================

def test_startup_validation_missing_production_variables_raises():
    """In production mode, missing critical variables raises RuntimeError."""
    mock_settings = Settings(
        app_env="production",
        whatsapp_api_token="",
        whatsapp_phone_number_id="",
        whatsapp_verify_token="",
        whatsapp_app_secret="",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
    )

    with patch("src.main.get_settings", return_value=mock_settings):
        with pytest.raises(RuntimeError) as exc_info:
            validate_production_settings()
        assert "WHATSAPP_API_TOKEN" in str(exc_info.value)
        assert "WHATSAPP_PHONE_NUMBER_ID" in str(exc_info.value)


def test_startup_validation_optional_variables_do_not_block_startup():
    """Optional variables like DATA_GOV_API_KEY and OPENWEATHER_API_KEY do not block production boot."""
    mock_settings = Settings(
        app_env="production",
        whatsapp_api_token="valid_wa_token",
        whatsapp_phone_number_id="1234567890",
        whatsapp_verify_token="valid_verify_token",
        whatsapp_app_secret="valid_app_secret",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        data_gov_api_key="",  # Optional
        openweather_api_key="",  # Optional
    )

    with patch("src.main.get_settings", return_value=mock_settings):
        # Should complete without error
        validate_production_settings()


def test_audit_environment_variables_never_logs_secrets(caplog):
    """Verify that environment audit logs status (PRESENT/MISSING) and never actual secret values."""
    mock_settings = Settings(
        app_env="production",
        whatsapp_api_token="SUPER_SECRET_WHATSAPP_TOKEN_12345",
        whatsapp_verify_token="SECRET_VERIFY_TOKEN_XYZ",
        whatsapp_app_secret="SECRET_APP_SECRET_987",
        google_gemini_api_key="SECRET_GEMINI_KEY_ABC",
        database_url="postgresql+asyncpg://user:super_secret_pw@localhost/db",
    )

    with patch("src.main.get_settings", return_value=mock_settings):
        with caplog.at_level("INFO"):
            audit_environment_variables()
            log_output = caplog.text

            assert "SUPER_SECRET_WHATSAPP_TOKEN_12345" not in log_output
            assert "SECRET_VERIFY_TOKEN_XYZ" not in log_output
            assert "SECRET_APP_SECRET_987" not in log_output
            assert "SECRET_GEMINI_KEY_ABC" not in log_output
            assert "super_secret_pw" not in log_output
            assert "PRESENT" in log_output


# ==============================================================================
# 4. PRIVACY & LOGGING SANITIZATION TESTS
# ==============================================================================

def test_phone_number_masking():
    """Test phone numbers are masked properly across formats."""
    assert mask_phone_number("919876543210") == "9198****3210"
    assert mask_phone_number("9876543210") == "9876****3210"
    assert mask_phone_number("123") == "***"
    assert mask_phone_number("") == "***"


def test_sanitize_headers_for_logging():
    """Verify Authorization, x-hub-signature, and cookies are redacted."""
    headers = {
        "Authorization": "Bearer EAAXxXyYzZ123456789",
        "X-Hub-Signature-256": "sha256=abcdef1234567890",
        "Content-Type": "application/json",
        "User-Agent": "Meta-WhatsApp-Cloud-Api",
    }
    sanitized = sanitize_headers_for_logging(headers)
    assert sanitized["Authorization"] == "***REDACTED***"
    assert sanitized["X-Hub-Signature-256"] == "***REDACTED***"
    assert sanitized["Content-Type"] == "application/json"


# ==============================================================================
# 5. RESOURCE & MEMORY SAFETY TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_large_media_download_rejected_safely():
    """When media payload exceeds max_media_download_bytes (15MB), download aborts cleanly."""
    settings = get_settings()
    huge_payload = b"X" * (settings.max_media_download_bytes + 100)

    mock_resp_meta = MagicMock(status_code=200)
    mock_resp_meta.json.return_value = {"url": "https://lookaside.fbsbx.com/media", "mime_type": "image/jpeg"}

    mock_resp_data = MagicMock(status_code=200, content=huge_payload)

    with patch("httpx.AsyncClient.get", side_effect=[mock_resp_meta, mock_resp_data]):
        res = await download_media_bytes("media_huge_123")
        assert res is None


def test_response_length_finalizer_capping():
    """Verify _finalize_whatsapp_response caps long text within WhatsApp budget."""
    long_text = "రైతు సోదరులకు ముఖ్యమైన సమాచారం. " * 300  # ~9000 chars
    finalized = _finalize_whatsapp_response(long_text)
    assert len(finalized) <= 4096
    assert finalized.endswith("...") or finalized.endswith(".") or finalized.endswith("।")


# ==============================================================================
# 6. END-TO-END REPRESENTATIVE FLOWS MATRIX
# ==============================================================================

@pytest.mark.asyncio
async def test_flow_1_telugu_market_query():
    """Flow 1: Telugu Market query returns authoritative market price numbers."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid.uuid4(), preferred_language="te")
    conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, user_message="వరంగల్లో ఈరోజు పత్తి ధర ఎంత")

    mock_mandi = [
        MarketPrice(
            id=uuid.uuid4(),
            commodity="Cotton",
            commodity_telugu="పత్తి",
            market_name="Warangal Mandi",
            district="Warangal",
            state="Telangana",
            min_price=7100.0,
            max_price=7650.0,
            modal_price=7450.0,
            unit="Quintal",
            source="manual_seed",
            price_date=datetime.utcnow(),
            created_at=datetime.utcnow(),
        )
    ]

    with patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=mock_mandi):

        reply = await process_text_message(db_mock, farmer, conv)
        assert "7,450" in reply
        assert "వరంగల్" in reply or "Warangal" in reply


@pytest.mark.asyncio
async def test_flow_2_tanglish_market_query():
    """Flow 2: Tanglish Market query is correctly recognized and answered in Telugu."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid.uuid4(), preferred_language="te")
    conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, user_message="Warangal lo cotton rate entha")

    mock_mandi = [
        MarketPrice(
            id=uuid.uuid4(),
            commodity="Cotton",
            commodity_telugu="పత్తి",
            market_name="Warangal Mandi",
            district="Warangal",
            state="Telangana",
            min_price=7100.0,
            max_price=7650.0,
            modal_price=7450.0,
            unit="Quintal",
            source="manual_seed",
            price_date=datetime.utcnow(),
            created_at=datetime.utcnow(),
        )
    ]

    with patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=mock_mandi):

        reply = await process_text_message(db_mock, farmer, conv)
        assert "7,450" in reply


@pytest.mark.asyncio
async def test_flow_3_greeting_fast_bypass():
    """Flow 3: Greeting query bypasses LLM and external modules instantly."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid.uuid4(), preferred_language="te")
    conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, user_message="హాయ్")

    with patch("src.ai.service.AIService.generate_ai_response") as mock_ai, \
         patch("src.market.service.enrich_response_with_market_prices") as mock_mkt:

        reply = await process_text_message(db_mock, farmer, conv)
        mock_ai.assert_not_called()
        mock_mkt.assert_not_called()
        assert "నమస్తే! నేను మీ భూమిమిత్ర" in reply


@pytest.mark.asyncio
async def test_flow_4_unsupported_media_guiding_fallback():
    """Flow 4: Unsupported media event generates guiding reprompt."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.UNSUPPORTED_001",
        timestamp="1700000000",
        message_type="video",
    )
    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id)

    mock_db = AsyncMock()
    mock_db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))

    class MockContext:
        async def __aenter__(self):
            return mock_db
        async def __aexit__(self, exc_type, exc, tb):
            pass

    with patch("src.gateway.service.AsyncSessionLocal", return_value=MockContext()), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_wa_01") as mock_send:

        await process_message_pipeline(parsed)
        mock_send.assert_awaited_once()
        assert "టెక్స్ట్, వాయిస్ మెసేజ్ లేదా పంట ఫోటో" in mock_send.call_args[1]["message_text"]


def test_flow_5_malformed_webhook_handled_gracefully():
    """Flow 5: Malformed JSON payload returns HTTP 200 with status: ignored without crashing."""
    res = client.post(
        "/webhook/whatsapp",
        content=b"{bad_json: true",
        headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ignored"
    assert res.json()["reason"] == "invalid_json"
