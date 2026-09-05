import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from sqlalchemy.exc import IntegrityError

from src.main import app
from src.core.models import Farmer
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import (
    store_incoming_message,
    process_message_pipeline,
)

client = TestClient(app)


@patch("src.gateway.router.get_settings")
def test_webhook_verification_success(mock_get_settings):
    mock_get_settings.return_value.whatsapp_verify_token = "bhoomimitra_verify_2026"
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "bhoomimitra_verify_2026",
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 200
    assert response.text == "115820120"


def test_webhook_verification_failure():
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 403


def test_whatsapp_health_endpoint():
    response = client.get("/webhook/whatsapp/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "whatsapp_configured" in data["data"]
    assert "phone_number_id_configured" in data["data"]
    assert "phone_number_id" in data["data"]
    assert "gemini_configured" in data["data"]


@patch("src.gateway.router.process_message_pipeline")
def test_post_webhook_text_message_success(mock_pipeline):
    mock_pipeline.return_value = None

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1993680168018884",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "1211671805365875"
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Farmer"},
                                    "wa_id": "919876543210"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.HBgLTEST12345",
                                    "timestamp": "1700000000",
                                    "text": {"body": "Hi BhoomiMitra test"},
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["messages_queued"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TESTS: Duplicate Message & Webhook Concurrency Race Protection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_message_pipeline_duplicate_stage1_skips_ai():
    """Verify that if a message_id is already in DB, Stage 1 immediately skips the pipeline."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DUPLICATE_STAGE1_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Where to buy urea?",
    )

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=True) as mock_dup, \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_farmer, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_dup.assert_awaited_once()
        mock_farmer.assert_not_called()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_pipeline_concurrent_duplicate_db_constraint_aborts():
    """
    Verify that if two requests race past Stage 1, the unique DB constraint in Stage 4
    rejects the second insert and exits immediately without calling Gemini or sending a reply.
    """
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.CONCURRENT_RACE_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Where to buy urea?",
    )

    mock_farmer_obj = Farmer(id=uuid.uuid4(), phone_number="919876543210")

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer_obj), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=None) as mock_store, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_store.assert_awaited_once()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_store_incoming_message_catches_integrity_error():
    """Verify that store_incoming_message rolls back and returns None on IntegrityError."""
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("duplicate key value violates unique constraint", None, None)
    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210")
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.INTEGRITY_ERR_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Test integrity",
    )

    result = await store_incoming_message(mock_db, mock_farmer, parsed)
    assert result is None
    mock_db.rollback.assert_awaited_once()


@patch("src.gateway.router.process_message_pipeline")
def test_post_webhook_duplicate_messages_in_payload_queued_once(mock_pipeline):
    """Verify that if the same message_id appears multiple times in a webhook payload, it is only queued once."""
    mock_pipeline.return_value = None

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1993680168018884",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "1211671805365875"
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Farmer"},
                                    "wa_id": "919876543210"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.DUPLICATE_BATCH_ID_999",
                                    "timestamp": "1700000000",
                                    "text": {"body": "First instance"},
                                    "type": "text"
                                },
                                {
                                    "from": "919876543210",
                                    "id": "wamid.DUPLICATE_BATCH_ID_999",
                                    "timestamp": "1700000000",
                                    "text": {"body": "Second duplicate instance"},
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["messages_queued"] == 1


@pytest.mark.asyncio
async def test_process_message_pipeline_in_flight_lock_aborts_concurrent_run():
    """Verify that if a message ID is already in-flight, a second concurrent task aborts at Stage 0."""
    from src.gateway.service import _IN_FLIGHT_MESSAGE_IDS

    msg_id = "wamid.CONCURRENT_IN_FLIGHT_TEST"
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id=msg_id,
        timestamp="1700000000",
        message_type="text",
        text_content="Multi intent query",
    )

    _IN_FLIGHT_MESSAGE_IDS.add(msg_id)
    try:
        with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock) as mock_dup, \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_farmer, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

            await process_message_pipeline(parsed)

            mock_dup.assert_not_called()
            mock_farmer.assert_not_called()
            mock_send.assert_not_called()
    finally:
        _IN_FLIGHT_MESSAGE_IDS.discard(msg_id)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 HARDENING TESTS: GET Verification, App Secret & Startup Validation
# ─────────────────────────────────────────────────────────────────────────────

import hmac
import hashlib


@patch("src.gateway.router.get_settings")
def test_webhook_verification_empty_server_token_fails(mock_get_settings):
    """Verify that if WHATSAPP_VERIFY_TOKEN is unset/empty on server, verification fails with 403."""
    mock_get_settings.return_value.whatsapp_verify_token = ""
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 403


@patch("src.gateway.router.get_settings")
def test_webhook_verification_missing_hub_token_fails(mock_get_settings):
    """Verify that if hub.verify_token is missing or empty, verification fails with 403."""
    mock_get_settings.return_value.whatsapp_verify_token = "valid_token_2026"
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": None,
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 403


@pytest.mark.asyncio
@patch("src.gateway.security.get_settings")
async def test_webhook_signature_missing_secret_in_production_fails(mock_get_settings):
    """Verify that missing WHATSAPP_APP_SECRET in production causes verification rejection (False)."""
    from src.gateway.security import verify_webhook_signature
    mock_settings = mock_get_settings.return_value
    mock_settings.whatsapp_app_secret = ""
    mock_settings.is_production = True
    mock_request = AsyncMock()
    result = await verify_webhook_signature(mock_request, b'{"test": "payload"}')
    assert result is False


@pytest.mark.asyncio
@patch("src.gateway.security.get_settings")
async def test_webhook_signature_missing_secret_in_dev_allowed(mock_get_settings):
    """Verify that missing WHATSAPP_APP_SECRET in development is tolerated (returns True)."""
    from src.gateway.security import verify_webhook_signature
    mock_settings = mock_get_settings.return_value
    mock_settings.whatsapp_app_secret = ""
    mock_settings.is_production = False
    mock_request = AsyncMock()
    result = await verify_webhook_signature(mock_request, b'{"test": "payload"}')
    assert result is True


@pytest.mark.asyncio
@patch("src.gateway.security.get_settings")
async def test_webhook_signature_valid_hmac_verified(mock_get_settings):
    """Verify that a valid HMAC-SHA256 signature passes verification."""
    from src.gateway.security import verify_webhook_signature
    secret = "super_secret_key_123"
    body = b'{"test": "valid_body"}'
    expected_hash = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    mock_settings = mock_get_settings.return_value
    mock_settings.whatsapp_app_secret = secret
    mock_settings.is_production = True

    mock_request = AsyncMock()
    mock_request.headers = {"x-hub-signature-256": f"sha256={expected_hash}"}
    mock_request.client.host = "127.0.0.1"

    result = await verify_webhook_signature(mock_request, body)
    assert result is True


@pytest.mark.asyncio
@patch("src.gateway.security.get_settings")
async def test_webhook_signature_invalid_hmac_rejected(mock_get_settings):
    """Verify that an invalid HMAC-SHA256 signature is rejected."""
    from src.gateway.security import verify_webhook_signature
    secret = "super_secret_key_123"
    body = b'{"test": "valid_body"}'

    mock_settings = mock_get_settings.return_value
    mock_settings.whatsapp_app_secret = secret
    mock_settings.is_production = True

    mock_request = AsyncMock()
    mock_request.headers = {"x-hub-signature-256": "sha256=invalid_hash_value"}
    mock_request.client.host = "127.0.0.1"

    result = await verify_webhook_signature(mock_request, body)
    assert result is False


@patch("src.main.get_settings")
def test_validate_production_settings_raises_if_missing_vars(mock_get_settings):
    """Verify that validate_production_settings raises RuntimeError if any critical WhatsApp secret is missing."""
    from src.main import validate_production_settings
    mock_settings = mock_get_settings.return_value
    mock_settings.is_production = True
    mock_settings.whatsapp_api_token = ""
    mock_settings.whatsapp_phone_number_id = "123456"
    mock_settings.whatsapp_verify_token = "token"
    mock_settings.whatsapp_app_secret = "secret"

    with pytest.raises(RuntimeError, match="Missing required WhatsApp settings for production: WHATSAPP_API_TOKEN"):
        validate_production_settings()


@patch("src.main.get_settings")
def test_validate_production_settings_passes_if_all_present(mock_get_settings):
    """Verify that validate_production_settings succeeds when all critical WhatsApp variables are present in production."""
    from src.main import validate_production_settings
    mock_settings = mock_get_settings.return_value
    mock_settings.is_production = True
    mock_settings.whatsapp_api_token = "EAA_test_token"
    mock_settings.whatsapp_phone_number_id = "123456789"
    mock_settings.whatsapp_verify_token = "bhoomimitra_verify_2026"
    mock_settings.whatsapp_app_secret = "test_app_secret_999"

    # Should succeed without error
    validate_production_settings()


@patch("src.main.get_settings")
def test_validate_production_settings_skipped_in_development(mock_get_settings):
    """Verify that validate_production_settings does nothing in development mode even if variables are unset."""
    from src.main import validate_production_settings
    mock_settings = mock_get_settings.return_value
    mock_settings.is_production = False
    mock_settings.whatsapp_api_token = ""
    mock_settings.whatsapp_phone_number_id = ""
    mock_settings.whatsapp_verify_token = ""
    mock_settings.whatsapp_app_secret = ""

    # Should not raise in development
    validate_production_settings()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 TESTS: Pipeline Stability, Failure Feedback & STT Configuration
# ─────────────────────────────────────────────────────────────────────────────

from src.core.models import Conversation
from src.ai.prompts import (
    VOICE_FAILURE_RESPONSE_TE,
    IMAGE_FAILURE_RESPONSE_TE,
    FALLBACK_RESPONSE_TE,
)


@pytest.mark.asyncio
async def test_voice_transcription_failure_sends_fallback_response():
    """Verify that when audio transcription fails, the farmer receives the localized voice fallback message."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.VOICE_FAIL_TEST_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_999",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message=None,
        user_message_type="audio"
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_VOICE_FAIL") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        mock_lang_svc.return_value.transcribe_audio.side_effect = Exception("Google STT Connection Error")

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=VOICE_FAILURE_RESPONSE_TE,
        )


@pytest.mark.asyncio
async def test_image_download_failure_sends_fallback_response():
    """Verify that when image download fails, the farmer receives the localized image fallback message."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.IMG_FAIL_TEST_01",
        timestamp="1700000000",
        message_type="image",
        media_id="img_media_888",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message=None,
        user_message_type="image"
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=None), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_IMG_FAIL") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=IMAGE_FAILURE_RESPONSE_TE,
        )


@pytest.mark.asyncio
async def test_ai_failure_empty_response_sends_fallback_response():
    """Verify that when AI returns empty text or raises an exception, the farmer receives the general fallback message."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.AI_FAIL_TEST_01",
        timestamp="1700000000",
        message_type="text",
        text_content="పత్తిలో తెగులు వచ్చింది",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message=parsed.text_content,
        user_message_type="text"
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, side_effect=Exception("Gemini API Timeout")), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_AI_FAIL") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text=FALLBACK_RESPONSE_TE,
        )


@pytest.mark.asyncio
async def test_ogg_opus_configuration_does_not_force_16000_hz():
    """Verify that Speech-to-Text configuration for OGG_OPUS does not force a 16000 Hz sample rate."""
    from src.language.service import LanguageService
    from google.cloud import speech

    service = LanguageService()
    captured_config = []

    mock_speech_client = AsyncMock()
    mock_response = speech.RecognizeResponse(
        results=[
            speech.SpeechRecognitionResult(
                alternatives=[
                    speech.SpeechRecognitionAlternative(transcript="పత్తి పంట రక్షణ", confidence=0.95)
                ]
            )
        ]
    )

    async def fake_recognize(config, audio):
        captured_config.append(config)
        return mock_response

    mock_speech_client.recognize = fake_recognize

    with patch.object(LanguageService, "google_client", new_callable=lambda: property(lambda self: mock_speech_client)):
        res = await service._transcribe_with_google(b"fake_ogg_bytes", "audio/ogg")
        assert res.transcription_text == "పత్తి పంట రక్షణ"
        assert len(captured_config) == 1
        cfg = captured_config[0]
        # Encoding must be OGG_OPUS
        assert cfg.encoding == speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        # sample_rate_hertz must not be 16000 (default unset in protobuf is 0)
        assert cfg.sample_rate_hertz == 0


@pytest.mark.asyncio
async def test_incoming_message_stored_before_stt_gemini_processing():
    """Verify that incoming Conversation is stored in the database BEFORE calling STT or Gemini."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.ORDER_EXEC_TEST_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_111",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message=None,
        user_message_type="audio"
    )

    execution_order = []

    async def mock_store(db, farmer, message):
        execution_order.append("store_db")
        return mock_conv

    async def mock_download(media_id):
        execution_order.append("download_media")
        return (b"bytes", "audio/ogg")

    mock_transcription = AsyncMock()
    mock_transcription.transcription_text = "టమాటా ధర ఎంత"

    async def mock_transcribe(audio_bytes, mime_type):
        execution_order.append("stt_transcribe")
        return mock_transcription

    async def mock_ai(db, farmer, conv):
        execution_order.append("gemini_ai")
        return "టమాటా ధర వివరాలు..."

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", side_effect=mock_store), \
         patch("src.gateway.service.download_media_bytes", side_effect=mock_download), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", side_effect=mock_ai), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_OK"), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        mock_lang_svc.return_value.transcribe_audio = mock_transcribe

        await process_message_pipeline(parsed)

        # Confirm DB store executed BEFORE external media download, STT, and AI
        assert execution_order == ["store_db", "download_media", "stt_transcribe", "gemini_ai"]


@pytest.mark.asyncio
async def test_duplicate_webhook_delivery_does_not_invoke_stt_twice():
    """Verify that when a duplicate webhook arrives (already stored in DB), STT and AI are skipped completely."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DUPLICATE_STT_TEST_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_222",
    )

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=True) as mock_dup, \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock) as mock_download, \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_dup.assert_awaited_once()
        mock_download.assert_not_called()
        mock_lang_svc.assert_not_called()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_existing_successful_text_message_processing_still_works():
    """Verify that standard successful text message flow continues to work seamlessly."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.SUCCESSFUL_TEXT_01",
        timestamp="1700000000",
        message_type="text",
        text_content="యూరియా ఎక్కడ దొరుకుతుంది?",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
    mock_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=mock_farmer.id,
        message_id=parsed.message_id,
        user_message=parsed.text_content,
        user_message_type="text"
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="సమీప దుకాణాల్లో యూరియా అందుబాటులో ఉంది."), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_OK") as mock_send, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        mock_send.assert_awaited_once_with(
            to_phone="919876543210",
            message_text="సమీప దుకాణాల్లో యూరియా అందుబాటులో ఉంది.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 REGRESSION TESTS: Fast Acknowledgement & Multi-Layer Duplicate Defense
# ─────────────────────────────────────────────────────────────────────────────

def test_post_webhook_acknowledges_http_200_immediately():
    """Verify that Meta webhook POST returns HTTP 200 OK immediately and queues background task."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1993680168018884",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "1211671805365875"
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Rao"},
                                    "wa_id": "919876543210"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.FAST_ACK_TEST_123",
                                    "timestamp": "1700000000",
                                    "text": {"body": "రైతు బంధు పథకం వివరాలు"},
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }

    with patch("src.gateway.router.process_message_pipeline") as mock_pipeline:
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["messages_queued"] == 1


@pytest.mark.asyncio
async def test_duplicate_webhook_in_flight_logged_and_aborted():
    """Verify that in-flight duplicate delivery is halted at Stage 0 before DB or AI processing."""
    from src.gateway.service import _IN_FLIGHT_MESSAGE_IDS

    msg_id = "wamid.IN_FLIGHT_REGRESSION_TEST"
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id=msg_id,
        timestamp="1700000000",
        message_type="text",
        text_content="Price of tomato?",
    )

    _IN_FLIGHT_MESSAGE_IDS.add(msg_id)
    try:
        with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock) as mock_dup, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

            await process_message_pipeline(parsed)

            # Neither DB, AI, nor outbound send should be triggered
            mock_dup.assert_not_called()
            mock_ai.assert_not_called()
            mock_send.assert_not_called()
    finally:
        _IN_FLIGHT_MESSAGE_IDS.discard(msg_id)


@pytest.mark.asyncio
async def test_duplicate_webhook_db_exists_logged_and_aborted():
    """Verify that duplicate message already committed to DB is halted at Stage 1 before AI or outbound sending."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DB_EXISTS_REGRESSION_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Price of cotton?",
    )

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=True) as mock_dup, \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_farmer, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_dup.assert_awaited_once()
        mock_farmer.assert_not_called()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_webhook_concurrent_db_collision_logged_and_rolled_back():
    """Verify that a concurrent DB collision during store_incoming_message safely rolls back and halts the pipeline."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DB_COLLISION_REGRESSION_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Weather report today",
    )

    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=None) as mock_store, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_store.assert_awaited_once()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()
