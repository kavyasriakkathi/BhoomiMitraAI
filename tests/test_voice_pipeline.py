"""
BhoomiMitra AI — Voice Pipeline Test Suite

Tests the complete WhatsApp voice pipeline:
- STT configuration (alternative_language_codes limit <= 3, MIME mapping)
- Audio validation (short/empty audio rejection)
- WhatsApp webhook extraction (audio and voice payload types)
- Telugu voice -> STT -> Language detection -> Intent -> RAG -> Safety -> Gemini -> Telugu response
- Unclear/corrupt audio fallback handling
- Preservation of fertilizer/pesticide safety gate for voice queries
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.language.service import LanguageService, TranscriptionResponse
from src.language.detector import detect_language
from src.core.exceptions import BhoomiMitraException
from src.core.models import Farmer, Conversation
from src.gateway.schemas import (
    WhatsAppWebhookPayload,
    WhatsAppEntry,
    WhatsAppChange,
    WhatsAppValue,
    WhatsAppMessage,
    WhatsAppAudioPayload,
    ParsedIncomingMessage,
)
from src.gateway.router import _extract_message
from src.gateway.service import process_message_pipeline
from src.ai.prompts import VOICE_FAILURE_RESPONSES


# =============================================================================
# 1. Google Speech-to-Text Configuration & Audio Validation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stt_alternative_language_codes_limit_compliance():
    """Verify that Google STT configuration never exceeds the strict 3-language limit for alternative codes."""
    svc = LanguageService()
    svc.settings.stt_provider = "google"
    svc.settings.stt_default_language = "te-IN"

    captured_config = {}

    def mock_recognize(config, audio):
        captured_config["language_code"] = config.language_code
        captured_config["alternative_language_codes"] = list(config.alternative_language_codes)
        captured_config["encoding"] = config.encoding

        mock_resp = MagicMock()
        mock_result = MagicMock()
        mock_result.language_code = "te-IN"
        mock_alt = MagicMock()
        mock_alt.transcript = "నమస్కారం"
        mock_alt.confidence = 0.95
        mock_result.alternatives = [mock_alt]
        mock_resp.results = [mock_result]
        return mock_resp

    mock_client = MagicMock()
    mock_client.recognize = mock_recognize
    svc._google_client = mock_client

    dummy_audio = b"\x00" * 256
    res = await svc.transcribe_audio(dummy_audio, "audio/ogg; codecs=opus")

    assert res.transcription_text == "నమస్కారం"
    assert captured_config["language_code"] == "te-IN"
    # Google Cloud STT v1 allows at most 3 alternative languages
    assert len(captured_config["alternative_language_codes"]) <= 3
    assert "te-IN" not in captured_config["alternative_language_codes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type,expected_encoding_name", [
    ("audio/ogg; codecs=opus", "OGG_OPUS"),
    ("audio/ogg", "OGG_OPUS"),
    ("audio/opus", "OGG_OPUS"),
    ("audio/mp3", "MP3"),
    ("audio/mpeg", "MP3"),
    ("audio/amr", "AMR"),
    ("audio/amr-wb", "AMR_WB"),
    ("audio/wav", "LINEAR16"),
    ("audio/x-wav", "LINEAR16"),
    ("audio/mp4", "ENCODING_UNSPECIFIED"),
    ("audio/aac", "ENCODING_UNSPECIFIED"),
])
async def test_stt_mime_type_encoding_mapping(mime_type, expected_encoding_name):
    """Verify that incoming audio MIME types map dynamically to the correct SpeechRecognition AudioEncoding."""
    from google.cloud import speech_v1 as speech

    svc = LanguageService()
    svc.settings.stt_provider = "google"
    svc.settings.stt_default_language = "te-IN"

    captured_encoding = None

    def mock_recognize(config, audio):
        nonlocal captured_encoding
        captured_encoding = config.encoding
        mock_resp = MagicMock()
        mock_result = MagicMock()
        mock_alt = MagicMock(transcript="పంట", confidence=0.9)
        mock_result.alternatives = [mock_alt]
        mock_resp.results = [mock_result]
        return mock_resp

    mock_client = MagicMock()
    mock_client.recognize = mock_recognize
    svc._google_client = mock_client

    dummy_audio = b"\x00" * 200
    await svc.transcribe_audio(dummy_audio, mime_type)

    expected_enum_val = getattr(speech.RecognitionConfig.AudioEncoding, expected_encoding_name)
    assert captured_encoding == expected_enum_val


@pytest.mark.asyncio
async def test_stt_rejects_empty_audio_payload():
    """Verify that empty audio payloads are rejected with a 400 Bad Request status code."""
    svc = LanguageService()

    with pytest.raises(BhoomiMitraException) as exc_info:
        await svc.transcribe_audio(b"", "audio/ogg")
    assert exc_info.value.status_code == 400


# =============================================================================
# 2. Webhook Extraction Tests for Audio and Voice
# =============================================================================

def test_webhook_message_extraction_audio_and_voice():
    """Verify that incoming WhatsApp webhook messages of type 'audio' or 'voice' are correctly parsed."""
    audio_msg = WhatsAppMessage(
        **{"from": "919876543210"},
        id="wamid.AUDIO_TEST_01",
        timestamp="1700000000",
        type="audio",
        audio=WhatsAppAudioPayload(id="media_audio_123", mime_type="audio/ogg; codecs=opus"),
    )
    parsed_audio = _extract_message(audio_msg, sender_name="Ramesh")
    assert parsed_audio is not None
    assert parsed_audio.message_type == "audio"
    assert parsed_audio.media_id == "media_audio_123"
    assert parsed_audio.media_mime_type == "audio/ogg; codecs=opus"

    voice_msg = WhatsAppMessage(
        **{"from": "919876543210"},
        id="wamid.VOICE_TEST_01",
        timestamp="1700000001",
        type="voice",
        voice=WhatsAppAudioPayload(id="media_voice_456", mime_type="audio/ogg"),
    )
    parsed_voice = _extract_message(voice_msg, sender_name="Suresh")
    assert parsed_voice is not None
    assert parsed_voice.message_type == "audio"
    assert parsed_voice.media_id == "media_voice_456"


# =============================================================================
# 3. End-to-End Telugu Voice Flow Tests
# =============================================================================

@pytest.mark.asyncio
async def test_telugu_voice_to_market_price_flow():
    """
    Test complete flow:
    Telugu audio -> STT transcribes 'వరంగల్ లో పత్తి ధర ఎంత?' ->
    Language detected as 'te' -> Market Intent detected ->
    Market service queried -> Localized Telugu WhatsApp response sent.
    """
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.TELUGU_VOICE_MKT_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="media_telugu_voice_mkt",
    )

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.phone_number = "919876543210"
    mock_farmer.preferred_language = "te"
    mock_farmer.district = "Warangal"
    mock_farmer.state = "Telangana"

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    mock_transcription = TranscriptionResponse(
        transcription_text="వరంగల్ లో పత్తి ధర ఎంత?",
        detected_language="te",
        confidence=0.96,
        provider_used="google",
    )

    sent_message_text = None

    async def mock_send(to_phone, message_text):
        nonlocal sent_message_text
        sent_message_text = message_text
        return "wamid.OUT_TELUGU_VOICE_OK"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"\x00" * 300, "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.send_text_message", side_effect=mock_send), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        mock_lang_svc.return_value.transcribe_audio = AsyncMock(return_value=mock_transcription)

        await process_message_pipeline(parsed)

        assert sent_message_text is not None
        # Must be in Telugu and contain market information
        assert "ధర" in sent_message_text or "వరంగల్" in sent_message_text or "పత్తి" in sent_message_text or "క్వింటాల్" in sent_message_text


@pytest.mark.asyncio
async def test_telugu_voice_unclear_audio_returns_safe_fallback():
    """Verify that when voice transcription fails or audio is corrupted, localized Telugu fallback is returned."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.UNCLEAR_VOICE_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="media_unclear_01",
    )

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.phone_number = "919876543210"
    mock_farmer.preferred_language = "te"

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    sent_message_text = None

    async def mock_send(to_phone, message_text):
        nonlocal sent_message_text
        sent_message_text = message_text
        return "wamid.OUT_UNCLEAR_OK"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"\x00" * 300, "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.send_text_message", side_effect=mock_send), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        # STT fails due to unclear audio
        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            side_effect=BhoomiMitraException("Audio is unclear", status_code=422)
        )

        await process_message_pipeline(parsed)

        assert sent_message_text is not None
        # Exactly matches the safe Telugu voice fallback
        assert sent_message_text == VOICE_FAILURE_RESPONSES["te"]


@pytest.mark.asyncio
async def test_telugu_voice_empty_transcription_returns_safe_fallback():
    """Verify that when STT returns an empty/whitespace transcription, localized Telugu fallback is returned."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.EMPTY_VOICE_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="media_empty_01",
    )

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.phone_number = "919876543210"
    mock_farmer.preferred_language = "te"

    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    sent_message_text = None

    async def mock_send(to_phone, message_text):
        nonlocal sent_message_text
        sent_message_text = message_text
        return "wamid.OUT_EMPTY_OK"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"\x00" * 300, "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.send_text_message", side_effect=mock_send), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            side_effect=BhoomiMitraException("Transcription resulted in empty text.", status_code=422)
        )

        await process_message_pipeline(parsed)

        assert sent_message_text is not None
        assert sent_message_text == VOICE_FAILURE_RESPONSES["te"]


# =============================================================================
# 4. Critical Chemical & Dosage Safety Gate Voice Test
# =============================================================================

@pytest.mark.asyncio
async def test_telugu_voice_dosage_query_triggers_safety_gate_when_ungrounded():
    """
    SAFETY TEST:
    A Telugu voice query asks for dosage: 'ఎకరానికి యూరియా ఎంత వేయాలి?'
    When verified RAG grounding documents are unavailable, the safety gate MUST trigger
    and block any invented numbers, returning the official Telugu AEO/KVK escalation.
    """
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.phone_number = "919876543210"
    mock_farmer.preferred_language = "te"
    mock_farmer.state = "Telangana"
    mock_farmer.district = "Warangal"

    mock_repo = AsyncMock()
    mock_repo.get_farmer_by_id = AsyncMock(return_value=mock_farmer)
    mock_repo.get_recent_conversations = AsyncMock(return_value=[])
    mock_repo.session = AsyncMock()

    service = AIService(repository=mock_repo)

    # Simulate RAG returning 0 relevant verified ground truth chunks
    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""):
        req = AIGenerateRequest(
            farmer_id=mock_farmer.id,
            message="ఎకరానికి యూరియా ఎంత వేయాలి?",
        )
        resp = await service.generate_ai_response(req)

        # Confirm safety gate triggered
        assert resp.intent == "dosage_unverified_fallback"
        assert resp.provider_used == "hard_grounding_gate"

        # Must be in Telugu and mention AEO / KVK
        assert "AEO" in resp.response_text or "KVK" in resp.response_text or "వ్యవసాయ విస్తరణ అధికారి" in resp.response_text


# =============================================================================
# 5. Meta WhatsApp Media Download & Pipeline Edge Case Tests
# =============================================================================

@pytest.mark.asyncio
async def test_download_media_bytes_missing_token_returns_none():
    """Verify download_media_bytes fails safely and returns None when WHATSAPP_API_TOKEN is unconfigured."""
    from src.gateway.whatsapp_client import download_media_bytes

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
        mock_settings.return_value.whatsapp_api_token = ""
        result = await download_media_bytes("media_test_123")
        assert result is None


@pytest.mark.asyncio
async def test_download_media_bytes_metadata_404_returns_none():
    """Verify download_media_bytes handles Meta metadata 404 cleanly."""
    from src.gateway.whatsapp_client import download_media_bytes
    import httpx

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 5.0
        mock_settings.return_value.max_media_download_bytes = 15_000_000

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404
        mock_resp.text = '{"error": {"message": "Not Found"}}'

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await download_media_bytes("media_missing_404")
            assert result is None


@pytest.mark.asyncio
async def test_download_media_bytes_successful_resolution_and_download():
    """Verify download_media_bytes resolves URL and returns bytes with correct MIME type."""
    from src.gateway.whatsapp_client import download_media_bytes
    import httpx

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 5.0
        mock_settings.return_value.max_media_download_bytes = 15_000_000

        mock_meta_resp = MagicMock(spec=httpx.Response)
        mock_meta_resp.status_code = 200
        mock_meta_resp.json.return_value = {
            "url": "https://lookaside.fbsbx.com/whatsapp_business/attachments/audio.ogg",
            "mime_type": "audio/ogg; codecs=opus",
        }

        mock_bin_resp = MagicMock(spec=httpx.Response)
        mock_bin_resp.status_code = 200
        mock_bin_resp.content = b"OggS_mock_audio_content"

        async def mock_get(url, headers=None, **kwargs):
            if "lookaside" in url:
                return mock_bin_resp
            return mock_meta_resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            res = await download_media_bytes("media_valid_123")
            assert res is not None
            raw_bytes, mime = res
            assert raw_bytes == b"OggS_mock_audio_content"
            assert mime == "audio/ogg; codecs=opus"


@pytest.mark.asyncio
async def test_voice_pipeline_missing_media_id_triggers_voice_fallback():
    """Verify that an audio message with no media_id returns the voice fallback immediately."""
    mock_db = AsyncMock()
    mock_db_cm = MagicMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.phone_number = "919876543210"
    mock_farmer.preferred_language = "te"

    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.NO_MEDIA_ID_01",
        timestamp="1700000000",
        message_type="audio",
        media_id=None,
    )

    sent_message_text = None

    async def mock_send(to_phone, message_text, **kwargs):
        nonlocal sent_message_text
        sent_message_text = message_text
        return "wamid.OUT_NO_MEDIA_OK"

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.send_text_message", side_effect=mock_send), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock):

        await process_message_pipeline(parsed)

        assert sent_message_text == VOICE_FAILURE_RESPONSES["te"]
