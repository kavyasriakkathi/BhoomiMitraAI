"""
BhoomiMitra AI — 13-Language End-to-End Voice Integration Test Suite (Step 6C-2)

Comprehensive local integration simulation validating the entire Voice-In -> Voice-Out
pipeline across all 13 supported Indian languages without requiring external Google or
WhatsApp credentials.

Pipeline simulated:
WhatsApp audio inbound
  -> Media download
  -> STT transcription & language detection
  -> AI decision engine & localized text response
  -> Google TTS synthesis (or text fallback for unsupported TTS languages)
  -> Meta media upload
  -> WhatsApp outbound text + native voice note delivery
"""

import uuid
import struct
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import Farmer, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.language.languages import (
    SUPPORTED_LANGUAGES,
    get_language,
    is_supported_language,
)
from src.language.schemas import TranscriptionResponse


# 13-Language Test Scenarios Data Table
LANGUAGE_TEST_CASES = [
    {
        "code": "te",
        "name": "Telugu",
        "stt_code": "te-IN",
        "tts_code": "te-IN",
        "tts_voice": "te-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "వరి పంటలో ఎరువుల యాజమాన్యం ఎలా చేయాలి?",
        "sample_response": "వరి పంటలో ఎకరానికి 50 కిలోల యూరియా మరియు 25 కిలోల పొటాష్ వేయండి.",
    },
    {
        "code": "hi",
        "name": "Hindi",
        "stt_code": "hi-IN",
        "tts_code": "hi-IN",
        "tts_voice": "hi-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "गेहूं की फसल में पहली सिंचाई कब करनी चाहिए?",
        "sample_response": "गेहूं की फसल में पहली सिंचाई बुवाई के 20 से 25 दिनों बाद करें।",
    },
    {
        "code": "en",
        "name": "English",
        "stt_code": "en-IN",
        "tts_code": "en-IN",
        "tts_voice": "en-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "What is the recommended fertilizer schedule for cotton?",
        "sample_response": "Apply 100 kg Nitrogen, 50 kg Phosphorus, and 50 kg Potassium per hectare for cotton.",
    },
    {
        "code": "ta",
        "name": "Tamil",
        "stt_code": "ta-IN",
        "tts_code": "ta-IN",
        "tts_voice": "ta-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "நெல் பயிரில் பூச்சி தாக்குதலை கட்டுப்படுத்துவது எப்படி?",
        "sample_response": "நெல் பயிரில் குருத்துப்பூச்சி தாக்கினால் வேப்பங்கொட்டை சாறு தெளிக்கவும்.",
    },
    {
        "code": "kn",
        "name": "Kannada",
        "stt_code": "kn-IN",
        "tts_code": "kn-IN",
        "tts_voice": "kn-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "ಜೋಳದ ಬೆಳೆಗೆ ಯಾವ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?",
        "sample_response": "ಜೋಳದ ಬೆಳೆಗೆ ಬಿತ್ತನೆಯ ಸಮಯದಲ್ಲಿ ಸೂಕ್ತ ಪ್ರಮಾಣದ ಡಿಎಪಿ ಮತ್ತು ಯೂರಿಯಾ ಬಳಸಿ.",
    },
    {
        "code": "ml",
        "name": "Malayalam",
        "stt_code": "ml-IN",
        "tts_code": "ml-IN",
        "tts_voice": "ml-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "തെങ്ങിലെ മണ്ഡരി ബാധ എങ്ങനെ തടയാം?",
        "sample_response": "തെങ്ങിലെ മണ്ഡരി ബാധ തടയാൻ വേപ്പെണ്ണ-വെളുത്തുള്ളി മിശ്രിതം തളിക്കുക.",
    },
    {
        "code": "mr",
        "name": "Marathi",
        "stt_code": "mr-IN",
        "tts_code": "mr-IN",
        "tts_voice": "mr-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "सोयाबीन पिकातील तण व्यवस्थापन कसे करावे?",
        "sample_response": "सोयाबीन पिकात पेरणीनंतर २० दिवसांनी खुरपणी करून तण नियंत्रण करावे.",
    },
    {
        "code": "bn",
        "name": "Bengali",
        "stt_code": "bn-IN",
        "tts_code": "bn-IN",
        "tts_voice": "bn-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "ধান চাষে সার প্রয়োগের সঠিক নিয়ম কি?",
        "sample_response": "ধান চাষে জমির প্রস্তুতিতে জৈব সার এবং পরবর্তীতে সময়মত ইউরিয়া প্রয়োগ করুন।",
    },
    {
        "code": "gu",
        "name": "Gujarati",
        "stt_code": "gu-IN",
        "tts_code": "gu-IN",
        "tts_voice": "gu-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "કપાસના પાકમાં ગુલાબી ઈયળ નિયંત્રણ કેવી રીતે કરવું?",
        "sample_response": "કપાસમાં ગુલાબી ઈયળ માટે ફેરોમોન ટ્રેપ લગાવો અને જૈવિક કીટનાશક છાંટો.",
    },
    {
        "code": "pa",
        "name": "Punjabi",
        "stt_code": "pa-Guru-IN",
        "tts_code": "pa-IN",
        "tts_voice": "pa-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "ਕਣਕ ਦੀ ਫ਼ਸਲ ਲਈ ਖਾਦ ਪ੍ਰਬੰਧਨ ਕਿਵੇਂ ਕਰੀਏ?",
        "sample_response": "ਕਣਕ ਦੀ ਫ਼ਸਲ ਵਿੱਚ ਪਹਿਲੇ ਪਾਣੀ ਵੇਲੇ ਯੂਰੀਆ ਦੀ ਸਹੀ ਮਾਤਰਾ ਪਾਓ।",
    },
    {
        "code": "or",
        "name": "Odia",
        "stt_code": "or-IN",
        "tts_code": None,
        "tts_voice": None,
        "supported_tts": False,
        "sample_query": "ଧାନ ଫସଲରେ ରୋଗ ନିୟନ୍ତ୍ରଣ ପାଇଁ କଣ କରିବା ଉଚିତ?",
        "sample_response": "ଧାନ ଫସଲରେ ରୋଗ ନିୟନ୍ତ୍ରଣ ପାଇଁ ଜୈବିକ ଔଷଧ ସିଞ୍ଚନ କରନ୍ତୁ।",
    },
    {
        "code": "as",
        "name": "Assamese",
        "stt_code": "as-IN",
        "tts_code": None,
        "tts_voice": None,
        "supported_tts": False,
        "sample_query": "ধান খেতিত পোক-পৰুৱা নিয়ন্ত্ৰণৰ উপায় কি?",
        "sample_response": "ধান খেতিত পোক নিয়ন্ত্ৰণৰ বাবে নিম তেলৰ মিশ্ৰণ ব্যৱহাৰ কৰক।",
    },
    {
        "code": "ur",
        "name": "Urdu",
        "stt_code": "ur-IN",
        "tts_code": "ur-IN",
        "tts_voice": "ur-IN-Standard-A",
        "supported_tts": True,
        "sample_query": "گندم کی فصل میں کھاد ڈالنے کا صحیح طریقہ کیا ہے؟",
        "sample_response": "گندم کی بوائی کے وقت ڈی اے پی اور پہلے پانی پر یوریا کھاد استعمال کریں۔",
    },
]


def make_valid_ogg_opus_bytes(serial: int = 12345) -> bytes:
    """Constructs a valid Ogg Opus container header."""
    opus_head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 0, 48000, 0, 0)
    page_hdr = struct.pack("<4sBBqIIIB", b"OggS", 0, 2, 0, serial, 0, 0, 1) + bytes([len(opus_head)])
    return page_hdr + opus_head


def build_mock_db_context(mock_farmer, mock_conv):
    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None
    return mock_db_cm


class Test13LanguageEndToEndVoiceSimulation:
    """
    Simulates end-to-end voice interactions for all 13 official BhoomiMitra languages.
    """

    @pytest.mark.parametrize("case", LANGUAGE_TEST_CASES, ids=[c["name"] for c in LANGUAGE_TEST_CASES])
    @pytest.mark.asyncio
    async def test_e2e_voice_pipeline_per_language(self, case):
        """
        Validates Voice Input -> STT -> Language Detection -> AI Advisory -> TTS (or fallback) -> WhatsApp.
        """
        lang_code = case["code"]
        lang_meta = get_language(lang_code)
        assert lang_meta is not None
        assert lang_meta.stt_code == case["stt_code"]
        assert lang_meta.supported_tts == case["supported_tts"]

        # Setup inbound WhatsApp audio message
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.E2E_{lang_code.upper()}_01",
            timestamp="1700000000",
            message_type="audio",
            media_id=f"inbound_audio_{lang_code}",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language=lang_code)
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = build_mock_db_context(mock_farmer, mock_conv)

        ogg_bytes = make_valid_ogg_opus_bytes()

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=case["sample_response"]), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value=f"wamid.TEXT_{lang_code.upper()}") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value=f"meta_media_{lang_code}") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value=f"wamid.AUDIO_{lang_code.upper()}") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True

            # STT returns query text & detected language
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(
                    provider_used="google",
                    transcription_text=case["sample_query"],
                    detected_language=lang_code,
                )
            )

            # TTS mock returns audio payload for supported languages, empty list for Odia & Assamese
            if case["supported_tts"]:
                mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[ogg_bytes])
            else:
                mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[])

            # Execute Gateway Background Pipeline
            await process_message_pipeline(parsed)

            # 1. Verify Text message is ALWAYS delivered
            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text=case["sample_response"],
            )

            # 2. Verify TTS & Outbound Voice Note behavior
            if case["supported_tts"]:
                # TTS must be requested with the canonical language code
                mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(
                    case["sample_response"], lang_code
                )
                # Meta media upload called with Ogg binary payload
                mock_upload.assert_awaited_once_with(ogg_bytes, mime_type="audio/ogg")
                # Meta audio send called with media ID
                mock_send_audio.assert_awaited_once_with("919876543210", f"meta_media_{lang_code}")
            else:
                # Odia & Assamese must skip upload and audio send safely
                mock_upload.assert_not_called()
                mock_send_audio.assert_not_called()

    @pytest.mark.parametrize("case", LANGUAGE_TEST_CASES, ids=[c["name"] for c in LANGUAGE_TEST_CASES])
    @pytest.mark.asyncio
    async def test_feature_flag_off_suppresses_voice_all_languages(self, case):
        """When ENABLE_VOICE_RESPONSES=False, all 13 languages produce text-only responses."""
        lang_code = case["code"]
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.FLAG_OFF_{lang_code.upper()}",
            timestamp="1700000000",
            message_type="audio",
            media_id=f"audio_{lang_code}",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language=lang_code)
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = build_mock_db_context(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=case["sample_response"]), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = False
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(
                    provider_used="google",
                    transcription_text=case["sample_query"],
                    detected_language=lang_code,
                )
            )

            await process_message_pipeline(parsed)

            # Text delivered
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=case["sample_response"])
            # Voice suppressed
            mock_lang_svc.return_value.synthesize_speech.assert_not_called()
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_punjabi_and_urdu_explicit_registry_codes(self):
        """Verifies specific STT/TTS codes for Punjabi and Urdu."""
        pa = get_language("pa")
        assert pa.stt_code == "pa-Guru-IN"
        assert pa.tts_code == "pa-IN"
        assert pa.tts_voice_name == "pa-IN-Standard-A"

        ur = get_language("ur")
        assert ur.stt_code == "ur-IN"
        assert ur.tts_code == "ur-IN"
        assert ur.tts_voice_name == "ur-IN-Standard-A"

    @pytest.mark.asyncio
    async def test_multi_chunk_option_c_fallback_preserves_full_text(self):
        """
        Verifies multi-chunk synthesis follows Option C:
        Sends primary chunk only without OGG concatenation, preserving complete text.
        """
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.MULTI_CHUNK_OPTION_C",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_multi",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = build_mock_db_context(mock_farmer, mock_conv)

        long_text = ("వరి పంటలో అగ్గి తెగులు నివారణకు సమగ్ర యాజమాన్యం. " * 10).strip()
        chunk1 = make_valid_ogg_opus_bytes(serial=101)
        chunk2 = make_valid_ogg_opus_bytes(serial=102)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=long_text), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_MULTI") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_media_c1") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_MULTI") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(
                    provider_used="google",
                    transcription_text="సుదీర్ఘ ప్రశ్న",
                    detected_language="te",
                )
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[chunk1, chunk2])

            await process_message_pipeline(parsed)

            # Full text delivered
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=long_text)
            # Only first chunk uploaded (NO b"".join)
            mock_upload.assert_awaited_once_with(chunk1, mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_media_c1")

    @pytest.mark.asyncio
    async def test_failure_isolation_tts_upload_send(self):
        """Verifies failure in TTS, media upload, or audio send does not break text delivery."""
        for failure_mode in ["tts_error", "upload_error", "send_error"]:
            parsed = ParsedIncomingMessage(
                phone_number="919876543210",
                message_id=f"wamid.FAIL_{failure_mode.upper()}",
                timestamp="1700000000",
                message_type="audio",
                media_id="audio_fail",
            )

            mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
            mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
            mock_db_cm = build_mock_db_context(mock_farmer, mock_conv)

            with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
                 patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
                 patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
                 patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
                 patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_bytes", "audio/ogg")), \
                 patch("src.gateway.service.get_language_service") as mock_lang_svc, \
                 patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="రైతు మిత్ర సమాచారం"), \
                 patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT") as mock_send_text, \
                 patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
                 patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
                 patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
                 patch("src.gateway.service.get_settings") as mock_settings:

                mock_settings.return_value.enable_voice_responses = True
                mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                    return_value=TranscriptionResponse(provider_used="google", transcription_text="ప్రశ్న", detected_language="te")
                )

                if failure_mode == "tts_error":
                    mock_lang_svc.return_value.synthesize_speech = AsyncMock(side_effect=Exception("TTS Network Error"))
                elif failure_mode == "upload_error":
                    mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[make_valid_ogg_opus_bytes()])
                    mock_upload.return_value = None
                elif failure_mode == "send_error":
                    mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[make_valid_ogg_opus_bytes()])
                    mock_upload.return_value = "media_id_123"
                    mock_send_audio.return_value = None

                await process_message_pipeline(parsed)

                # In ALL failure modes, text message MUST be delivered successfully
                mock_send_text.assert_awaited_once_with(
                    to_phone="919876543210",
                    message_text="రైతు మిత్ర సమాచారం",
                )
