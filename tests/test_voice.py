"""
BhoomiMitra AI — Step 6A Multilingual Voice Support & TTS Service Tests
"""

import pytest
import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings, get_settings
from src.language.languages import (
    SUPPORTED_LANGUAGES,
    get_language,
    is_supported_language,
    list_supported_languages,
)
from src.language.service import (
    LanguageService,
    clean_text_for_speech,
    chunk_text_for_speech,
    extract_ogg_opus_sample_rate,
)


class TestVoiceConfiguration:
    """Test voice and TTS configuration defaults."""

    def test_tts_config_defaults(self):
        settings = Settings()
        assert settings.enable_voice_responses is False
        assert settings.tts_provider == "google"
        assert settings.tts_voice_type == "Standard"
        assert settings.tts_api_timeout_seconds == 10.0
        assert settings.tts_max_text_chars == 3500


class TestLanguageRegistryVoiceMappings:
    """Test STT and TTS voice mappings for all 13 supported languages."""

    def test_total_13_languages_supported(self):
        assert len(SUPPORTED_LANGUAGES) == 13
        assert len(list_supported_languages()) == 13

    def test_punjabi_stt_and_tts_code_separation(self):
        pa = get_language("pa")
        assert pa is not None
        assert pa.code == "pa"
        assert pa.stt_code == "pa-Guru-IN"  # Gurmukhi script for Google STT
        assert pa.tts_code == "pa-IN"
        assert pa.tts_voice_name == "pa-IN-Standard-A"
        assert pa.supported_stt is True
        assert pa.supported_tts is True

    def test_urdu_tts_indian_locale_mapping(self):
        ur = get_language("ur")
        assert ur is not None
        assert ur.code == "ur"
        assert ur.stt_code == "ur-IN"
        assert ur.tts_code == "ur-IN"  # Indian Urdu locale
        assert ur.tts_voice_name == "ur-IN-Standard-A"
        assert ur.supported_stt is True
        assert ur.supported_tts is True

    def test_odia_and_assamese_unsupported_tts(self):
        # Odia
        or_lang = get_language("or")
        assert or_lang is not None
        assert or_lang.supported_stt is True
        assert or_lang.supported_tts is False
        assert or_lang.tts_code is None
        assert or_lang.tts_voice_name is None

        # Assamese
        as_lang = get_language("as")
        assert as_lang is not None
        assert as_lang.supported_stt is True
        assert as_lang.supported_tts is False
        assert as_lang.tts_code is None
        assert as_lang.tts_voice_name is None

    @pytest.mark.parametrize(
        "lang_code,expected_stt,expected_tts,expected_voice",
        [
            ("te", "te-IN", "te-IN", "te-IN-Standard-A"),
            ("hi", "hi-IN", "hi-IN", "hi-IN-Standard-A"),
            ("en", "en-IN", "en-IN", "en-IN-Standard-A"),
            ("ta", "ta-IN", "ta-IN", "ta-IN-Standard-A"),
            ("kn", "kn-IN", "kn-IN", "kn-IN-Standard-A"),
            ("ml", "ml-IN", "ml-IN", "ml-IN-Standard-A"),
            ("mr", "mr-IN", "mr-IN", "mr-IN-Standard-A"),
            ("bn", "bn-IN", "bn-IN", "bn-IN-Standard-A"),
            ("gu", "gu-IN", "gu-IN", "gu-IN-Standard-A"),
            ("pa", "pa-Guru-IN", "pa-IN", "pa-IN-Standard-A"),
            ("ur", "ur-IN", "ur-IN", "ur-IN-Standard-A"),
        ],
    )
    def test_all_11_supported_tts_languages(self, lang_code, expected_stt, expected_tts, expected_voice):
        meta = get_language(lang_code)
        assert meta is not None
        assert meta.supported_stt is True
        assert meta.supported_tts is True
        assert meta.stt_code == expected_stt
        assert meta.tts_code == expected_tts
        assert meta.tts_voice_name == expected_voice


class TestSpeechTextCleaning:
    """Test clean_text_for_speech ensures proper sanitization and preservation."""

    def test_strips_markdown_and_emojis(self):
        text = """### 🌾 **వరి పంట సూచనలు:**
* వేప నూనె (Neem Oil): 2.5 ml / L మోతాదులో పిచికారీ చేయండి.
* ధర: ₹2,500/క్వింటాల్.
⚠️ **హెచ్చరిక:** మందు పిచికారీ చేసేటప్పుడు మాస్క్ ధరించండి!
Find more at: https://bhoomimitra.in/shops or /shops"""

        cleaned = clean_text_for_speech(text)
        assert "###" not in cleaned
        assert "**" not in cleaned
        assert "🌾" not in cleaned
        assert "⚠️" not in cleaned
        assert "https://" not in cleaned
        assert "/shops" not in cleaned
        assert "వరి పంట సూచనలు:" in cleaned
        assert "2.5 ml / L" in cleaned
        assert "₹2,500/క్వింటాల్" in cleaned
        assert "మాస్క్ ధరించండి" in cleaned

    def test_preserves_indic_scripts_and_vowel_signs(self):
        # Telugu
        te = clean_text_for_speech("వరి పంటకు 50 కిలోల యూరియా వేయండి.")
        assert te == "వరి పంటకు 50 కిలోల యూరియా వేయండి."

        # Hindi
        hi = clean_text_for_speech("गेहूं की फसल में 2 लीटर पानी प्रति एकड़ छिड़काव करें।")
        assert hi == "गेहूं की फसल में 2 लीटर पानी प्रति एकड़ छिड़काव करें।"

        # Tamil
        ta = clean_text_for_speech("நெல் பயிருக்கு 25 கிலோ உரம் இடவும்.")
        assert ta == "நெல் பயிருக்கு 25 கிலோ உரம் இடவும்."

        # Bengali
        bn = clean_text_for_speech("ধানের জন্য ১০ কেজি সার প্রয়োগ করুন।")
        assert bn == "ধানের জন্য ১০ কেজি সার প্রয়োগ করুন।"

        # Urdu
        ur = clean_text_for_speech("گندم کی فصل کے لیے مناسب کھاد کا استعمال کریں۔")
        assert ur == "گندم کی فصل کے لیے مناسب کھاد کا استعمال کریں۔"

    def test_preserves_numbers_prices_dosages(self):
        raw = "Apply 1.5 - 2.0 ml/L of Monocrotophos 36% SL at ₹450 per 500ml bottle on 12/09/2026."
        cleaned = clean_text_for_speech(raw)
        assert "1.5 - 2.0 ml/L" in cleaned
        assert "36%" in cleaned
        assert "₹450" in cleaned
        assert "500ml" in cleaned
        assert "12/09/2026" in cleaned

    def test_empty_and_whitespace_handling(self):
        assert clean_text_for_speech("") == ""
        assert clean_text_for_speech("   \n\n  ") == ""
        assert clean_text_for_speech(None) == ""


class TestSentenceAwareChunking:
    """Test chunk_text_for_speech respects character/byte limits and preserves content."""

    def test_single_short_text_no_split(self):
        text = "This is a short advisory sentence."
        chunks = chunk_text_for_speech(text, max_chars=3500, max_bytes=4800)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_across_sentence_boundaries_preserves_decimals(self):
        text = "Sentence one with 2.5 ml dosage. Sentence two costs Rs. 500. Sentence three is a warning!"
        chunks = chunk_text_for_speech(text, max_chars=40, max_bytes=100)
        assert len(chunks) >= 2
        # Verify 2.5 is not fractured into '2.' and '5'
        joined = " ".join(chunks)
        assert "2.5 ml dosage" in joined
        assert "Rs. 500" in joined
        assert "Sentence three is a warning!" in joined

    def test_utf8_byte_limit_enforcement(self):
        # Indic text takes 3 bytes per character
        indic_sentence = "వరి పంట సూచనలు మరియు ఎరువుల మోతాదు వివరాలు."
        chunks = chunk_text_for_speech(indic_sentence * 5, max_chars=3500, max_bytes=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 100

    def test_oversized_single_sentence_sub_chunking(self):
        # A huge single sentence without periods
        long_sentence = "This is an extremely long single sentence " * 20
        chunks = chunk_text_for_speech(long_sentence, max_chars=100, max_bytes=150)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100
            assert len(chunk.encode("utf-8")) <= 150

    def test_no_content_silently_dropped(self):
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
        chunks = chunk_text_for_speech(text, max_chars=25, max_bytes=50)
        joined = " ".join(chunks)
        for i in range(1, 6):
            assert f"Sentence {i}" in joined


@pytest.mark.asyncio
class TestTTSServiceSynthesis:
    """Test Google Cloud TTS async synthesis, mocking, and error handling."""

    async def test_synthesis_returns_opus_bytes_for_supported_language(self):
        service = LanguageService()
        mock_audio_content = b"OggS\x00\x02mock_opus_audio_bytes"

        mock_response = MagicMock()
        mock_response.audio_content = mock_audio_content

        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = AsyncMock(return_value=mock_response)

        with patch.object(service, "_google_tts_client", mock_tts_client):
            audio_chunks = await service.synthesize_speech(
                text="వరి పంటకు ఎరువుల మోతాదు తెలుసుకోండి.",
                language_code="te",
            )
            assert isinstance(audio_chunks, list)
            assert len(audio_chunks) == 1
            assert audio_chunks[0] == mock_audio_content
            mock_tts_client.synthesize_speech.assert_called_once()
            call_kwargs = mock_tts_client.synthesize_speech.call_args.kwargs
            req = call_kwargs.get("request")
            assert req.voice.language_code == "te-IN"
            assert req.voice.name == "te-IN-Standard-A"

    async def test_unsupported_languages_return_empty_list_without_calling_google(self):
        service = LanguageService()
        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = AsyncMock()

        with patch.object(service, "_google_tts_client", mock_tts_client):
            # Odia
            res_or = await service.synthesize_speech("ଓଡ଼ିଆ ପରାମର୍ଶ", language_code="or")
            assert res_or == []

            # Assamese
            res_as = await service.synthesize_speech("অসমীয়া পৰামৰ্শ", language_code="as")
            assert res_as == []

            mock_tts_client.synthesize_speech.assert_not_called()

    async def test_synthesis_empty_text_returns_empty_list(self):
        service = LanguageService()
        res = await service.synthesize_speech("", language_code="hi")
        assert res == []

        res_none = await service.synthesize_speech(None, language_code="hi")
        assert res_none == []

    async def test_synthesis_timeout_fails_safely_returning_empty_list(self):
        service = LanguageService()

        async def timeout_synth(*args, **kwargs):
            await asyncio.sleep(0.5)
            raise asyncio.TimeoutError()

        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = timeout_synth

        with patch.object(service, "_google_tts_client", mock_tts_client):
            with patch.object(service.settings, "tts_api_timeout_seconds", 0.01):
                audio_chunks = await service.synthesize_speech("Hindi test", language_code="hi")
                assert audio_chunks == []

    async def test_synthesis_exception_fails_safely_returning_empty_list(self):
        service = LanguageService()

        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = AsyncMock(side_effect=RuntimeError("Google Cloud Quota Exceeded"))

        with patch.object(service, "_google_tts_client", mock_tts_client):
            audio_chunks = await service.synthesize_speech("Punjabi test", language_code="pa")
            assert audio_chunks == []

    @pytest.mark.parametrize(
        "lang_code,voice_name,tts_code",
        [
            ("te", "te-IN-Standard-A", "te-IN"),
            ("hi", "hi-IN-Standard-A", "hi-IN"),
            ("en", "en-IN-Standard-A", "en-IN"),
            ("ta", "ta-IN-Standard-A", "ta-IN"),
            ("kn", "kn-IN-Standard-A", "kn-IN"),
            ("ml", "ml-IN-Standard-A", "ml-IN"),
            ("mr", "mr-IN-Standard-A", "mr-IN"),
            ("bn", "bn-IN-Standard-A", "bn-IN"),
            ("gu", "gu-IN-Standard-A", "gu-IN"),
            ("pa", "pa-IN-Standard-A", "pa-IN"),
            ("ur", "ur-IN-Standard-A", "ur-IN"),
        ],
    )
    async def test_synthesis_all_11_supported_languages(self, lang_code, voice_name, tts_code):
        service = LanguageService()
        mock_response = MagicMock()
        mock_response.audio_content = b"OggS\x00\x02test_audio"

        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = AsyncMock(return_value=mock_response)

        with patch.object(service, "_google_tts_client", mock_tts_client):
            result = await service.synthesize_speech("Agricultural advisory test", language_code=lang_code)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == b"OggS\x00\x02test_audio"
            req = mock_tts_client.synthesize_speech.call_args.kwargs.get("request")
            assert req.voice.language_code == tts_code
            assert req.voice.name == voice_name

    async def test_multi_chunk_synthesis_returns_ordered_payloads(self):
        service = LanguageService()
        chunk1_audio = b"OggS_chunk1_valid_container"
        chunk2_audio = b"OggS_chunk2_valid_container"

        resp1 = MagicMock()
        resp1.audio_content = chunk1_audio
        resp2 = MagicMock()
        resp2.audio_content = chunk2_audio

        mock_tts_client = MagicMock()
        mock_tts_client.synthesize_speech = AsyncMock(side_effect=[resp1, resp2])

        # Force max_chars=40 so it splits into exactly 2 sentences/chunks
        with patch.object(service, "_google_tts_client", mock_tts_client):
            with patch.object(service.settings, "tts_max_text_chars", 40):
                long_text = "Sentence one about rice crops. Sentence two about fertilizer dosage."
                audio_results = await service.synthesize_speech(long_text, language_code="te")
                assert isinstance(audio_results, list)
                assert len(audio_results) == 2
                # Verify order is strictly preserved and chunks are independent payloads
                assert audio_results[0] == chunk1_audio
                assert audio_results[1] == chunk2_audio
                assert mock_tts_client.synthesize_speech.call_count == 2

    async def test_voice_name_failure_falls_back_to_generic_language_locale(self):
        service = LanguageService()
        fallback_resp = MagicMock()
        fallback_resp.audio_content = b"OggS_fallback_audio"

        mock_tts_client = MagicMock()
        # First call fails on voice name, second call succeeds with generic locale
        mock_tts_client.synthesize_speech = AsyncMock(
            side_effect=[RuntimeError("Voice te-IN-Standard-A not found"), fallback_resp]
        )

        with patch.object(service, "_google_tts_client", mock_tts_client):
            audio_results = await service.synthesize_speech("తెలుగు సమాచారం", language_code="te")
            assert isinstance(audio_results, list)
            assert len(audio_results) == 1
            assert audio_results[0] == b"OggS_fallback_audio"
            assert mock_tts_client.synthesize_speech.call_count == 2
            # Second call must use te-IN without switching language
            second_call_req = mock_tts_client.synthesize_speech.call_args_list[1].kwargs["request"]
            assert second_call_req.voice.language_code == "te-IN"
            assert second_call_req.voice.name == "" or second_call_req.voice.name is None

    async def test_unsupported_provider_returns_empty_list(self):
        service = LanguageService()
        with patch.object(service.settings, "tts_provider", "unsupported_tts"):
            res = await service.synthesize_speech("Test text", language_code="en")
            assert res == []


@pytest.mark.asyncio
class TestWhatsAppOutboundAudio:
    """Test WhatsApp Cloud API media upload and audio message dispatch."""

    async def test_upload_media_bytes_success(self):
        from src.gateway.whatsapp_client import upload_media_bytes

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "meta_media_id_9999"}

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
            with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
                mock_settings.return_value.whatsapp_api_token = "valid_token"
                mock_settings.return_value.whatsapp_phone_number_id = "123456789"
                media_id = await upload_media_bytes(b"OggS\x00\x02audio_binary", mime_type="audio/ogg")
                assert media_id == "meta_media_id_9999"

    async def test_upload_media_bytes_empty_payload(self):
        from src.gateway.whatsapp_client import upload_media_bytes

        res = await upload_media_bytes(b"")
        assert res is None

        res_none = await upload_media_bytes(None)
        assert res_none is None

    async def test_upload_media_bytes_missing_credentials(self):
        from src.gateway.whatsapp_client import upload_media_bytes

        with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
            mock_settings.return_value.whatsapp_api_token = ""
            mock_settings.return_value.whatsapp_phone_number_id = ""
            media_id = await upload_media_bytes(b"OggS_binary")
            assert media_id is None

    async def test_upload_media_bytes_http_error(self):
        from src.gateway.whatsapp_client import upload_media_bytes

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
            with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
                mock_settings.return_value.whatsapp_api_token = "token"
                mock_settings.return_value.whatsapp_phone_number_id = "12345"
                media_id = await upload_media_bytes(b"OggS_binary")
                assert media_id is None

    async def test_upload_media_bytes_timeout(self):
        from src.gateway.whatsapp_client import upload_media_bytes
        import httpx

        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=httpx.TimeoutException("Timeout"))):
            with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
                mock_settings.return_value.whatsapp_api_token = "token"
                mock_settings.return_value.whatsapp_phone_number_id = "12345"
                media_id = await upload_media_bytes(b"OggS_binary")
                assert media_id is None

    async def test_send_audio_message_success(self):
        from src.gateway.whatsapp_client import send_audio_message

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"messages": [{"id": "wamid.AUDIO_MSG_123"}]}

        mock_post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient.post", mock_post):
            with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
                mock_settings.return_value.whatsapp_api_token = "token"
                mock_settings.return_value.whatsapp_phone_number_id = "12345"
                wa_id = await send_audio_message("919876543210", "meta_media_id_9999")
                assert wa_id == "wamid.AUDIO_MSG_123"

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args.kwargs
                sent_payload = call_kwargs.get("json")
                assert sent_payload["type"] == "audio"
                assert sent_payload["to"] == "919876543210"
                assert sent_payload["audio"]["id"] == "meta_media_id_9999"
                assert sent_payload["audio"]["voice"] is True

    async def test_send_audio_message_missing_inputs(self):
        from src.gateway.whatsapp_client import send_audio_message

        # Missing phone
        res_phone = await send_audio_message("", "media_123")
        assert res_phone is None

        # Missing media_id
        res_media = await send_audio_message("919876543210", "")
        assert res_media is None

    async def test_send_audio_message_missing_credentials(self):
        from src.gateway.whatsapp_client import send_audio_message

        with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
            mock_settings.return_value.whatsapp_api_token = ""
            mock_settings.return_value.whatsapp_phone_number_id = ""
            wa_id = await send_audio_message("919876543210", "media_123")
            assert wa_id is None

    async def test_send_audio_message_http_failure(self):
        from src.gateway.whatsapp_client import send_audio_message

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
            with patch("src.gateway.whatsapp_client.get_settings") as mock_settings:
                mock_settings.return_value.whatsapp_api_token = "token"
                mock_settings.return_value.whatsapp_phone_number_id = "12345"
                wa_id = await send_audio_message("919876543210", "media_123")
                assert wa_id is None


class TestGatewayVoiceIntegration:
    """Integration tests for Gateway Voice-In -> Voice-Out routing in Stage 6."""

    def _setup_mocks(self, mock_farmer, mock_conv):
        mock_db = AsyncMock()
        mock_db_cm = AsyncMock()
        mock_db_cm.__aenter__.return_value = mock_db
        mock_db_cm.__aexit__.return_value = None
        return mock_db_cm

    @pytest.mark.asyncio
    async def test_voice_input_feature_flag_off_text_only(self):
        """When enable_voice_responses is False, audio input results in text response only."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.VOICE_FLAG_OFF_01",
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
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_in_ogg", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరి సాగు సలహా"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_01") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = False
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="వరిలో తెగులు", detected_language="te")
            )

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once()
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_input_feature_flag_on_text_only(self):
        """When enable_voice_responses is True but input is text, outbound is text-only."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.TEXT_FLAG_ON_01",
            timestamp="1700000000",
            message_type="text",
            text_content="వరి సాగు వివరాలు",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message="వరి సాగు వివరాలు",
            user_message_type="text",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరి సాగు సమాధానం"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_02") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once()
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_input_feature_flag_on_single_chunk_success(self):
        """Voice input + flag ON + single TTS chunk -> both text and audio dispatched."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.VOICE_SUCCESS_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_222",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="రైతు మిత్ర సమాధానం"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_03") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_media_id_333") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_OUT_01") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="పంట సమస్య", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(
                return_value=[b"OggS_single_chunk_bytes"]
            )

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text="రైతు మిత్ర సమాధానం",
            )
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("రైతు మిత్ర సమాధానం", "te")
            mock_upload.assert_awaited_once_with(b"OggS_single_chunk_bytes", mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_media_id_333")

    @pytest.mark.asyncio
    async def test_voice_input_tts_failure_preserves_text_delivery(self):
        """Voice input + TTS exception -> text response is still delivered reliably."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.TTS_FAIL_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_333",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="సలహా సమాచారం"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_04") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="తెగులు మందు", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(side_effect=Exception("Google TTS API Error"))

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once()
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_input_media_upload_failure_preserves_text_delivery(self):
        """Voice input + upload failure (returns None) -> text delivered, audio send skipped."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.UPLOAD_FAIL_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_444",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="సలహా"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_05") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value=None) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="తెగులు", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"chunk_bytes"])

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once()
            mock_upload.assert_awaited_once()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_input_audio_send_failure_preserves_text_delivery(self):
        """Voice input + send_audio_message failure (returns None) -> text delivered safely."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.AUDIO_SEND_FAIL_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_555",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="సలహా"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_06") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="media_555") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value=None) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="తెగులు", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"chunk_bytes"])

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once()
            mock_upload.assert_awaited_once()
            mock_send_audio.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_odia_and_assamese_voice_input_tts_fallback(self):
        """Odia and Assamese voice inputs return empty audio list from TTS; text response is delivered."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        for lang_code, query_text in [("or", "ଧାନ ଫସଲ ରୋଗ"), ("as", "ধান খেতিৰ ৰোগ")]:
            parsed = ParsedIncomingMessage(
                phone_number="919876543210",
                message_id=f"wamid.UNSUPPORTED_TTS_{lang_code.upper()}",
                timestamp="1700000000",
                message_type="audio",
                media_id=f"audio_media_{lang_code}",
            )

            mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language=lang_code)
            mock_conv = Conversation(
                id=uuid.uuid4(),
                farmer_id=mock_farmer.id,
                message_id=parsed.message_id,
                user_message=None,
                user_message_type="audio",
            )
            mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

            with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
                 patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
                 patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
                 patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
                 patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
                 patch("src.gateway.service.get_language_service") as mock_lang_svc, \
                 patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="Text Advisory"), \
                 patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_UNSUP") as mock_send_text, \
                 patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
                 patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
                 patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
                 patch("src.gateway.service.get_settings") as mock_settings:

                mock_settings.return_value.enable_voice_responses = True
                mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                    return_value=TranscriptionResponse(provider_used="google", transcription_text=query_text, detected_language=lang_code)
                )
                mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[])  # Unsupported TTS

                await process_message_pipeline(parsed)

                mock_send_text.assert_awaited_once()
                mock_upload.assert_not_called()
                mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_chunk_response_option_c_fallback(self):
        """Multi-chunk TTS response dispatches primary chunk only without OGG concatenation, plus complete text."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.MULTI_CHUNK_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_multi",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        chunk1 = b"OggS_primary_chunk_audio"
        chunk2 = b"OggS_secondary_chunk_audio"
        chunk3 = b"OggS_tertiary_chunk_audio"

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"audio_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="దీర్ఘ సమాచారం"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_MULTI") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="media_multi_chunk") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_OUT_MULTI") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సుదీర్ఘ ప్రశ్న", detected_language="te")
            )
            # Returns 3 separate OGG chunks
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(
                return_value=[chunk1, chunk2, chunk3]
            )

            await process_message_pipeline(parsed)

            # Full text delivered
            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text="దీర్ఘ సమాచారం",
            )
            # Only primary chunk uploaded (NOT raw byte concatenation)
            mock_upload.assert_awaited_once_with(chunk1, mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "media_multi_chunk")

    @pytest.mark.asyncio
    async def test_existing_non_voice_messages_remain_unchanged(self):
        """Image and non-voice inputs do not trigger voice note generation even when flag is ON."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.IMG_NON_VOICE_01",
            timestamp="1700000000",
            message_type="image",
            media_id="img_media_999",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="image",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"image_bytes", "image/jpeg")), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value="ఆకు మచ్చ తెగులు"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_IMG") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text="ఆకు మచ్చ తెగులు",
            )
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()


class TestControlledEndToEndVoiceScenarios:
    """
    Controlled End-to-End Voice Integration Scenarios (Step 6C).
    Validates end-to-end routing, binary OGG/Opus payloads, multilingual fallbacks,
    feature flag toggles, failure isolation, and multi-chunk Option C strategy.
    """

    def _generate_valid_ogg_opus_payload(self) -> bytes:
        """Constructs a valid Ogg container header containing an OpusHead stream marker."""
        import struct
        opus_head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 0, 48000, 0, 0)
        page_hdr = struct.pack("<4sBBqIIIB", b"OggS", 0, 2, 0, 12345, 0, 0, 1) + bytes([len(opus_head)])
        return page_hdr + opus_head

    def _setup_mocks(self, mock_farmer, mock_conv):
        mock_db = AsyncMock()
        mock_db_cm = AsyncMock()
        mock_db_cm.__aenter__.return_value = mock_db
        mock_db_cm.__aexit__.return_value = None
        return mock_db_cm

    @pytest.mark.asyncio
    async def test_1_telugu_voice_input_e2e_valid_ogg_opus(self):
        """Case 1: Telugu voice input -> STT -> AI -> text -> TTS -> WhatsApp voice with playable OGG/Opus payload."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.TELUGU_E2E_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="telugu_inbound_media_id",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        ogg_payload = self._generate_valid_ogg_opus_payload()

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_user_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరిలో తెగులు నివారణకు సిఫార్సు చేయబడిన మందు కొట్టండి."), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TELUGU_OUT_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_media_telugu_999") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.TELUGU_OUT_AUDIO") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="వరిలో తెగులు వచ్చింది", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[ogg_payload])

            await process_message_pipeline(parsed)

            # 1. Text is delivered
            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text="వరిలో తెగులు నివారణకు సిఫార్సు చేయబడిన మందు కొట్టండి.",
            )
            # 2. TTS was called with Telugu
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(
                "వరిలో తెగులు నివారణకు సిఫార్సు చేయబడిన మందు కొట్టండి.", "te"
            )
            # 3. Uploaded audio is valid Ogg Opus container
            mock_upload.assert_awaited_once_with(ogg_payload, mime_type="audio/ogg")
            uploaded_bytes = mock_upload.call_args.args[0]
            assert uploaded_bytes[:4] == b"OggS"
            assert b"OpusHead" in uploaded_bytes
            # 4. WhatsApp audio message sent
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_media_telugu_999")

    @pytest.mark.asyncio
    async def test_2_hindi_voice_input_e2e(self):
        """Case 2: Hindi voice input -> STT -> AI -> text -> TTS -> WhatsApp voice with hi-IN voice."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.HINDI_E2E_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="hindi_inbound_media_id",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="hi")
        mock_conv = Conversation(
            id=uuid.uuid4(),
            farmer_id=mock_farmer.id,
            message_id=parsed.message_id,
            user_message=None,
            user_message_type="audio",
        )
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        ogg_payload = self._generate_valid_ogg_opus_payload()

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_user_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="गेहूं की फसल में यूरिया का उचित उपयोग करें।"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.HINDI_OUT_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_media_hindi_888") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.HINDI_OUT_AUDIO") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="गेहूं में खाद कैसे डालें", detected_language="hi")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[ogg_payload])

            await process_message_pipeline(parsed)

            # 1. Text is delivered
            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text="गेहूं की फसल में यूरिया का उचित उपयोग करें।",
            )
            # 2. TTS was called with Hindi
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(
                "गेहूं की फसल में यूरिया का उचित उपयोग करें।", "hi"
            )
            # 3. Audio upload and send
            mock_upload.assert_awaited_once_with(ogg_payload, mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_media_hindi_888")

    @pytest.mark.asyncio
    async def test_3_difficult_languages_fallback_and_mapping(self):
        """Case 3: Difficult languages — Assamese/Odia fallback to text, Urdu/Marathi synthesize speech."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        ogg_payload = self._generate_valid_ogg_opus_payload()

        # Part A: Assamese (Unsupported TTS -> Text delivered, Audio skipped)
        parsed_as = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.ASSAMESE_E2E_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="as_media_id",
        )
        mock_farmer_as = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="as")
        mock_conv_as = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer_as.id, message_id=parsed_as.message_id, user_message_type="audio")
        mock_db_as = self._setup_mocks(mock_farmer_as, mock_conv_as)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_as), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer_as), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv_as), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="ধান খেতিৰ পৰামৰ্শ"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.AS_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="ধান খেতি", detected_language="as")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[])  # Unsupported TTS

            await process_message_pipeline(parsed_as)

            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="ধান খেতিৰ পৰামৰ্শ")
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

        # Part B: Urdu (Supported TTS with ur-IN -> Audio + Text delivered)
        parsed_ur = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.URDU_E2E_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="ur_media_id",
        )
        mock_farmer_ur = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="ur")
        mock_conv_ur = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer_ur.id, message_id=parsed_ur.message_id, user_message_type="audio")
        mock_db_ur = self._setup_mocks(mock_farmer_ur, mock_conv_ur)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_ur), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer_ur), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv_ur), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="کپاس کی فصل کے لئے رہنمائی۔"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.UR_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_ur_media_id") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.UR_AUDIO") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="کپاس کی بیماری", detected_language="ur")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[ogg_payload])

            await process_message_pipeline(parsed_ur)

            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="کپاس کی فصل کے لئے رہنمائی۔")
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with("کپاس کی فصل کے لئے رہنمائی۔", "ur")
            mock_upload.assert_awaited_once_with(ogg_payload, mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_ur_media_id")

    @pytest.mark.asyncio
    async def test_4_feature_flag_off_text_only(self):
        """Case 4: Feature flag OFF -> Voice input produces text response only; zero TTS/media upload."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.FLAG_OFF_E2E",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_flag_off",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వరి సాగు సూచనలు"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.FLAG_OFF_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = False
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="వరి సాగు", detected_language="te")
            )

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="వరి సాగు సూచనలు")
            mock_lang_svc.return_value.synthesize_speech.assert_not_called()
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_5_tts_failure_preserves_text_delivery(self):
        """Case 5: TTS synthesis failure -> Error handled gracefully, complete text delivered."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.TTS_FAIL_E2E",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_tts_fail",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="వివరమైన సమాచారం."), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TTS_FAIL_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సమస్య", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(side_effect=Exception("TTS Network Timeout"))

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="వివరమైన సమాచారం.")
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_6_media_upload_failure_preserves_text_delivery(self):
        """Case 6: Meta media upload failure -> Handled gracefully, text response preserved."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.UPLOAD_FAIL_E2E",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_upload_fail",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        ogg_payload = self._generate_valid_ogg_opus_payload()

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="సలహా సందేశం"), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.UPLOAD_FAIL_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value=None) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సలహా", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[ogg_payload])

            await process_message_pipeline(parsed)

            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="సలహా సందేశం")
            mock_upload.assert_awaited_once_with(ogg_payload, mime_type="audio/ogg")
            mock_send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_7_multi_chunk_response_option_c_fallback(self):
        """Case 7: Multi-chunk response -> Option C fallback dispatches first chunk only without OGG concatenation, preserving full text."""
        import uuid
        from src.core.models import Farmer, Conversation
        from src.gateway.schemas import ParsedIncomingMessage
        from src.gateway.service import process_message_pipeline
        from src.language.schemas import TranscriptionResponse

        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.MULTI_CHUNK_E2E",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_multi_e2e",
        )

        mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        mock_conv = Conversation(id=uuid.uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message_type="audio")
        mock_db_cm = self._setup_mocks(mock_farmer, mock_conv)

        long_ai_response = (
            "వరి పంటలో అగ్గి తెగులు నివారణకు ట్రైసైక్లాజోల్ 75% WP మందును 0.6 గ్రాములు ఒక లీటరు నీటికి కలిపి పిచికారీ చేయాలి. "
            "వాతావరణంలో తేమ ఎక్కువగా ఉన్నందున వెంటనే నివారణ చర్యలు చేపట్టండి. "
            "సమీప రైతు భరోసా కేంద్రాన్ని లేదా వ్యవసాయ అధికారిని సంప్రదించండి."
        )

        chunk1_ogg = self._generate_valid_ogg_opus_payload()
        chunk2_ogg = self._generate_valid_ogg_opus_payload()
        chunk3_ogg = self._generate_valid_ogg_opus_payload()

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_voice", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=long_ai_response), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.MULTI_TEXT_OUT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_multi_media_id") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.MULTI_AUDIO_OUT") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="వరిలో అగ్గి తెగులు", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(
                return_value=[chunk1_ogg, chunk2_ogg, chunk3_ogg]
            )

            await process_message_pipeline(parsed)

            # 1. Complete text delivered (no truncation)
            mock_send_text.assert_awaited_once_with(
                to_phone="919876543210",
                message_text=long_ai_response,
            )
            # 2. Only first chunk is uploaded (Option C, NOT b"".join)
            mock_upload.assert_awaited_once_with(chunk1_ogg, mime_type="audio/ogg")
            # 3. Audio dispatched once
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_multi_media_id")


class TestOggOpusSampleRateHandling:
    """Verify that OGG_OPUS sample rate is correctly parsed and never set to 0 for Google STT."""

    def test_extract_ogg_opus_sample_rate_valid_rates(self):
        for rate in [8000, 12000, 16000, 24000, 48000]:
            header = b"OggS" + b"\x00" * 24 + b"OpusHead" + b"\x01\x01\x00\x00" + struct.pack("<I", rate)
            assert extract_ogg_opus_sample_rate(header) == rate

    def test_extract_ogg_opus_sample_rate_unsupported_rate_falls_back(self):
        # 44100 is not in Google STT's supported Opus rates {8000, 12000, 16000, 24000, 48000}
        header = b"OggS" + b"\x00" * 24 + b"OpusHead" + b"\x01\x01\x00\x00" + struct.pack("<I", 44100)
        assert extract_ogg_opus_sample_rate(header, default_rate=16000) == 16000

    def test_extract_ogg_opus_sample_rate_missing_header_fallback(self):
        assert extract_ogg_opus_sample_rate(b"random_audio_bytes", default_rate=16000) == 16000
        assert extract_ogg_opus_sample_rate(b"", default_rate=16000) == 16000

    @pytest.mark.asyncio
    async def test_transcribe_with_google_sets_sample_rate_for_all_formats(self):
        from google.cloud import speech

        service = LanguageService()
        captured_configs = []

        mock_client = AsyncMock()
        mock_response = speech.RecognizeResponse(
            results=[
                speech.SpeechRecognitionResult(
                    alternatives=[speech.SpeechRecognitionAlternative(transcript="పంట సమాచారం", confidence=0.95)]
                )
            ]
        )

        async def fake_recognize(config, audio):
            captured_configs.append(config)
            return mock_response

        mock_client.recognize = fake_recognize

        with patch.object(LanguageService, "google_client", new_callable=lambda: property(lambda self: mock_client)):
            # 1. Ogg Opus with OpusHead (24000 Hz)
            ogg_24k = b"OggS" + b"\x00" * 24 + b"OpusHead" + b"\x01\x01\x00\x00" + struct.pack("<I", 24000)
            res1 = await service._transcribe_with_google(ogg_24k, "audio/ogg; codecs=opus")
            assert res1.transcription_text == "పంట సమాచారం"
            assert captured_configs[-1].encoding == speech.RecognitionConfig.AudioEncoding.OGG_OPUS
            assert captured_configs[-1].sample_rate_hertz == 24000

            # 2. Ogg Opus fallback (16000 Hz)
            res2 = await service._transcribe_with_google(b"raw_opus", "audio/ogg")
            assert res2.transcription_text == "పంట సమాచారం"
            assert captured_configs[-1].encoding == speech.RecognitionConfig.AudioEncoding.OGG_OPUS
            assert captured_configs[-1].sample_rate_hertz == 16000

            # 3. MP3 (sample_rate_hertz not set / 0)
            res3 = await service._transcribe_with_google(b"raw_mp3", "audio/mp3")
            assert res3.transcription_text == "పంట సమాచారం"
            assert captured_configs[-1].encoding == speech.RecognitionConfig.AudioEncoding.MP3
            assert captured_configs[-1].sample_rate_hertz == 0

            # 4. AMR (8000 Hz)
            res4 = await service._transcribe_with_google(b"raw_amr", "audio/amr")
            assert res4.transcription_text == "పంట సమాచారం"
            assert captured_configs[-1].encoding == speech.RecognitionConfig.AudioEncoding.AMR
            assert captured_configs[-1].sample_rate_hertz == 8000
