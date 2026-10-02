"""
BhoomiMitra AI — Omni-Channel Voice Response Pipeline Tests

Verifies that for EVERY farmer query (Voice, Text, and Image):
1. Inbound voice -> STT -> single AI response -> WhatsApp TEXT + WhatsApp AUDIO
2. Inbound text -> single AI response -> WhatsApp TEXT + WhatsApp AUDIO
3. Inbound image -> image/vision analysis -> single AI response -> WhatsApp TEXT + WhatsApp AUDIO
4. Text and Audio use the EXACT SAME ai_response (no secondary LLM call)
5. TTS failure fails soft (text response still delivered)
6. Audio upload / send failure fails soft (text response still delivered)
7. No second Gemini call is made for TTS
8. Preferred language resolution: Telugu farmer -> Telugu TTS ("te")
9. Preferred language resolution: English farmer -> English TTS ("en")
10. Image crop diagnosis pipeline produces diagnosis and dispatches text + audio
11. Explicit administrator disable override (enable_voice_responses=False) suppresses audio
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import Farmer, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.language.schemas import TranscriptionResponse


class TestOmniChannelVoiceResponses:
    """Test suite covering WhatsApp voice responses for Voice, Text, and Image inbound queries."""

    def _setup_db_cm(self, farmer: Farmer, conv: Conversation):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        mock_db_cm = AsyncMock()
        mock_db_cm.__aenter__.return_value = mock_db
        mock_db_cm.__aexit__.return_value = None
        return mock_db_cm, mock_db

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Voice Inbound -> Text + Audio Outbound
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_voice_message_dispatches_text_and_audio(self):
        """Voice inbound: STT -> 1 AI response -> WhatsApp text + WhatsApp audio."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.VOICE_OMNI_01",
            timestamp="1700000000",
            message_type="audio",
            media_id="audio_media_omni_01",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        expected_reply = "వరిలో అగ్గితెగులు నివారణకు ట్రైసైక్లాజోల్ 0.6 గ్రాములు లీటరు నీటికి కలపండి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"voice_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_reply) as mock_proc_text, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_V1") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_v1") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_V1") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="వరిలో అగ్గితెగులు ఉంది", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_voice_audio_bytes"])

            await process_message_pipeline(parsed)

            # 1 AI generation
            mock_proc_text.assert_awaited_once()
            # WhatsApp text sent
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=expected_reply)
            # WhatsApp audio synthesized with exact same text & language
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(expected_reply, "te")
            mock_upload.assert_awaited_once_with(b"OggS_voice_audio_bytes", mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_audio_v1")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Text Inbound -> Text + Audio Outbound
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_text_message_dispatches_text_and_audio(self):
        """Text inbound: single AI response -> WhatsApp text + WhatsApp audio."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.TEXT_OMNI_01",
            timestamp="1700000000",
            message_type="text",
            text_content="వరి సాగు సలహాలు ఇవ్వండి",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id, user_message=parsed.text_content, user_message_type="text")
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        expected_reply = "వరి సాగులో సరైన నీటి యాజమాన్యం మరియు ఎరువుల మోతాదు పాటించండి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_reply) as mock_proc_text, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_T1") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_t1") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_T1") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_text_audio_bytes"])

            await process_message_pipeline(parsed)

            mock_proc_text.assert_awaited_once()
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=expected_reply)
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(expected_reply, "te")
            mock_upload.assert_awaited_once_with(b"OggS_text_audio_bytes", mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_audio_t1")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Image Inbound (without voice/text) -> Text + Audio Outbound
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_image_message_without_voice_dispatches_text_and_audio(self):
        """Image inbound without voice: vision analysis -> 1 AI response -> WhatsApp text + WhatsApp audio."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.IMAGE_OMNI_01",
            timestamp="1700000000",
            message_type="image",
            media_id="img_media_999",
            media_mime_type="image/jpeg",
            text_content=None,
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="image")
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        expected_diagnosis = "చిత్రం పరిశీలించిన తర్వాత, ఆకులపై కనిపించే మచ్చలు సర్కోస్పోరా ఆకుమచ్చ తెగులు లక్షణాలతో సమానంగా కనిపిస్తున్నాయి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"\xff\xd8\xffimage_bytes", "image/jpeg")), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=expected_diagnosis) as mock_proc_image, \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_IMG") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_img") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_IMG") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_image_audio_bytes"])

            await process_message_pipeline(parsed)

            # Vision diagnosis called once
            mock_proc_image.assert_awaited_once()
            # WhatsApp text sent
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=expected_diagnosis)
            # WhatsApp audio synthesized with exact same diagnosis text in preferred language
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(expected_diagnosis, "te")
            mock_upload.assert_awaited_once_with(b"OggS_image_audio_bytes", mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_audio_img")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Text and Audio Use EXACT SAME ai_response (No Variation)
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["audio", "text", "image"])
    async def test_text_and_audio_use_exact_same_ai_response(self, msg_type):
        """Verifies text and audio receive the exact identical ai_response string across all message types."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.SAME_RESP_{msg_type.upper()}",
            timestamp="1700000000",
            message_type=msg_type,
            media_id=f"media_{msg_type}_123" if msg_type in ("audio", "image") else None,
            text_content="పంట సలహా" if msg_type == "text" else None,
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        single_generated_answer = f"ఖచ్చితమైన సమాధానం — {msg_type}: వేప నూనె 5 మి.లీ లీటరు నీటికి పిచికారీ చేయండి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"raw_bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=single_generated_answer), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=single_generated_answer), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OK") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_ok") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_OK") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సలహా", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_audio"])

            await process_message_pipeline(parsed)

            # Assert text sent
            sent_text_arg = mock_send_text.call_args[1]["message_text"]
            # Assert TTS input text
            tts_text_arg = mock_lang_svc.return_value.synthesize_speech.call_args[0][0]

            assert sent_text_arg == single_generated_answer
            assert tts_text_arg == single_generated_answer
            assert sent_text_arg == tts_text_arg

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Fail-Soft: TTS Failure Still Delivers Text
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["audio", "text", "image"])
    async def test_tts_failure_still_delivers_text(self, msg_type):
        """TTS failure logs warning and preserves successful WhatsApp text delivery."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.TTS_FAIL_{msg_type.upper()}",
            timestamp="1700000000",
            message_type=msg_type,
            media_id=f"media_{msg_type}_fail" if msg_type in ("audio", "image") else None,
            text_content="సలహా కావాలి" if msg_type == "text" else None,
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        expected_reply = "సారవంతమైన నేల కోసం సేంద్రీయ ఎరువులను వాడండి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"bytes", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_reply), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=expected_reply), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_DELIVERED") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సలహా", detected_language="te")
            )
            # TTS raises exception
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(side_effect=RuntimeError("Google Cloud TTS quota exceeded"))

            await process_message_pipeline(parsed)

            # Text MUST be delivered
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=expected_reply)
            # Audio upload and send are skipped cleanly
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()
            # Conversation delivery status is sent
            assert conv.delivery_status == "sent"

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Fail-Soft: Audio Upload / Send Failure Still Delivers Text
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    @pytest.mark.parametrize("fail_point", ["upload_none", "upload_exc", "send_none", "send_exc"])
    async def test_audio_delivery_failure_still_delivers_text(self, fail_point):
        """Audio upload or send failure logs fail-soft warning and preserves text delivery."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.AUDIO_FAIL_01",
            timestamp="1700000000",
            message_type="text",
            text_content="మార్కెట్ ధరలు చెప్పండి",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        expected_reply = "ఈరోజు వరి క్వింటాలు ధర ₹2,200 ఉంది."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_reply), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OUT_OK") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_market_audio"])

            if fail_point == "upload_none":
                mock_upload.return_value = None
            elif fail_point == "upload_exc":
                mock_upload.side_effect = ConnectionError("Meta media endpoint unreachable")
            elif fail_point == "send_none":
                mock_upload.return_value = "meta_media_valid"
                mock_send_audio.return_value = None
            elif fail_point == "send_exc":
                mock_upload.return_value = "meta_media_valid"
                mock_send_audio.side_effect = TimeoutError("Meta send timeout")

            await process_message_pipeline(parsed)

            # Text MUST be delivered successfully
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=expected_reply)
            assert conv.delivery_status == "sent"

    # ─────────────────────────────────────────────────────────────────────────
    # 7. No Second Gemini Call for TTS
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["audio", "text", "image"])
    async def test_no_second_gemini_call_for_tts(self, msg_type):
        """Verify that Gemini is NEVER called again for TTS audio generation."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.NO_2ND_GEMINI_{msg_type.upper()}",
            timestamp="1700000000",
            message_type=msg_type,
            media_id=f"media_{msg_type}_gemini" if msg_type in ("audio", "image") else None,
            text_content="పంట తెగులు సమాచారం" if msg_type == "text" else None,
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        mock_reply = "పురుగు నివారణకు వేపనూనె పిచికారీ చేయండి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"data", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=mock_reply) as mock_text_gen, \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=mock_reply) as mock_image_gen, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_OK"), \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_ok"), \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_OK"), \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.ai.gemini_client.generate_response", new_callable=AsyncMock) as mock_gemini_direct, \
             patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock) as mock_multimodal_direct, \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సమాచారం", detected_language="te")
            )
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_voice"])

            await process_message_pipeline(parsed)

            # Gemini direct clients were NOT invoked by TTS stage
            mock_gemini_direct.assert_not_called()
            mock_multimodal_direct.assert_not_called()

            # Generation service called exactly once for the message
            if msg_type in ("audio", "text"):
                assert mock_text_gen.await_count == 1
                mock_image_gen.assert_not_called()
            else:
                assert mock_image_gen.await_count == 1
                mock_text_gen.assert_not_called()

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Preferred Language Resolution: Telugu Farmer -> Telugu TTS
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_telugu_preferred_language_resolves_telugu_tts(self):
        """Farmer with Telugu preference receives Telugu text and Telugu TTS."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.LANG_TE_01",
            timestamp="1700000000",
            message_type="text",
            text_content="వరిలో తెగులు",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        te_response = "వరి పంటలో అగ్గితెగులు నివారణ సలహా."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=te_response), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_TE"), \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_te"), \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_TE"), \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_te_audio"])

            await process_message_pipeline(parsed)

            # synthesize_speech called with "te"
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(te_response, "te")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Preferred Language Resolution: English Farmer -> English TTS
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_english_preferred_language_resolves_english_tts(self):
        """Farmer with English preference receives English text and English TTS."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.LANG_EN_01",
            timestamp="1700000000",
            message_type="text",
            text_content="How to control blast disease in paddy?",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="en")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        en_response = "Apply Tricyclazole 0.6g per litre of water to control paddy blast disease."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=en_response), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_EN"), \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_en"), \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.AUDIO_EN"), \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_en_audio"])

            await process_message_pipeline(parsed)

            # synthesize_speech called with "en"
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(en_response, "en")

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Image Diagnosis Still Works (End-to-End Image Handler Check)
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_image_diagnosis_pipeline_with_caption_and_voice(self):
        """Image message with caption correctly executes vision pipeline and dispatches text + audio."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id="wamid.IMG_DIAG_01",
            timestamp="1700000000",
            message_type="image",
            media_id="media_chilli_leaf_01",
            media_mime_type="image/jpeg",
            text_content="మిరప ఆకుపై మచ్చలు",
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id, user_message=parsed.text_content, user_message_type="image")
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        diagnosis_response = "మిరప ఆకుపై మచ్చలు సెర్కోస్పోరా ఆకుమచ్చ తెగులుగా కనిపిస్తున్నాయి."

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"\xff\xd8fake_jpg", "image/jpeg")), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value=diagnosis_response) as mock_proc_img, \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.IMG_TEXT") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_img_audio") as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.IMG_AUDIO") as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            mock_settings.return_value.enable_voice_responses = True
            mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_chilli_audio"])

            await process_message_pipeline(parsed)

            mock_proc_img.assert_awaited_once()
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text=diagnosis_response)
            mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(diagnosis_response, "te")
            mock_upload.assert_awaited_once_with(b"OggS_chilli_audio", mime_type="audio/ogg")
            mock_send_audio.assert_awaited_once_with("919876543210", "meta_img_audio")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Administrator Disable Setting (enable_voice_responses=False)
    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["audio", "text", "image"])
    async def test_explicit_admin_disable_suppresses_voice_for_all_types(self, msg_type):
        """When enable_voice_responses=False, audio is completely suppressed for Voice, Text, and Image."""
        parsed = ParsedIncomingMessage(
            phone_number="919876543210",
            message_id=f"wamid.ADMIN_DISABLED_{msg_type.upper()}",
            timestamp="1700000000",
            message_type=msg_type,
            media_id=f"media_{msg_type}" if msg_type in ("audio", "image") else None,
            text_content="సలహా" if msg_type == "text" else None,
        )
        farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210", preferred_language="te")
        conv = Conversation(id=uuid.uuid4(), farmer_id=farmer.id, message_id=parsed.message_id)
        mock_db_cm, _ = self._setup_db_cm(farmer, conv)

        with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
             patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
             patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
             patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"data", "audio/ogg")), \
             patch("src.gateway.service.get_language_service") as mock_lang_svc, \
             patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value="టెక్స్ట్ మాత్రమే."), \
             patch("src.gateway.service.process_image_message", new_callable=AsyncMock, return_value="టెక్స్ట్ మాత్రమే."), \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.TEXT_ONLY") as mock_send_text, \
             patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
             patch("src.gateway.service.send_audio_message", new_callable=AsyncMock) as mock_send_audio, \
             patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
             patch("src.gateway.service.get_settings") as mock_settings:

            # Explicit disable
            mock_settings.return_value.enable_voice_responses = False
            mock_lang_svc.return_value.transcribe_audio = AsyncMock(
                return_value=TranscriptionResponse(provider_used="google", transcription_text="సలహా", detected_language="te")
            )

            await process_message_pipeline(parsed)

            # Text delivered
            mock_send_text.assert_awaited_once_with(to_phone="919876543210", message_text="టెక్స్ట్ మాత్రమే.")
            # Audio never uploaded or sent
            mock_upload.assert_not_called()
            mock_send_audio.assert_not_called()
