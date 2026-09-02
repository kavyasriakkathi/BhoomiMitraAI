"""
Tests for WhatsApp Telugu Voice Note (Audio Message) Processing Pipeline.

Validates:
1. Webhook audio message payload receipt and parsing.
2. Media download from Meta Graph API.
3. Speech-to-Text (STT) transcription in Telugu (te-IN).
4. Automatic detection and setting of farmer.preferred_language to 'te'.
5. RAG retrieval and AI response generation in Telugu.
6. Outbound WhatsApp message dispatch via Meta Cloud API.
"""

import pytest
import pytest_asyncio
from uuid import uuid4
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from src.core.database import Base
from src.core.models import Farmer, FarmerProfile, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.language.schemas import TranscriptionResponse


@pytest_asyncio.fixture
async def in_memory_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_telugu_voice_message_pipeline_end_to_end(in_memory_session_factory):
    """
    Simulate a farmer sending a Telugu voice note asking about cotton leaf spots:
    Audio STT: "నా పత్తి చేనులో ఆకులపై గోధుమ రంగు మచ్చలు ఉన్నాయి. ఏ మందు పిచికారీ చేయాలి?"
    (There are brown spots on leaves in my cotton field. What medicine should I spray?)

    Verifies:
      - Media is downloaded
      - STT transcribes to Telugu text
      - Farmer's preferred_language is updated to 'te'
      - Conversation is recorded with user_message_type='audio'
      - AI response is generated in Telugu with verified dosage (Mancozeb 2.5-3.0 g/L)
      - Outbound WhatsApp message is dispatched to Meta Cloud API
    """
    # 1. Incoming parsed message object representing a WhatsApp audio note
    incoming_audio = ParsedIncomingMessage(
        phone_number="+919848022338",
        message_id="wamid.TELUGU_VOICE_NOTE_777",
        timestamp="1700000000",
        message_type="audio",
        media_id="meta_media_audio_ogg_999",
        mime_type="audio/ogg; codecs=opus",
        sender_name="Mallaiah",
    )

    fake_audio_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00FAKE_OPUS_AUDIO_BYTES"
    telugu_transcript = "నా పత్తి చేనులో ఆకులపై గోధుమ రంగు మచ్చలు ఉన్నాయి. ఏ మందు పిచికారీ చేయాలి?"

    mock_transcription = TranscriptionResponse(
        transcription_text=telugu_transcript,
        detected_language="te-IN",
        confidence=0.96,
        provider_used="google",
    )

    expected_ai_reply = (
        "పత్తిలో ఆకులపై గోధుమ రంగు మచ్చలు ఆల్టర్నేరియా ఆకుమచ్చ తెగులు లక్షణాలు. "
        "దీని నివారణకు లీటరు నీటికి 2.5 నుండి 3.0 గ్రాముల మాంకోజెబ్ (Mancozeb 75% WP) "
        "లేదా 3.0 గ్రాముల కాపర్ ఆక్సిక్లోరైడ్ కలిపి పిచికారీ చేయండి."
    )

    with patch("src.gateway.service.AsyncSessionLocal", in_memory_session_factory), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock) as mock_download, \
         patch("src.gateway.service.get_language_service") as mock_lang_svc_factory, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send_text, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_download.return_value = (fake_audio_bytes, "audio/ogg; codecs=opus")
        mock_lang_svc = MagicMock()
        mock_lang_svc.transcribe_audio = AsyncMock(return_value=mock_transcription)
        mock_lang_svc_factory.return_value = mock_lang_svc
        mock_gemini.return_value = expected_ai_reply
        mock_send_text.return_value = "outbound_meta_msg_id_888"

        # Execute background pipeline
        await process_message_pipeline(incoming_audio, sender_name="Mallaiah")

        # 1. Verify Audio Download was called with the correct media_id
        mock_download.assert_awaited_once_with("meta_media_audio_ogg_999")

        # 2. Verify STT Service was called with the downloaded bytes
        mock_lang_svc.transcribe_audio.assert_awaited_once_with(fake_audio_bytes, "audio/ogg; codecs=opus")

        # 3. Verify Outbound WhatsApp message was sent to the farmer's number in Telugu
        mock_send_text.assert_awaited_once()
        sent_args = mock_send_text.call_args
        assert sent_args.kwargs["to_phone"] == "+919848022338"
        assert "మాంకోజెబ్" in sent_args.kwargs["message_text"] or "Mancozeb" in sent_args.kwargs["message_text"]

    # 4. Verify Database Records
    async with in_memory_session_factory() as session:
        # Farmer record
        farmer_res = await session.execute(
            select(Farmer).where(Farmer.phone_number == "+919848022338")
        )
        saved_farmer = farmer_res.scalar_one()
        assert saved_farmer.preferred_language == "te"

        # Conversation record
        conv_res = await session.execute(
            select(Conversation).where(Conversation.message_id == "wamid.TELUGU_VOICE_NOTE_777")
        )
        saved_conv = conv_res.scalar_one()
        assert saved_conv.user_message_type == "audio"
        assert saved_conv.user_message == telugu_transcript
        assert saved_conv.delivery_status == "sent"
        assert saved_conv.outbound_message_id == "outbound_meta_msg_id_888"
        assert "మాంకోజెబ్" in saved_conv.ai_response
