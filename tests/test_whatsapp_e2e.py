"""
Real WhatsApp End-to-End Flow Tests for BhoomiMitra.
Verifies the complete flow through the gateway pipeline:
WhatsApp Webhook / Media -> STT (if voice) -> Decision Engine -> Gemini (if safe) -> Safety Validator -> Outbound WhatsApp Send.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from contextlib import asynccontextmanager

from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.core.models import Farmer, Conversation
from src.voice.models import VoiceTranscriptionResult


@pytest.fixture
def mock_pipeline_db():
    """Sets up a mock database session for the message pipeline."""
    mock_db = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.refresh = AsyncMock()

    farmer = Farmer(
        id=uuid.uuid4(),
        phone_number="919876543210",
        preferred_language="te",
    )

    profile = MagicMock()
    profile.current_crop = "Cotton"
    profile.district = "Guntur"
    profile.state = "Andhra Pradesh"
    profile.land_size_acres = 3.5
    async def mock_execute(stmt):
        res = MagicMock()
        stmt_str = str(stmt)
        if "farmer_profile" in stmt_str.lower():
            res.scalar_one_or_none.return_value = profile
        elif "crop" in stmt_str.lower():
            res.scalar_one_or_none.return_value = uuid.uuid4()
        else:
            res.scalar_one_or_none.return_value = profile
        res.scalars.return_value.all.return_value = []
        return res

    mock_db.execute = mock_execute

    @asynccontextmanager
    async def mock_session_local():
        yield mock_db

    return mock_db, farmer, profile, mock_session_local


@pytest.mark.asyncio
async def test_scenario_1_whatsapp_text_to_ai_response(mock_pipeline_db):
    """Scenario 1: WhatsApp text -> AI response delivered via WhatsApp outbound send."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT001",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="My cotton crop has whitefly infestation",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_001") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Use yellow sticky traps and neem spray."
        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is True
        assert mock_send.called is True
        sent_phone, sent_text = mock_send.call_args[1]["to_phone"], mock_send.call_args[1]["message_text"]
        assert sent_phone == "919876543210"
        assert "yellow sticky traps" in sent_text


@pytest.mark.asyncio
async def test_scenario_2_whatsapp_telugu_text_response(mock_pipeline_db):
    """Scenario 2: Telugu text -> Telugu AI response."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT002",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="నా పత్తి పంటలో తెల్లదోమ వచ్చింది",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_002") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "పత్తిలో తెల్లదోమ నివారణకు వేపనూనె పిచికారీ చేయండి."
        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is True
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "వేపనూనె" in sent_text


@pytest.mark.asyncio
async def test_scenario_3_whatsapp_voice_stt_ai_pipeline(mock_pipeline_db):
    """Scenario 3: WhatsApp voice -> STT -> AI -> WhatsApp response."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLAUDIO003",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="audio",
        media_id="media_audio_123",
        media_mime_type="audio/ogg; codecs=opus",
    )

    voice_result = VoiceTranscriptionResult(
        text="How to control whiteflies in cotton?",
        detected_language="en-IN",
        is_success=True,
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"ogg_bytes", "audio/ogg")), \
         patch("src.voice.service.VoiceService.transcribe_audio", new_callable=AsyncMock, return_value=voice_result), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_003") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Deploy yellow sticky traps and apply botanical insecticides."
        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is True
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "yellow sticky traps" in sent_text


@pytest.mark.asyncio
async def test_scenario_4_dosage_question_blocked(mock_pipeline_db):
    """Scenario 4: Dosage question -> Decision Engine blocks it before Gemini."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT004",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="How much chemical dose should I spray per acre?",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_004") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is False
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "dosage cannot be safely confirmed" in sent_text.lower() or "consult" in sent_text.lower()


@pytest.mark.asyncio
async def test_scenario_5_unknown_pesticide_blocked(mock_pipeline_db):
    """Scenario 5: Unknown pesticide such as ABC-999 -> blocked by Decision Engine."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT005",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="Can I spray ABC-999 on cotton?",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_005") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is False
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "not recognized" in sent_text.lower() or "unverified" in sent_text.lower()


@pytest.mark.asyncio
async def test_scenario_6_normal_farming_question_passes(mock_pipeline_db):
    """Scenario 6: Normal farming question -> Gemini -> Safety Validator -> WhatsApp response."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT006",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="My cotton crop has aphids. What is the organic treatment?",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_006") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Spray 5% neem seed kernel extract or 10,000 ppm Azadirachtin."
        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is True
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "neem" in sent_text.lower()


@pytest.mark.asyncio
async def test_scenario_7_unsafe_gemini_recommendation_replaced(mock_pipeline_db):
    """Scenario 7: Unsafe Gemini recommendation (banned chemical) -> Safety Validator replaces it."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLTEXT007",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="text",
        text_content="My cotton has heavy pink bollworm infestation",
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_007") as mock_send, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Spray Endosulfan 35 EC across the crop."
        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_gemini.called is True
        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "Endosulfan" not in sent_text
        assert "Safety Warning" in sent_text or "banned or restricted" in sent_text


@pytest.mark.asyncio
async def test_scenario_8_image_crop_diagnosis_pipeline(mock_pipeline_db):
    """Scenario 8: Image message -> existing crop diagnosis flow -> WhatsApp response."""
    mock_db, farmer, _, mock_session_local = mock_pipeline_db
    parsed = ParsedIncomingMessage(
        message_id="wamid.HBgLIMAGE008",
        phone_number="919876543210",
        timestamp="1700000000",
        message_type="image",
        media_id="media_img_888",
        media_mime_type="image/jpeg",
        text_content="Analyze my leaf photo",
    )

    vision_json = (
        '{"disease_name": "Leaf Curl Virus", "confidence_score": 0.92, '
        '"severity": "medium", "symptoms": "Upward curling and thickening of leaf veins", '
        '"treatment_recommendation": "Remove infected plants and control whiteflies", '
        '"friendly_whatsapp_reply": "Your crop shows symptoms of Leaf Curl Virus. Please control whiteflies."}'
    )

    with patch("src.gateway.service.AsyncSessionLocal", mock_session_local), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"jpeg_bytes", "image/jpeg")), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="outbound_msg_008") as mock_send, \
         patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=vision_json), \
         patch("src.crop_health.service.CropHealthService.create_diagnosis", new_callable=AsyncMock), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        await process_message_pipeline(parsed, sender_name="Ramesh")

        assert mock_send.called is True
        sent_text = mock_send.call_args[1]["message_text"]
        assert "Leaf Curl Virus" in sent_text
