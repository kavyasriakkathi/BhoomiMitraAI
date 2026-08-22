"""
Voice / STT Integration Tests for BhoomiMitra.
Verifies:
1. Successful English transcription.
2. Successful Telugu transcription.
3. STT failure handling (no Gemini call, localized retry).
4. Empty transcription handling (no Gemini call, localized retry).
5. Voice -> Decision Engine SAFE_FALLBACK.
6. Voice -> Normal Gemini response.
7. Voice -> Post-generation Safety Validator interception.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.core.models import Farmer, Conversation
from src.ai.service import process_voice_message
from src.voice.models import VoiceTranscriptionResult
from src.voice.service import VoiceService, BaseSTTProvider


class MockSTTProvider(BaseSTTProvider):
    """Configurable mock STT provider for testing."""

    def __init__(self, result_text: str = "", is_success: bool = True, error_msg: str = None, detected_lang: str = "te-IN"):
        self.result_text = result_text
        self.is_success = is_success
        self.error_msg = error_msg
        self.detected_lang = detected_lang

    async def transcribe(self, audio_bytes: bytes, mime_type: str, language_code: str = None) -> VoiceTranscriptionResult:
        return VoiceTranscriptionResult(
            text=self.result_text,
            detected_language=self.detected_lang,
            confidence=0.95 if self.is_success else 0.0,
            is_success=self.is_success,
            error_message=self.error_msg,
            provider_used="mock",
        )


@pytest.fixture
def mock_db_and_farmer():
    """Sets up a mocked database session and Farmer."""
    mock_db = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.execute = AsyncMock()

    farmer = Farmer(
        id=uuid.uuid4(),
        phone_number="+919876543210",
        preferred_language="te",
    )

    profile = MagicMock()
    profile.current_crop = "Cotton"
    profile.district = "Guntur"
    profile.state = "Andhra Pradesh"
    profile.land_size_acres = 3.5

    mock_exec_result = MagicMock()
    mock_exec_result.scalar_one_or_none.return_value = profile
    mock_exec_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_exec_result)

    conversation = Conversation(
        id=uuid.uuid4(),
        farmer_id=farmer.id,
        user_message="",
        user_message_type="audio",
    )

    return mock_db, farmer, conversation


@pytest.mark.asyncio
async def test_1_successful_english_transcription(mock_db_and_farmer):
    """Test 1: Successful English transcription feeds into existing AI pipeline."""
    mock_db, farmer, conv = mock_db_and_farmer
    farmer.preferred_language = "en"

    mock_provider = MockSTTProvider(result_text="My cotton has whiteflies", is_success=True, detected_lang="en-IN")
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Use yellow sticky traps and spray neem oil."
        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"dummy_ogg_audio_bytes",
            mime_type="audio/ogg",
        )

        assert conv.user_message == "My cotton has whiteflies"
        assert mock_gemini.called is True
        assert "yellow sticky traps" in response


@pytest.mark.asyncio
async def test_2_successful_telugu_transcription(mock_db_and_farmer):
    """Test 2: Successful Telugu transcription feeds into existing AI pipeline."""
    mock_db, farmer, conv = mock_db_and_farmer
    farmer.preferred_language = "te"

    mock_provider = MockSTTProvider(result_text="నా పత్తి పంటలో తెల్లదోమ వచ్చింది", is_success=True, detected_lang="te-IN")
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "పత్తిలో తెల్లదోమ నివారణకు వేపనూనెను పిచికారీ చేయండి."
        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"dummy_ogg_audio_bytes",
            mime_type="audio/ogg",
        )

        assert conv.user_message == "నా పత్తి పంటలో తెల్లదోమ వచ్చింది"
        assert mock_gemini.called is True
        assert "వేపనూనెను" in response


@pytest.mark.asyncio
async def test_3_stt_failure_returns_retry_message(mock_db_and_farmer):
    """Test 3: STT failure does NOT call Gemini and returns localized retry message."""
    mock_db, farmer, conv = mock_db_and_farmer
    farmer.preferred_language = "te"

    mock_provider = MockSTTProvider(result_text="", is_success=False, error_msg="Network timeout")
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"corrupted_audio",
            mime_type="audio/ogg",
        )

        assert mock_gemini.called is False
        assert "వాయిస్ మెసేజ్ స్పష్టంగా వినబడలేదు" in response
        assert conv.ai_response == response


@pytest.mark.asyncio
async def test_4_empty_transcription_returns_retry_message(mock_db_and_farmer):
    """Test 4: Empty transcription does NOT call Gemini and returns localized retry message."""
    mock_db, farmer, conv = mock_db_and_farmer
    farmer.preferred_language = "en"

    mock_provider = MockSTTProvider(result_text="   ", is_success=True)
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"silent_audio",
            mime_type="audio/ogg",
        )

        assert mock_gemini.called is False
        assert "could not clearly understand your voice message" in response
        assert conv.ai_response == response


@pytest.mark.asyncio
async def test_5_voice_followed_by_decision_engine_fallback(mock_db_and_farmer):
    """Test 5: Voice audio asking for ABC-999 dosage triggers Decision Engine SAFE_FALLBACK before Gemini."""
    mock_db, farmer, conv = mock_db_and_farmer

    mock_provider = MockSTTProvider(result_text="What is the ABC-999 dosage for cotton?", is_success=True)
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"audio_bytes",
            mime_type="audio/ogg",
        )

        assert mock_gemini.called is False
        assert "not recognized as a verified agricultural chemical" in response or "unverified" in response


@pytest.mark.asyncio
async def test_6_voice_followed_by_normal_gemini_response(mock_db_and_farmer):
    """Test 6: Normal agricultural voice message triggers Gemini and delivers response."""
    mock_db, farmer, conv = mock_db_and_farmer

    mock_provider = MockSTTProvider(result_text="How to control stem borer in paddy?", is_success=True)
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "To manage stem borer in paddy, use pheromone traps at 8/acre and release Trichogramma egg parasitoids."
        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"audio_bytes",
            mime_type="audio/ogg",
        )

        assert mock_gemini.called is True
        assert "pheromone traps" in response


@pytest.mark.asyncio
async def test_7_voice_followed_by_safety_validator(mock_db_and_farmer):
    """Test 7: Voice message where Gemini attempts to recommend banned pesticide is intercepted by Safety Validator."""
    mock_db, farmer, conv = mock_db_and_farmer

    mock_provider = MockSTTProvider(result_text="My cotton has severe bollworm pest issue", is_success=True)
    mock_voice_service = VoiceService(provider=mock_provider)

    with patch("src.voice.service.get_voice_service", return_value=mock_voice_service), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        # Gemini returns banned Monocrotophos
        mock_gemini.return_value = "Spray Monocrotophos 36 SL to kill all pests."
        response = await process_voice_message(
            db=mock_db,
            farmer=farmer,
            conversation=conv,
            audio_bytes=b"audio_bytes",
            mime_type="audio/ogg",
        )

        assert mock_gemini.called is True
        assert "Monocrotophos" not in response
        assert "Safety Warning" in response or "banned or restricted" in response
