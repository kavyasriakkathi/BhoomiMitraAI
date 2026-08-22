"""
End-to-End AI Pipeline Integration Tests for BhoomiMitra.
Verifies the complete flow:
Farmer Message -> Decision Engine -> Gemini (if safe) -> Safety Validator -> Final Farmer Response.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.ai.schemas import AIGenerateRequest
from src.ai.service import AIService
from src.decision_engine.models import DecisionType, RiskLevel
from src.decision_engine.validators import validate_generated_ai_response


@pytest.fixture
def mock_ai_service():
    """Creates an AIService with a mocked repository."""
    mock_repo = MagicMock()
    mock_profile = MagicMock()
    mock_profile.current_crop = None
    mock_profile.district = "Guntur"
    mock_profile.state = "Andhra Pradesh"
    mock_profile.land_size_acres = 3.5

    mock_repo.get_farmer_profile = AsyncMock(return_value=mock_profile)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    mock_repo.session = MagicMock()

    service = AIService(mock_repo)
    return service, mock_repo, mock_profile


@pytest.mark.asyncio
async def test_scenario_1_normal_english_agri_question(mock_ai_service):
    """Scenario 1: Normal English agricultural question."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="My cotton crop has whiteflies. What should I do?"
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "For whitefly in cotton, use yellow sticky traps and spray Neem seed kernel extract (NSKE 5%)."
        response = await service.generate_ai_response(req)

        # 1. Decision Engine evaluated as ANSWER
        # 2. Gemini was called
        assert mock_gemini.called is True
        # 3. Provider used is gemini
        assert response.provider_used == "gemini"
        # 4. Safety validator allowed safe advice
        assert "yellow sticky traps" in response.response_text
        # 5. Final response is safe
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_2_normal_telugu_agri_question(mock_ai_service):
    """Scenario 2: Normal Telugu agricultural question."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="నా పత్తి పంటలో తెల్లదోమ వచ్చింది. ఏం చేయాలి?"
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "పత్తిలో తెల్లదోమ నివారణకు పసుపు రంగు జిగురు అట్టలను ఏర్పాటు చేయండి మరియు వేపనూనెను పిచికారీ చేయండి."
        response = await service.generate_ai_response(req)

        assert mock_gemini.called is True
        assert response.provider_used == "gemini"
        assert "వేపనూనెను" in response.response_text
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_3_dosage_request_blocked(mock_ai_service):
    """Scenario 3: Dosage request -> Decision Engine blocks before Gemini."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="How much pesticide should I spray per acre?"
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:
        response = await service.generate_ai_response(req)

        # Gemini must NOT be called
        assert mock_gemini.called is False
        # Provider is decision_engine
        assert response.provider_used == "decision_engine"
        assert response.intent == DecisionType.SAFE_FALLBACK.value
        # Safe refusal response
        assert "dosage cannot be safely confirmed" in response.response_text.lower() or "consult" in response.response_text.lower()
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_4_unknown_pesticide_blocked(mock_ai_service):
    """Scenario 4: Unknown pesticide -> Decision Engine blocks before Gemini."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="Can I use ABC-999 on cotton?"
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:
        response = await service.generate_ai_response(req)

        # Gemini must NOT be called
        assert mock_gemini.called is False
        assert response.provider_used == "decision_engine"
        assert response.intent == DecisionType.SAFE_FALLBACK.value
        assert "not recognized as a verified agricultural chemical" in response.response_text.lower() or "unverified" in response.response_text.lower()
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_5_missing_crop_context_clarification(mock_ai_service):
    """Scenario 5: Missing crop/context -> clarification response."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="My crop has a problem."
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:
        response = await service.generate_ai_response(req)

        assert mock_gemini.called is False
        assert response.provider_used == "decision_engine"
        assert response.intent == DecisionType.ASK_CLARIFICATION.value
        assert "which crop" in response.response_text.lower() or "crop" in response.response_text.lower()
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_6_non_agricultural_question(mock_ai_service):
    """Scenario 6: Non-agricultural question -> agricultural-only response."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="Write Python code for me."
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:
        response = await service.generate_ai_response(req)

        # Gemini is not called
        assert mock_gemini.called is False
        assert response.provider_used == "decision_engine"
        assert "BhoomiMitra exclusively assists with agriculture" in response.response_text
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_7_unsafe_gemini_output_sanitized(mock_ai_service):
    """Scenario 7: Unsafe Gemini output (banned pesticide) -> Safety Validator replaces it."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="My cotton crop has severe pink bollworm infestation."
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        # Gemini attempts to recommend a banned chemical (Endosulfan)
        mock_gemini.return_value = "Spray Endosulfan 35 EC at 2ml/L to control pink bollworm in cotton."
        response = await service.generate_ai_response(req)

        # Gemini was called
        assert mock_gemini.called is True
        # Safety validator sanitized the output
        assert "Endosulfan" not in response.response_text
        assert "banned or restricted" in response.response_text or "Safety Warning" in response.response_text
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True


@pytest.mark.asyncio
async def test_scenario_8_safe_gemini_output_passed_through(mock_ai_service):
    """Scenario 8: Safe Gemini output -> normal response reaches the farmer."""
    service, _, _ = mock_ai_service
    req = AIGenerateRequest(
        farmer_id=uuid.uuid4(),
        message="My tomato plants have early blight disease."
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        safe_advice = "To manage early blight in tomato, remove infected lower leaves and apply copper oxychloride as per agricultural guidelines."
        mock_gemini.return_value = safe_advice
        response = await service.generate_ai_response(req)

        assert mock_gemini.called is True
        assert response.response_text == safe_advice
        safety_check = validate_generated_ai_response(response.response_text)
        assert safety_check.is_safe is True
