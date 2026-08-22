"""
Regression Tests for Live WhatsApp Issue Fixes:
1. Telugu-only response (no English leakage or mixed language)
2. English-only response
3. No mixed-language output (stripping English parenthetical advice)
4. No unsafe chemical dosage / technical formulations (e.g., 22.9 EC, 10% EC @ 2 ml/l)
5. Unknown pesticide refusal (ABC-999)
6. Missing crop-stage clarification for stage-specific nutrient questions
7. Follow-up conversation context preservation
8. UTF-8 Telugu text preservation & cleaning of truncated leading vowel signs
9. Response not unexpectedly truncated
10. Post-generation safety validator execution before WhatsApp dispatch
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.decision_engine.validators import (
    validate_generated_ai_response,
    clean_telugu_text,
    is_mixed_language_output,
    detect_unsafe_chemical_formulations,
)
from src.decision_engine.engine import DecisionEngine
from src.decision_engine.models import FarmerInput, DecisionType
from src.ai.schemas import AIGenerateRequest
from src.ai.service import AIService, process_text_message
from src.core.models import Farmer, Conversation


@pytest.fixture
def mock_repo_and_farmer():
    """Mock repository and farmer context for AI service."""
    mock_db = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

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

    mock_repo = MagicMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=profile)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    mock_repo.session = mock_db

    return mock_repo, farmer, profile, mock_db


def test_1_clean_telugu_text_removes_english_parentheticals():
    """Test 1: clean_telugu_text removes parenthetical English expressions."""
    raw_mixed = "పత్తిలో తెల్లదోమ నివారణకు వేపనూనె పిచికారీ చేయండి. (If nymph population is high, spray Pyriproxyfen 10% EC @ 2 ml/l)"
    cleaned = clean_telugu_text(raw_mixed)
    assert "(If nymph population" not in cleaned
    assert "Pyriproxyfen" not in cleaned
    assert "వేపనూనె" in cleaned


def test_2_clean_telugu_text_fixes_truncated_leading_vowel_signs():
    """Test 2: clean_telugu_text removes orphaned leading Telugu vowel signs from truncated text."""
    truncated = "ెన్ 22.9 EC కలిపి పిచికారీ చేయాలి. మీ పత్తి"
    cleaned = clean_telugu_text(truncated)
    # The leading dependent vowel 'ె' without consonant is stripped
    assert not cleaned.startswith("ె")


def test_3_detect_unsafe_chemical_formulations():
    """Test 3: detect_unsafe_chemical_formulations identifies unverified EC/SC/WP rates."""
    unsafe_text = "Spray Spiromesifen 22.9% SC @ 1 ml/l or Pyriproxyfen 10% EC"
    detected = detect_unsafe_chemical_formulations(unsafe_text)
    assert len(detected) > 0
    assert any("22.9" in d or "10%" in d or "ml" in d for d in detected)


def test_4_safety_validator_intercepts_unsafe_chemical_rates():
    """Test 4: validate_generated_ai_response replaces unverified chemical percentages with safe advice."""
    unsafe_resp = "మీ పత్తిలో స్పైరోమెసిఫెన్ 22.9 EC కలిపి పిచికారీ చేయాలి."
    res = validate_generated_ai_response(
        response_text=unsafe_resp,
        user_message="నా పత్తి పంటలో తెల్లదోమ వచ్చింది",
    )
    assert res.is_safe is False
    assert "unsafe_formulation" in res.violations
    assert "22.9 EC" not in res.safe_response
    assert "అధికారిక ఉత్పత్తి లేబుల్" in res.safe_response or "వ్యవసాయ అధికారి" in res.safe_response


def test_5_safety_validator_rejects_mixed_language_output():
    """Test 5: validate_generated_ai_response intercepts heavily mixed language in Telugu replies."""
    mixed_resp = "వేపనూనె spray చేయండి. If severe attack occurs, consult local agriculture officer immediately for best chemical treatment."
    res = validate_generated_ai_response(
        response_text=mixed_resp,
        user_message="నా పంటలో పురుగులు ఉన్నాయి",
    )
    assert res.is_safe is False
    assert "mixed_language" in res.violations
    # Safe response must be in Telugu
    assert any("\u0C00" <= ch <= "\u0C7F" for ch in res.safe_response)


def test_6_unknown_pesticide_blocked():
    """Test 6: Decision engine blocks synthetic unknown pesticide ABC-999."""
    engine = DecisionEngine()
    farmer_input = FarmerInput(
        message="Can I use ABC-999 on cotton?",
        crop="Cotton",
        growth_stage="Flowering",
        problem="Pest",
        location="Guntur",
    )
    decision = engine.evaluate(farmer_input)
    assert decision.decision_type == DecisionType.SAFE_FALLBACK
    assert "ABC-999" in decision.reasons[0]


def test_7_missing_crop_stage_clarification_for_fertilizer():
    """Test 7: Stage-dependent fertilizer questions trigger clarification for growth stage."""
    engine = DecisionEngine()
    farmer_input = FarmerInput(
        message="వరి పంటలో యూరియా ఎప్పుడు వేయాలి? fertilizer schedule",
        crop="వరి",
        growth_stage=None,
        problem=None,
        location="నల్గొండ",
    )
    decision = engine.evaluate(farmer_input)
    assert decision.decision_type == DecisionType.ASK_CLARIFICATION
    assert "దశ" in decision.response


@pytest.mark.asyncio
async def test_8_follow_up_conversation_context_preservation(mock_repo_and_farmer):
    """Test 8: Follow-up question 'What should I do?' inherits crop & problem from conversation history."""
    mock_repo, farmer, profile, _ = mock_repo_and_farmer
    profile.current_crop = None  # Not in profile

    # Past conversation established cotton and whiteflies
    past_conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=farmer.id,
        user_message="My cotton crop has whiteflies",
        ai_response="Whiteflies suck sap from leaves.",
    )
    mock_repo.get_conversation_history = AsyncMock(return_value=[past_conv])

    service = AIService(mock_repo)

    # Farmer now asks a brief follow-up: "What should I do?"
    req = AIGenerateRequest(
        farmer_id=farmer.id,
        message="What should I do?",
    )

    with patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        mock_gemini.return_value = "Apply neem oil (5 ml per litre of water) and install yellow sticky traps."
        resp = await service.generate_ai_response(req)

        # Gemini was called because DecisionEngine understood crop=Cotton and problem=whiteflies from history!
        assert mock_gemini.called is True
        assert "neem oil" in resp.response_text.lower()


@pytest.mark.asyncio
async def test_9_telugu_response_purity_end_to_end(mock_repo_and_farmer):
    """Test 9: Telugu input receives a pure Telugu response with no leaked English sentences or unverified chemicals."""
    mock_repo, farmer, profile, mock_db = mock_repo_and_farmer

    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=farmer.id,
        user_message="నా పత్తి పంటలో తెల్లదోమ వచ్చింది",
    )

    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    with patch("src.ai.service.AIRepository", return_value=mock_repo), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        # Gemini returns pure Telugu advice
        mock_gemini.return_value = "పత్తిలో తెల్లదోమ నివారణకు ఎకరాకు 10 పసుపు జిగురు అట్టలు అమర్చండి. అలాగే 5% వేప గింజల కషాయం పిచికారీ చేయండి."
        response_text = await process_text_message(mock_db, farmer, conv)

        assert "పసుపు జిగురు అట్టలు" in response_text
        assert "వేప గింజల కషాయం" in response_text
        # Ensure no English characters
        assert not any(ord('a') <= ord(c) <= ord('z') or ord('A') <= ord(c) <= ord('Z') for c in response_text.replace("EC", "").replace("WP", ""))


@pytest.mark.asyncio
async def test_10_safety_validator_runs_before_dispatch_in_process_text(mock_repo_and_farmer):
    """Test 10: If Gemini outputs banned chemical or unsafe rate in process_text_message, Safety Validator intercepts it."""
    mock_repo, farmer, profile, mock_db = mock_repo_and_farmer

    conv = Conversation(
        id=uuid.uuid4(),
        farmer_id=farmer.id,
        user_message="నా పత్తి పంటలో పురుగులు ఉన్నాయి",
    )

    with patch("src.ai.service.AIRepository", return_value=mock_repo), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", new_callable=AsyncMock):

        # Gemini attempts to recommend Monocrotophos
        mock_gemini.return_value = "మోనోక్రోటోఫాస్ 36 SL పిచికారీ చేయండి."
        final_text = await process_text_message(mock_db, farmer, conv)

        # Monocrotophos is blocked and replaced
        assert "మోనోక్రోటోఫాస్" not in final_text
        assert "నిషేధించబడిన" in final_text or "హెచ్చరిక" in final_text
