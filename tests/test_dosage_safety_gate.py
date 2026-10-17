"""
Focused Tests for Strengthened Multi-Attribute Chemical & Dosage Safety Validation Gate.

Validates that every pesticide/chemical dosage recommendation strictly matches retrieved ground truth by:
  - crop
  - disease/pest
  - chemical/active ingredient
  - formulation when present
  - numeric dosage
  - unit

Tests covered:
  a. exact verified chemical + dosage -> PASS
  b. correct number but wrong chemical -> FAIL
  c. correct chemical but wrong dosage -> FAIL
  d. dosage from unrelated disease -> FAIL
  e. unverified chemical -> FAIL
  f. fertilizer dosage safety preserved -> PASS / FAIL
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from src.ai.schemas import AIGenerateRequest
from src.ai.service import (
    AIService,
    validate_output_chemical_dosage_safety,
)

RAG_SNIPPETS = [
    (
        "• Document: ICAR / PJTSAU Cotton Alternaria Leaf Spot Disease Management Guide "
        "(Crop: Cotton, Source: ICAR-CICR / PJTSAU Package of Practices): "
        "Title: Cotton Alternaria Leaf Spot Disease Management Guide. "
        "Crop: Cotton (Gossypium hirsutum / Gossypium spp.). "
        "Disease: Alternaria Leaf Spot (ఆల్టర్నేరియా ఆకుమచ్చ తెగులు / ఆల్టర్నేరియా ఆకు మచ్చ). "
        "Causal Agents: Alternaria macrospora / Alternaria alternata. Disease Type: Fungal disease. "
        "Symptoms: Small, circular or irregular brown spots with concentric rings and purple margins on cotton leaves. "
        "Control & Verified Dosage Information: "
        "1. Mancozeb 75% WP: Dosage: 2.5 to 3.0 g per litre of water (Approximate field quantity: 500 to 600 g per acre in 200 L water). "
        "2. Copper Oxychloride 50% WP: Dosage: 3.0 g per litre of water. "
        "3. Propiconazole 25% EC: Dosage: 1.0 ml per litre of water. "
        "4. Pyraclostrobin 20% WG: Dosage: 1.0 g per litre of water. "
        "5. Azoxystrobin + Difenoconazole: Dosage: 1.0 ml per litre of water."
    ),
    (
        "• Document: ICAR / PJTSAU Cotton Pink Bollworm Management Guide "
        "(Crop: Cotton, Source: ICAR-CICR / PJTSAU): "
        "Title: Cotton Pink Bollworm Management. Crop: Cotton. "
        "Pest: Pink Bollworm (గులాబీ రంగు పురుగు - Pectinophora gossypiella). "
        "Symptoms: Rosette flowers, premature boll opening, damaged locules with pink larvae inside. "
        "Control & Verified Dosage: "
        "1. Install pheromone traps @ 8 per acre for monitoring. "
        "2. Profenofos 50% EC: 2.0 ml per litre of water. "
        "3. Emamectin Benzoate 5% SG: 0.4 g per litre of water. "
        "4. Spinetoram 11.7% SC: 0.8 to 1.0 ml per litre of water."
    ),
]


# =========================================================================
# Unit Tests for validate_output_chemical_dosage_safety
# =========================================================================

def test_a_exact_verified_chemical_dosage_passes():
    """a. Exact verified chemical + formulation + dosage + unit matches RAG -> PASS."""
    # English
    valid_en = "For Alternaria Leaf Spot on cotton, spray Mancozeb 75% WP at 2.5 to 3.0 g per litre of water."
    is_valid, reason = validate_output_chemical_dosage_safety(valid_en, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid is True
    assert reason is None

    # Telugu
    valid_te = "పత్తిలో ఆల్టర్నేరియా ఆకుమచ్చ తెగులు నివారణకు లీటరు నీటికి 2.5 నుండి 3.0 గ్రాముల మాంకోజెబ్ (Mancozeb 75% WP) కలిపి పిచికారీ చేయండి."
    is_valid_te, reason_te = validate_output_chemical_dosage_safety(valid_te, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid_te is True
    assert reason_te is None


def test_b_correct_number_but_wrong_chemical_fails():
    """b. Correct numeric value (2.5) borrowed for an unverified / wrong chemical -> FAIL."""
    # Glyphosate using Mancozeb's 2.5 number
    fake_chem = "For Alternaria Leaf Spot on cotton, spray Glyphosate at 2.5 g per litre of water."
    is_valid, reason = validate_output_chemical_dosage_safety(fake_chem, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid is False
    assert "glyphosate" in reason.lower()

    # Cartap (paddy chemical) using Mancozeb's 2.5 number
    wrong_chem = "For Alternaria Leaf Spot on cotton, spray Cartap Hydrochloride at 2.5 g per litre of water."
    is_valid_c, reason_c = validate_output_chemical_dosage_safety(wrong_chem, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid_c is False


def test_c_correct_chemical_but_wrong_dosage_fails():
    """c. Correct verified chemical (Mancozeb) but wrong/hallucinated dosage (10 g/L) -> FAIL."""
    wrong_dosage = "For Alternaria Leaf Spot on cotton, spray Mancozeb 75% WP at 10 g per litre of water."
    is_valid, reason = validate_output_chemical_dosage_safety(wrong_dosage, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid is False
    assert "mancozeb" in reason.lower()
    assert "10" in reason


def test_d_dosage_from_unrelated_disease_fails():
    """d. Chemical and dosage borrowed from a different disease/pest snippet -> FAIL."""
    # Profenofos 2.0 ml/L is in RAG for Pink Bollworm, NOT for Alternaria Leaf Spot
    unrelated_disease = "For Alternaria Leaf Spot on cotton, spray Profenofos 50% EC at 2.0 ml per litre of water."
    is_valid, reason = validate_output_chemical_dosage_safety(unrelated_disease, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid is False
    assert "profenofos" in reason.lower()


def test_e_unverified_chemical_fails():
    """e. Recommending an ungrounded chemical not present in RAG ground truth -> FAIL."""
    unverified = "For Alternaria Leaf Spot on cotton, spray Hexaconazole at 2.0 ml per litre of water."
    is_valid, reason = validate_output_chemical_dosage_safety(unverified, RAG_SNIPPETS, effective_crop="Cotton")
    assert is_valid is False
    assert "hexaconazole" in reason.lower()


def test_fertilizer_dosage_safety_preserved():
    """f. Fertilizer dosage safety remains functional: valid passes, hallucinated fails."""
    fert_snippets = [
        "• Document: Cotton Fertilizer Guide (Crop: Cotton): Apply 25 to 30 kg Urea per acre at flowering."
    ]
    # Valid fertilizer
    ok, _ = validate_output_chemical_dosage_safety("Apply 25 to 30 kg Urea per acre.", fert_snippets, effective_crop="Cotton")
    assert ok is True

    # Hallucinated fertilizer dosage
    bad, reason = validate_output_chemical_dosage_safety("Apply 150 kg Urea per acre.", fert_snippets, effective_crop="Cotton")
    assert bad is False
    assert "150" in reason


# =========================================================================
# Integration Tests: AIService.generate_ai_response fallback enforcement
# =========================================================================

@pytest.mark.asyncio
async def test_aiservice_replaces_invalid_chemical_with_safe_fallback():
    """Verify AIService catches unverified chemical dosage and returns localized safe fallback."""
    mock_ai_repo = AsyncMock()
    mock_ai_repo.get_farmer_profile.return_value = None
    mock_ai_repo.get_conversation_history.return_value = []
    mock_ai_repo.session = AsyncMock()

    service = AIService(mock_ai_repo)
    req = AIGenerateRequest(
        farmer_id=uuid4(),
        message="What medicine should I spray for Alternaria Leaf Spot on cotton?"
    )

    mock_rag_item = AsyncMock()
    mock_rag_item.document_title = "ICAR / PJTSAU Cotton Alternaria Leaf Spot Disease Management Guide"
    mock_rag_item.source = "ICAR-CICR / PJTSAU Package of Practices"
    mock_rag_item.crop = "Cotton"
    mock_rag_item.chunk_text = (
        "Title: Cotton Alternaria Leaf Spot Disease Management Guide. Crop: Cotton. "
        "Disease: Alternaria Leaf Spot. Control: Mancozeb 75% WP @ 2.5 to 3.0 g per litre of water."
    )

    # Gemini attempts to hallucinate Glyphosate at 2.5 g/L
    with patch("src.ai.service.generate_response", return_value="Spray Glyphosate at 2.5 g per litre of water for Alternaria on cotton."), \
         patch("src.rag.service.RAGService.search_knowledge", return_value=[mock_rag_item]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", return_value=None):

        resp = await service.generate_ai_response(req)
        # Glyphosate must be stripped, and safe fallback returned
        assert "Glyphosate" not in resp.response_text
        assert "AEO" in resp.response_text or "వ్యవసాయ" in resp.response_text or "consult" in resp.response_text.lower()


@pytest.mark.asyncio
async def test_unconfirmed_diagnosis_without_photo_blocks_chemical_dosage():
    """
    Scenario:
    1. Farmer reports cotton leaves yellowing with brown spots.
    2. Farmer follow-up states: 'అవును, మచ్చలు ఉన్నాయి కానీ ఫోటో ఇవ్వడం కుదరదు.'
    Expected:
    Must NOT prescribe Mancozeb 2.5-3 g/L or any pesticide dosage.
    Must state safe refusal requiring photo or AEO consultation:
    'ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను. దయచేసి పంట ఫోటోను పంపండి లేదా వ్యవసాయ అధికారిని సంప్రదించండి.'
    """
    mock_ai_repo = AsyncMock()
    mock_ai_repo.get_farmer_profile.return_value = None
    # Previous turn: farmer reported yellow leaves with brown spots
    mock_prev_record = AsyncMock()
    mock_prev_record.user_message = "పత్తి ఆకులు పసుపుగా మారి గోధుమ రంగు మచ్చలు ఉన్నాయి."
    mock_prev_record.ai_response = "ఆకులపై మచ్చలు ఎలా ఉన్నాయి? ఫోటో పంపగలరా?"
    mock_ai_repo.get_conversation_history.return_value = [mock_prev_record]
    mock_ai_repo.session = AsyncMock()

    service = AIService(mock_ai_repo)
    req = AIGenerateRequest(
        farmer_id=uuid4(),
        message="అవును, మచ్చలు ఉన్నాయి కానీ ఫోటో ఇవ్వడం కుదరదు."
    )

    mock_rag_item = AsyncMock()
    mock_rag_item.document_title = "ICAR / PJTSAU Cotton Alternaria Leaf Spot Disease Management Guide"
    mock_rag_item.source = "ICAR-CICR / PJTSAU Package of Practices"
    mock_rag_item.crop = "Cotton"
    mock_rag_item.chunk_text = (
        "Title: Cotton Alternaria Leaf Spot Disease Management Guide. Crop: Cotton. "
        "Disease: Alternaria Leaf Spot. Control: Mancozeb 75% WP @ 2.5 to 3.0 g per litre of water."
    )

    # Even if LLM attempts to output Mancozeb 2.5–3 g/L
    with patch("src.ai.service.generate_response", return_value="ఇది ఆల్టర్నేరియా కావచ్చు. మాంకోజెబ్ (Mancozeb) 2.5 నుండి 3.0 గ్రాములు లీటరు నీటికి పిచికారీ చేయండి."), \
         patch("src.rag.service.RAGService.search_knowledge", return_value=[mock_rag_item]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", return_value=None):

        resp = await service.generate_ai_response(req)
        # Chemical dosage must be BLOCKED
        assert "2.5" not in resp.response_text
        assert "3.0" not in resp.response_text
        assert "Mancozeb" not in resp.response_text
        # Must give the exact safe refusal
        assert "ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను" in resp.response_text
        assert "వ్యవసాయ అధికారిని సంప్రదించండి" in resp.response_text or "వ్యవసాయ" in resp.response_text


@pytest.mark.asyncio
async def test_safe_closing_replaces_farmer_choice_question():
    """Verify that 'మీరు ఏ మందు పిచికారీ చేయాలనుకుంటున్నారు?' is sanitized and replaced with safe closing."""
    mock_ai_repo = AsyncMock()
    mock_ai_repo.get_farmer_profile.return_value = None
    mock_ai_repo.get_conversation_history.return_value = []
    mock_ai_repo.session = AsyncMock()

    service = AIService(mock_ai_repo)
    req = AIGenerateRequest(
        farmer_id=uuid4(),
        message="మిరపలో తామర పురుగుల నివారణకు మందు ఏమిటి?"
    )

    mock_rag_item = AsyncMock()
    mock_rag_item.document_title = "ICAR / Dr. YSRHU Chilli Thrips Management"
    mock_rag_item.source = "ICAR-IIHR / Dr. YSRHU"
    mock_rag_item.crop = "Chilli"
    mock_rag_item.chunk_text = (
        "Title: Chilli Thrips. Crop: Chilli. Control: Fipronil 5% SC: 2.0 ml per litre of water."
    )

    with patch("src.ai.service.generate_response", return_value="తామర పురుగులకు ఫిప్రోనిల్ 2.0 ml/l వాడవచ్చు. మీరు ఏ మందు పిచికారీ చేయాలనుకుంటున్నారు?"), \
         patch("src.rag.service.RAGService.search_knowledge", return_value=[mock_rag_item]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", return_value=None):

        resp = await service.generate_ai_response(req)
        assert "మీరు ఏ మందు పిచికారీ చేయాలనుకుంటున్నారు" not in resp.response_text
        assert "స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి" in resp.response_text

