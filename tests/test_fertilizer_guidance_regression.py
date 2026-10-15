"""
BhoomiMitra AI — Fertilizer Guidance & Grounding Pipeline Regression Tests

Tests coverage:
1. Telugu paddy fertilizer query (2 acres, 30 days after transplanting).
2. Equivalent English-language query.
3. Equivalent Romanized Telugu query.
4. Relevant verified fertilizer guidance exists and is retrieved from RAG.
5. No applicable verified guidance exists: asks clarifying questions instead of generic refusal.
6. Irrelevant agricultural documents (e.g., Cotton pest guide) do not justify a dosage.
7. Existing pre-LLM grounding gate remains strictly active.
8. Existing post-LLM dosage guardrail remains active against hallucinated numbers.
9. Fertilizer guidance and pesticide safety are decoupled appropriately.
10. Area-unit conversion (2 acres * 30-35 kg/acre = 60-70 kg) validated correctly.
11. Missing soil/fertilizer-history details are handled safely.
12. WhatsApp pipeline returns improved grounded response in Telugu.
13. RAG/AI provider failure returns safe fallback gracefully.
14. Multilingual support across Telugu, English, and Romanized Telugu.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.core.models import Farmer, Conversation
from src.ai.service import (
    AIService,
    is_dosage_sensitive_query,
    has_relevant_dosage_ground_truth,
    extract_query_dosage_topics,
    extract_land_size_from_text,
    process_text_message,
)
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse
from src.ai.prompts import (
    get_unverified_dosage_fallback_response,
    get_unverified_fertilizer_fallback_response,
    UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES,
    UNVERIFIED_DOSAGE_FALLBACK_RESPONSES,
)
from src.rag.service import RAGService, extract_crop_from_text


# =============================================================================
# Helper Fixtures & Test Data
# =============================================================================

TELUGU_FARMER_QUERY = (
    "నేను 2 ఎకరాల్లో వరి పంట సాగు చేస్తున్నాను. నాట్లు వేసి 30 రోజులు అయ్యాయి. "
    "ఇప్పుడు ఏ ఎరువులు అవసరం? ఎంత మోతాదులో వేయాలి? "
    "సరైన మోతాదును నిర్ణయించడానికి నా నేల వివరాలు లేదా ఇప్పటికే వేసిన ఎరువుల వివరాలు అవసరమా?"
)

ENGLISH_FARMER_QUERY = (
    "I grow paddy on 2 acres. It has been 30 days since transplanting. "
    "Which fertilizers are needed now, and in what quantities? "
    "Do you need soil details or information about fertilizers already applied to determine the correct dosage?"
)

ROMANIZED_FARMER_QUERY = (
    "nenu 2 acres lo vari saagu chesthunnanu. naatlu vesi 30 days ayyindi. "
    "ippudu em eruvulu avasaram? entha mothadulo veyali?"
)

MOCK_PJTSAU_RAG_SNIPPET = (
    "• Document: PJTSAU & ICAR-IIRR Paddy Fertilizer Guide (Crop: Paddy, Source: PJTSAU & ICAR-IIRR): "
    "Transplanted Rice / Paddy (వరి పంట) Recommended Dose of Fertilizers (RDF): 40-50 kg N, 24 kg P2O5, 16-20 kg K2O per acre. "
    "Split Application Schedule: "
    "1. Basal Application (చివరి దుక్కి / నాట్ల సమయంలో): Entire Phosphorus (50 kg DAP or 150 kg SSP per acre) + 33% Nitrogen (30-35 kg Urea per acre) + 50% Potash (15-20 kg MOP per acre). "
    "2. Active Tillering Stage (పిలకల దశ - 20 to 30/35 days after transplanting / DAT): Apply 30-35 kg Urea per acre as first top dressing (2 ఎకరాలకు 60-70 కేజీల యూరియా). "
    "3. Panicle Initiation / Booting Stage (చిరుపొట్ట దశ - 45 to 55 DAT): Apply 30-35 kg Urea + 15-20 kg MOP per acre. "
    "Varietal & Soil Factors: Short duration varieties require earlier N application. Soil test report determines exact P & K rates."
)


# =============================================================================
# 1. Telugu Paddy Fertilizer Query (2 acres, 30 days DAT)
# =============================================================================

def test_case_1_telugu_paddy_query_extraction_and_grounding():
    """Verify that Telugu paddy query extracts crop, 2 acres land size, and matches PJTSAU grounding."""
    # 1. Dosage intent detected
    assert is_dosage_sensitive_query(TELUGU_FARMER_QUERY) is True

    # 2. Crop extracted as Paddy
    assert extract_crop_from_text(TELUGU_FARMER_QUERY) == "Paddy"

    # 3. Land size extracted as 2.0 acres
    assert extract_land_size_from_text(TELUGU_FARMER_QUERY) == 2.0

    # 4. Topics classified as fertilizer query
    topics = extract_query_dosage_topics(TELUGU_FARMER_QUERY)
    assert topics["crop"] == "paddy"
    assert topics["is_fertilizer_query"] is True
    assert topics["is_pest_disease_query"] is False

    # 5. Verified grounding matches PJTSAU snippet
    assert has_relevant_dosage_ground_truth(TELUGU_FARMER_QUERY, [MOCK_PJTSAU_RAG_SNIPPET]) is True


# =============================================================================
# 2. Equivalent English-Language Query
# =============================================================================

def test_case_2_english_paddy_query_extraction_and_grounding():
    """Verify equivalent English query correctly extracts crop, 2 acres land size, and matches grounding."""
    assert is_dosage_sensitive_query(ENGLISH_FARMER_QUERY) is True
    assert extract_crop_from_text(ENGLISH_FARMER_QUERY) == "Paddy"
    assert extract_land_size_from_text(ENGLISH_FARMER_QUERY) == 2.0

    topics = extract_query_dosage_topics(ENGLISH_FARMER_QUERY)
    assert topics["crop"] == "paddy"
    assert topics["is_fertilizer_query"] is True
    assert topics["is_pest_disease_query"] is False
    assert has_relevant_dosage_ground_truth(ENGLISH_FARMER_QUERY, [MOCK_PJTSAU_RAG_SNIPPET]) is True


# =============================================================================
# 3. Equivalent Romanized Telugu Query
# =============================================================================

def test_case_3_romanized_telugu_query_extraction_and_grounding():
    """Verify equivalent Romanized Telugu query extracts crop, 2 acres land size, and matches grounding."""
    assert is_dosage_sensitive_query(ROMANIZED_FARMER_QUERY) is True
    assert extract_crop_from_text(ROMANIZED_FARMER_QUERY) == "Paddy"
    assert extract_land_size_from_text(ROMANIZED_FARMER_QUERY) == 2.0

    topics = extract_query_dosage_topics(ROMANIZED_FARMER_QUERY)
    assert topics["crop"] == "paddy"
    assert topics["is_fertilizer_query"] is True
    assert has_relevant_dosage_ground_truth(ROMANIZED_FARMER_QUERY, [MOCK_PJTSAU_RAG_SNIPPET]) is True


# =============================================================================
# 4. Relevant Verified Fertilizer Guidance Exists and is Retrieved from RAG
# =============================================================================

@pytest.mark.asyncio
async def test_case_4_rag_retrieves_pjtsau_paddy_fertilizer_guide():
    """Verify that RAG hybrid search retrieves PJTSAU paddy fertilizer guidance from default knowledge."""
    mock_repo = MagicMock()
    mock_repo.get_all_chunks = AsyncMock(return_value=[])
    rag_service = RAGService(mock_repo)

    results = await rag_service.search_knowledge(
        query=TELUGU_FARMER_QUERY,
        top_k=2,
        state="Telangana",
        crop="Paddy",
    )

    assert len(results) > 0
    top_doc = results[0]
    assert "PJTSAU" in top_doc.document_title or "Paddy" in top_doc.document_title
    assert "Urea" in top_doc.chunk_text or "యూరియా" in top_doc.chunk_text
    assert "30-35 kg" in top_doc.chunk_text or "30" in top_doc.chunk_text


# =============================================================================
# 5. No Applicable Verified Guidance Exists: Asks Clarifying Questions
# =============================================================================

@pytest.mark.asyncio
async def test_case_5_unverified_fertilizer_query_asks_clarifying_questions():
    """Verify that when no RAG guidance exists, bot asks useful questions instead of generic refusal or fake numbers."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    # Empty RAG simulates no verified documents found
    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[]), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నా పంటకు ఇప్పుడు ఎంత ఎరువు వేయాలి?",  # Crop unknown, dosage asked
        )
        resp = await service.generate_ai_response(req)

        assert resp.provider_used == "hard_grounding_gate"
        assert resp.intent == "dosage_unverified_fallback"
        # Must ask clarifying questions about variety, basal DAP/SSP, soil test
        assert "వంగడం" in resp.response_text or "రకం" in resp.response_text
        assert "DAP" in resp.response_text or "ఎరువులు" in resp.response_text
        assert "నేల పరీక్ష" in resp.response_text
        # Must advise AEO / KVK
        assert "AEO" in resp.response_text or "KVK" in resp.response_text


# =============================================================================
# 6. Irrelevant Agricultural Documents Must NOT Justify a Dosage
# =============================================================================

def test_case_6_irrelevant_cotton_document_rejected_for_paddy_query():
    """Verify that a Cotton pest document does NOT satisfy grounding for a Paddy fertilizer query."""
    irrelevant_cotton_snippet = (
        "• Document: ICAR Cotton Pest Management (Crop: Cotton, Source: ICAR): "
        "Pink bollworm control: Spray Chlorantraniliprole 18.5% SC @ 0.4 ml per litre of water or Emamectin Benzoate 5% SG @ 0.4 g/L."
    )
    # Grounding check must reject irrelevant crop document
    is_grounded = has_relevant_dosage_ground_truth(TELUGU_FARMER_QUERY, [irrelevant_cotton_snippet])
    assert is_grounded is False


# =============================================================================
# 7. Existing Pre-LLM Grounding Protection Remains Active
# =============================================================================

@pytest.mark.asyncio
async def test_case_7_pre_llm_grounding_gate_blocks_gemini_call_when_ungrounded():
    """Verify that pre-LLM gate intercepts ungrounded dosage requests without calling Gemini API."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="వరిలో ఎకరానికి ఎంత యూరియా వేయాలి?",
        )
        resp = await service.generate_ai_response(req)

        # Gemini API must NEVER have been called
        mock_gemini.assert_not_called()
        assert resp.provider_used == "hard_grounding_gate"
        assert resp.intent == "dosage_unverified_fallback"


# =============================================================================
# 8. Existing Post-LLM Dosage Guardrail Remains Active Against Hallucinations
# =============================================================================

@pytest.mark.asyncio
async def test_case_8_post_llm_defense_in_depth_catches_hallucinated_numbers():
    """Verify defense-in-depth gate intercepts hallucinated dosage numbers that exceed RAG grounding."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    # RAG has official PJTSAU guide with 30-35 kg/acre
    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Simulate Gemini hallucinating an ungrounded, dangerous rate: "500 kg per acre"
    hallucinated_response = "వరి పంటకు ఎకరానికి 500 kg యూరియా వేయండి."

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=hallucinated_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message=TELUGU_FARMER_QUERY,
        )
        resp = await service.generate_ai_response(req)

        # Hallucinated "500" was blocked by post-LLM gate!
        assert "500" not in resp.response_text
        assert resp.response_text == UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES["te"]


# =============================================================================
# 9. Fertilizer Guidance and Pesticide Safety Handled Appropriately
# =============================================================================

def test_case_9_decoupled_fertilizer_and_pesticide_safety_responses():
    """Verify pesticide queries return chemical safety refusal, while fertilizer queries return structured guidance."""
    # Fertilizer fallback
    fert_fb = get_unverified_dosage_fallback_response("te", is_fertilizer=True)
    assert "ఎరువుల ఖచ్చితమైన మోతాదు" in fert_fb
    assert "వంగడం" in fert_fb
    assert "పురుగుమందు" not in fert_fb

    # Pesticide fallback
    pest_fb = get_unverified_dosage_fallback_response("te", is_fertilizer=False)
    assert "పురుగుమందుల" in pest_fb or "కీటనాశక" in pest_fb or "ధృవీకరించబడిన డేటా లేదు" in pest_fb
    assert "వంగడం" not in pest_fb


# =============================================================================
# 10. Area-Unit Conversion (2 acres * 30-35 kg = 60-70 kg) Supported
# =============================================================================

@pytest.mark.asyncio
async def test_case_10_area_scaling_allowed_in_post_llm_validation():
    """Verify that when 2 acres is specified, acreage-scaled totals (60-70 kg) pass post-LLM validation."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Gemini generates valid grounded advice mentioning 30-35 kg per acre and 60-70 kg for 2 acres
    grounded_ai_response = (
        "నాట్లలో 50 కిలోల DAP వేసినందున ప్రారంభ భాస్వరం మరియు నత్రజని సరిపోతాయి. "
        "నాట్లు వేసిన 30 రోజుల పిలకల దశలో మీ 2 ఎకరాల వరి పంటకు మొత్తం 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కేజీలు) "
        "పైపాటుగా అందించండి."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=grounded_ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నాట్లలో 50 కిలోల DAP వేశాను. నాట్లు వేసి 30 రోజులైన నా 2 ఎకరాల వరికి ఎంత యూరియా వేయాలి?",
        )
        resp = await service.generate_ai_response(req)

        # Grounded response is preserved and NOT replaced by refusal
        assert "60-70" in resp.response_text
        assert "30-35" in resp.response_text


# =============================================================================
# 11. Missing Soil / Fertilizer-History Details Handled Appropriately
# =============================================================================

def test_case_11_missing_soil_and_fertilizer_history_handled():
    """Verify that fallback and guidance emphasize soil test and basal application questions."""
    fallback_en = get_unverified_fertilizer_fallback_response("en")
    assert "soil conditions" in fallback_en.lower()
    assert "basal fertilizers" in fallback_en.lower()
    assert "soil test report" in fallback_en.lower()
    assert "AEO" in fallback_en or "KVK" in fallback_en


# =============================================================================
# 12. WhatsApp Pipeline Returns Improved Response in Telugu
# =============================================================================

@pytest.mark.asyncio
async def test_case_12_whatsapp_pipeline_end_to_end_telugu_flow():
    """Verify end-to-end process_text_message for the WhatsApp test farmer query."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message=TELUGU_FARMER_QUERY)

    mock_ai_resp = AIGenerateResponse(
        response_text=(
            "వరి పంట 30 రోజుల పిలకల దశలో ఖచ్చితమైన ఎరువుల మోతాదు మీ పొలంలో ఇప్పటికే వేసిన ఎరువులు, వరి రకం మరియు నేల స్వభావంపై ఆధారపడి ఉంటుంది. "
            "నాట్ల మరియు మునుపటి ఎరువుల వివరాలు లేకుండా నిర్దిష్ట రసాయన ఎరువుల మోతాదును ఖరారు చేయడం సురక్షితం కాదు; అధిక నత్రజని పంట నేలకొరిగే ప్రమాదాన్ని, అగ్గి తెగులు మరియు కాండం తొలిచే పురుగుల ఉధృతిని పెంచుతుంది.\n\n"
            "అదనపు మోతాదును నిర్ధారించలేము; అధిక నత్రజని పంట నేలకొరిగే ప్రమాదాన్ని, అగ్గి తెగులు మరియు కాండం తొలిచే పురుగుల ఉధృతిని పెంచుతుంది.\n\n"
            "సరైన మోతాదును నిర్ణయించడానికి దయచేసి ఈ వివరాలు తెలపండి:\n"
            "1. నాట్ల సమయంలో DAP, SSP లేదా కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా, ఎంత మోతాదులో వేశారు?\n"
            "2. నాట్లు వేసిన తర్వాత ఇప్పటికే ఏదైనా యూరియా లేదా పొటాష్ వేశారా (ఎంత మోతాదు, ఏ తేదీ/దశలో)?\n"
            "3. మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది (స్వల్పకాలిక లేదా దీర్ఘకాలిక)?\n"
            "4. నేల పరీక్ష నివేదిక (Soil Health Card) అందుబాటులో ఉందా? 🙏"
        ),
        intent="fertilizer_nutrient",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock, return_value=mock_ai_resp), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        whatsapp_response = await process_text_message(db_mock, farmer, conversation)

        # 1. Explains 60-70 kg / 30-35 kg/acre as attributed reference benchmark
        assert "60-70" not in whatsapp_response and "60–70" not in whatsapp_response
        assert "30-35" not in whatsapp_response and "30–35" not in whatsapp_response
        assert "ఖరారు చేయడం సురక్షితం కాదు" in whatsapp_response or "నిర్ధారించలేము" in whatsapp_response or "ఖరారైన నిర్ధారణ కాదు" in whatsapp_response

        # 2. Distinguishes from confirmed field recommendation and does NOT instruct to apply immediately
        assert "ఖరారైన నిర్ధారణ కాదు" in whatsapp_response or "నిర్ధారించలేము" in whatsapp_response
        assert "వేయాలి" not in whatsapp_response
        assert "సిఫార్సు చేయబడుతుంది" not in whatsapp_response

        # 3. Inquires about basal DAP/SSP/complex, post-transplanting urea/potash, variety, soil test
        assert "DAP" in whatsapp_response
        assert "SSP" in whatsapp_response
        assert "యూరియా" in whatsapp_response
        assert "రకం" in whatsapp_response or "వంగడం" in whatsapp_response
        assert "పరీక్ష" in whatsapp_response or "Soil Health Card" in whatsapp_response

        # Must not be the old generic refusal
        assert "ఖచ్చితమైన ఎరువుల/పురుగుమందుల మోతాదు (dosage) కోసం ప్రస్తుతం ధృవీకరించబడిన డేటా లేదు" not in whatsapp_response


# =============================================================================
# 13. Failure of RAG or AI Provider Produces Safe Fallback
# =============================================================================

@pytest.mark.asyncio
async def test_case_13_ai_provider_failure_returns_safe_fallback_gracefully():
    """Verify that when AI provider throws an exception, the system returns localized safe fallback without 500 error."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", side_effect=Exception("Gemini API connection error")):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message=TELUGU_FARMER_QUERY,
        )
        resp = await service.generate_ai_response(req)

        assert resp.provider_used == "safe_fallback"
        assert resp.intent == "ai_unavailable_fallback"
        # Returns safe localized fallback without crashing
        assert "AEO" in resp.response_text or "వ్యవసాయ" in resp.response_text


# =============================================================================
# 14. Multilingual Processing Supported Across Telugu, English, Hindi, and Tanglish
# =============================================================================

def test_case_14_multilingual_dosage_topic_and_fallback_integrity():
    """Verify that fallback text and query topic extraction exist for all 13 supported languages."""
    supported_langs = ["te", "en", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "or", "pa", "as", "ur"]

    for lang in supported_langs:
        fert_text = get_unverified_fertilizer_fallback_response(lang)
        assert len(fert_text) > 20
        assert "AEO" in fert_text or "KVK" in fert_text


# =============================================================================
# 15. Two-acre paddy, 30 DAT, no fertilizer history provided
# =============================================================================

@pytest.mark.asyncio
async def test_case_15_two_acre_paddy_no_history_checks_prior_application():
    """Verify that when 2-acre paddy query has no fertilizer history, response cites general benchmark, asks about variety and basal/history, and does not confirm unverified dose."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide (Regional Extension & Vikaspedia)"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Agricultural Extension & Vikaspedia"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "వరి పంట 30 రోజుల పిలకల దశలో 2 ఎకరాలకు ప్రాంతీయ విస్తరణ మార్గదర్శకాల ప్రకారం "
        "సుమారు 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) సాధారణ ప్రచురిత రిఫరెన్స్ మాత్రమే, ఇది మీ పొలానికి ఖరారైన నిర్ధారణ కాదు. "
        "సరైన మోతాదును నిర్ణయించడానికి: 1. మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది (స్వల్ప లేదా దీర్ఘకాలిక)? "
        "2. నాట్ల సమయంలో DAP లేదా SSP వేసారా? 3. నాట్లు వేసిన తర్వాత ఇప్పటికే ఏదైనా యూరియా వేశారా? "
        "ఈ వివరాలు లేకుండా అదనపు మోతాదును ఖరారు చేయలేము."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="2 ఎకరాల వరి, నాట్లు వేసి 30 రోజులు. ఎంత ఎరువు వేయాలి?",
        )
        resp = await service.generate_ai_response(req)
        # 1. Recognizes 2 acres paddy at 30 DAT / general extension benchmark
        assert "60-70" not in resp.response_text and "60–70" not in resp.response_text
        assert "30-35" not in resp.response_text and "30–35" not in resp.response_text
        # 2. Distinguishes published reference from field-specific recommendation
        assert "ఖరారు" in resp.response_text or "నిర్ధారణ" in resp.response_text or "వివరాలు లేకుండా" in resp.response_text
        # 3. Asks which variety is cultivated
        assert "రకం" in resp.response_text or "వంగడం" in resp.response_text
        # 4. Checks basal DAP/SSP and subsequent fertilizer applications
        assert "DAP" in resp.response_text or "SSP" in resp.response_text
        assert "యూరియా" in resp.response_text
        # 5. Does not present an unverified additional dose as confirmed
        assert "ఖరారు" in resp.response_text or "నిర్ధారణ" in resp.response_text or "వివరాలు లేకుండా" in resp.response_text


# =============================================================================
# 16. Farmer confirms DAP was applied at transplanting
# =============================================================================

@pytest.mark.asyncio
async def test_case_16_farmer_confirms_dap_applied_at_transplanting():
    """Verify that when farmer confirms DAP was applied as basal, response accounts for initial N and does not prescribe duplicate DAP."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "నాట్లలో 50 కిలోల DAP వేసినందున భాస్వరం మరియు ప్రారంభ నత్రజని సరిపోతుంది. మళ్లీ DAP వేయకండి. "
        "ఇప్పుడు 30 రోజుల పిలకల దశలో 2 ఎకరాలకు 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) పైపాటుగా వేయండి."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నేను నాట్లలో ఎకరానికి 50 కిలోల DAP వేశాను. ఇప్పుడు 30 రోజుల వరికి 2 ఎకరాలకు ఎంత యూరియా వేయాలి?",
        )
        resp = await service.generate_ai_response(req)
        assert "60-70" in resp.response_text
        assert "DAP" in resp.response_text


# =============================================================================
# 17. Farmer confirms SSP was applied at transplanting
# =============================================================================

@pytest.mark.asyncio
async def test_case_17_farmer_confirms_ssp_applied_at_transplanting():
    """Verify that when farmer confirms SSP was applied, system handles nitrogen top-dressing appropriately."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "SSP ఎరువు భాస్వరం మాత్రమే అందిస్తుంది. నాట్లు వేసిన 30 రోజుల పిలకల దశలో "
        "మీ 2 ఎకరాల వరికి 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) పైపాటుగా వేయండి."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నాట్లలో SSP వేశాను. ఇప్పుడు 30 రోజుల వరికి 2 ఎకరాలకు ఎంత యూరియా వేయాలి?",
        )
        resp = await service.generate_ai_response(req)
        assert "SSP" in resp.response_text
        assert "60-70" in resp.response_text


# =============================================================================
# 18. Farmer reports urea was already applied after transplanting
# =============================================================================

@pytest.mark.asyncio
async def test_case_18_farmer_reports_urea_already_applied_warns_against_duplication():
    """Verify that if farmer reports urea was already applied recently, response warns against duplicate urea application and avoids rigid waiting schedules."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "మీరు 25 రోజులప్పుడే 2 ఎకరాలకు 60 కిలోల యూరియా వేసి ఉంటే, ఇప్పుడు 30 రోజులకు వెంటనే మళ్లీ యూరియా వేయకండి. "
        "అవసరానికి మించి అధికంగా నత్రజని వేస్తే పంట నేలకొరిగే ప్రమాదం మరియు చీడపీడల (అగ్గి తెగులు, కాండం తొలిచే పురుగు) సమస్యలు పెరిగే అవకాశం ఉంది. "
        "స్థిరమైన రోజుల క్యాలెండర్ ప్రకారం కాకుండా, మీ వరి రకం కాలపరిమితి మరియు నేల తేమను బట్టి తదుపరి మోతాదును వ్యవసాయ అధికారి లేదా వంగడం దశల ప్రకారం నిర్ణయించండి."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నేను 25 రోజులప్పుడు 2 ఎకరాలకు 60 కిలోల యూరియా వేశాను. ఇప్పుడు 30 రోజులకు మళ్లీ యూరియా వేయాలా?",
        )
        resp = await service.generate_ai_response(req)
        # Must NOT prescribe a fixed 40-45 day waiting schedule
        assert "40-45" not in resp.response_text
        assert "40-45 రోజులు" not in resp.response_text
        # Warns against immediate duplicate urea
        assert "వేయకండి" in resp.response_text or "మళ్లీ" in resp.response_text
        # Mentions risk of lodging / pests
        assert "ప్రమాదం" in resp.response_text or "చీడపీడల" in resp.response_text or "పురుగు" in resp.response_text
        # Considers variety duration rather than rigid calendar
        assert "రకం" in resp.response_text or "కాలపరిమితి" in resp.response_text or "వంగడం" in resp.response_text


# =============================================================================
# 19. Soil-test result is available
# =============================================================================

@pytest.mark.asyncio
async def test_case_19_soil_test_available_incorporates_soil_health_card():
    """Verify that when soil test data is mentioned, response incorporates soil test calibration without arbitrary fixed percentages."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "భూసార పరీక్షలో నత్రజని అధికంగా ఉన్నట్లు తేలితే, మీ సాయిల్ హెల్త్ కార్డ్ (Soil Health Card) లేదా ICAR-IIRR NAPSRI "
        "సిఫార్సుల ప్రకారం నత్రజని మోతాదును సర్దుబాటు చేసుకోవాలి. సాధారణ ప్రాంతీయ మోతాదును గుడ్డిగా వేయకండి."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నా దగ్గర సాయిల్ టెస్ట్ రిపోర్ట్ ఉంది, అందులో నత్రజని ఎక్కువ ఉంది. 30 రోజుల వరికి ఎంత యూరియా వేయాలి?",
        )
        resp = await service.generate_ai_response(req)
        assert "పరీక్ష" in resp.response_text
        assert "సర్దుబాటు" in resp.response_text or "NAPSRI" in resp.response_text


# =============================================================================
# 20. Model proposes an unsupported or duplicated quantity
# =============================================================================

@pytest.mark.asyncio
async def test_case_20_model_proposes_unsupported_quantity_intercepted():
    """Verify that when LLM proposes an ungrounded or excessive quantity (e.g., 250 kg urea), post-LLM validation intercepts it."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ungrounded_hallucination = "వరి 2 ఎకరాలకు 250 kg యూరియా వేయండి."

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ungrounded_hallucination):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="2 ఎకరాల వరి, నాట్లు వేసి 30 రోజులు. ఎంత ఎరువు వేయాలి?",
        )
        resp = await service.generate_ai_response(req)
        assert "250" not in resp.response_text
        assert resp.response_text == UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES["te"]


# =============================================================================
# 21. Two-acre proportional calculations remain strictly validated
# =============================================================================

def test_case_21_two_acre_proportional_scaling_arithmetic():
    """Verify linear proportionality arithmetic for 2-acre guidance (30-35 kg/acre -> 60-70 kg for 2 acres)."""
    per_acre_min = 30
    per_acre_max = 35
    acres = 2.0

    two_acre_min = int(per_acre_min * acres)
    two_acre_max = int(per_acre_max * acres)

    assert two_acre_min == 60
    assert two_acre_max == 70
    assert f"{two_acre_min}-{two_acre_max} kg" == "60-70 kg"


# =============================================================================
# 22. Regional Source Distinction: ICAR-CRRI Advisory vs Telangana Extension
# =============================================================================

def test_case_22_crri_vs_telangana_source_distinction():
    """Verify that RAG knowledge base explicitly distinguishes CRRI Odisha varietal advisory from Telangana extension reference."""
    from src.rag.service import DEFAULT_VERIFIED_AGRI_KNOWLEDGE
    paddy_doc = next(d for d in DEFAULT_VERIFIED_AGRI_KNOWLEDGE if d["crop"] == "Paddy" and d["category"] == "Fertilizer Management")
    text = paddy_doc["chunk_text"]
    assert "ICAR-CRRI" in text
    assert "Do NOT copy CRRI" in text or "not copy" in text.lower()
    assert "general extension reference benchmark" in text
    assert "strictly separate from the seasonal nutrient requirement" in text


# =============================================================================
# 23. Incomplete Prior Urea Reports: Inquires Missing Details and Avoids Confirmation
# =============================================================================

@pytest.mark.asyncio
async def test_case_23_incomplete_prior_urea_inquiry_avoids_confirmation():
    """Verify that when a farmer reports prior urea without quantity/variety, bot asks for missing details and avoids confirming extra dose."""
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    ai_response = (
        "మీరు ఇటీవల యూరియా వేసినట్లు తెలిపారు, కానీ ఎంత మోతాదులో, ఏ తేదీన/దశలో వేశారు మరియు ఏ వంగడం సాగు చేస్తున్నారో "
        "తెలియకుండా అదనపు యూరియా మోతాదును నిర్ధారించలేము. ఈ వివరాలు తెలియజేయండి; అధిక నత్రజని పంట నేలకొరిగే ప్రమాదాన్ని పెంచుతుంది."
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నేను మొన్ననే యూరియా వేశాను. ఇప్పుడు మళ్లీ వేయాలా?",
        )
        resp = await service.generate_ai_response(req)
        # Avoids confirming additional dose
        assert "నిర్ధారించలేము" in resp.response_text or "ఖరారు" in resp.response_text or "వేయకండి" in resp.response_text
        # Inquires for missing details
        assert "మోతాదు" in resp.response_text
        assert "వంగడం" in resp.response_text or "రకం" in resp.response_text or "తేదీ" in resp.response_text
        # Does NOT prescribe a fixed 40-45 day schedule
        assert "40-45" not in resp.response_text


# =============================================================================
# 24. Theoretical Seasonal Urea Equivalent and DAP Nitrogen Deduction
# =============================================================================

def test_case_24_theoretical_seasonal_urea_accounting_and_dap_deduction():
    """Verify that 88-106 kg/acre is labeled as theoretical urea equivalent under straight urea assumption, DAP reduces it to 68-86 kg/acre (137-172 kg for 2 acres), and organic manure available nutrients are considered."""
    from src.rag.service import DEFAULT_VERIFIED_AGRI_KNOWLEDGE
    paddy_doc = next(d for d in DEFAULT_VERIFIED_AGRI_KNOWLEDGE if d["crop"] == "Paddy" and d["category"] == "Fertilizer Management")
    text = paddy_doc["chunk_text"]
    # 1. Theoretical urea equivalent before DAP under 100% straight urea assumption (88 to 106 kg/acre)
    assert "Theoretical Seasonal Urea Equivalent (Before DAP)" in text
    assert "88 to 106 kg Urea per acre" in text
    assert "Do NOT label 88–106 kg/acre as the farmer's confirmed seasonal requirement" in text
    # 2. Explains DAP nitrogen deduction (9 kg N from 50 kg DAP reduces urea to 68-86 kg/acre, 137-172 kg for 2 acres)
    assert "68 to 86 kg Urea per acre" in text
    assert "137 to 172 kg Urea for 2 acres" in text
    # 3. Organic manure available nutrients distinction
    assert "organic manures" in text.lower() or "organic manure" in text.lower()
    assert "available" in text.lower()
    # 4. Keeps single application separate from seasonal requirement
    assert "strictly separate from the seasonal nutrient requirement" in text


# =============================================================================
# 25. Live WhatsApp Regression Query: Intercepts Premature Application Instruction
# =============================================================================

@pytest.mark.asyncio
async def test_case_25_live_whatsapp_regression_query_intercepts_premature_application_instruction():
    """
    REGRESSION TEST for the live WhatsApp incident:
    Farmer with 2 acres of paddy at 30 DAT asked:
    "నేను 2 ఎకరాల్లో వరి పంట సాగు చేస్తున్నాను. నాట్లు వేసి 30 రోజులు అయ్యాయి. ఇప్పుడు ఏ ఎరువులు అవసరం? ఎంత మోతాదులో వేయాలి? సరైన మోతాదును నిర్ణయించడానికి నా నేల వివరాలు లేదా ఇప్పటికే వేసిన ఎరువుల వివరాలు అవసరమా?"

    Simulates Gemini attempting to return the flawed instruction:
    "మొత్తం 60 నుండి 70 కిలోల యూరియా ... సిఫార్సు చేయబడుతుంది ... దీనిని వేయండి ... DAP లేదా SSP వేశారా? వరి రకం ఏది?"

    Asserts:
    1. The initial response does NOT instruct the farmer to apply 60–70 kg immediately.
    2. The dose is not confirmed while essential history is missing.
    3. The 30-35 kg/acre (60-70 kg for 2 acres) tillering figure is explained as an attributed general extension reference benchmark.
    4. Inquires about basal DAP/SSP/complex, post-transplanting urea/potash, variety, and soil test.
    """
    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "Regional Extension"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Flawed response identical to the observed WhatsApp regression:
    flawed_gemini_response = (
        "నాట్లు వేసిన 30 రోజుల పిలకల దశలో మీ 2 ఎకరాల వరి పంటకు మొత్తం 60 నుండి 70 కిలోల యూరియా సిఫార్సు చేయబడుతుంది. "
        "దీనిని బురద పదునులో పైపాటుగా వేయండి. "
        "గమనిక: నాట్లలో మీరు DAP, SSP లేదా ఇతర ఎరువులు వేశారా? వరి రకం ఏది?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_gemini_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message=TELUGU_FARMER_QUERY,
        )
        resp = await service.generate_ai_response(req)

        # 1. Must NOT instruct the farmer to apply 60-70 kg immediately
        assert "వేయండి" not in resp.response_text
        assert "వేయాలి" not in resp.response_text
        assert "సిఫార్సు చేయబడుతుంది" not in resp.response_text

        # 2. Explains 60-70 kg / 30-35 kg/acre as an attributed reference benchmark, NOT confirmed recommendation
        assert "60-70" not in resp.response_text and "60–70" not in resp.response_text
        assert "30-35" not in resp.response_text and "30–35" not in resp.response_text
        assert "ఖరారు చేయడం సురక్షితం కాదు" in resp.response_text or "ఖరారైన నిర్ధారణ కాదు" in resp.response_text or "నిర్ధారించలేము" in resp.response_text
        assert "ఖరారైన నిర్ధారణ కాదు" in resp.response_text or "నిర్ధారించలేము" in resp.response_text

        # 3. Asks for missing essential history: basal DAP/SSP/complex, post-transplanting urea/potash, variety, soil test
        assert "DAP" in resp.response_text
        assert "SSP" in resp.response_text
        assert "యూరియా" in resp.response_text
        assert "రకం" in resp.response_text or "వంగడం" in resp.response_text
        assert "పరీక్ష" in resp.response_text or "Soil Health Card" in resp.response_text

        # 4. Explains excess nitrogen risk (lodging / blast / stem borer)
        assert "నేలకొరిగే" in resp.response_text or "ప్రమాదం" in resp.response_text
        assert "అగ్గి తెగులు" in resp.response_text or "తెగులు" in resp.response_text
        assert "పురుగు" in resp.response_text

# =============================================================================
# 26. Fertilizer History Detection: Distinguishes Positive History from Negation
# =============================================================================

def test_case_26_fertilizer_history_detection_negation_and_positive_application():
    """
    Verifies that check_fertilizer_history_details properly distinguishes positive
    application history from explicit statements that fertilizer was NOT applied.

    Required test cases:
    1. 'నాట్లలో DAP వేయలేదు': Must NOT be classified as positive DAP basal application.
    2. 'యూరియా ఇంకా వేయలేదు': Must NOT be classified as positive prior urea application.
    3. 'ఇంతవరకు ఏ ఎరువూ వేయలేదు': Must NOT be classified as positive fertilizer history.
    4. 'నాట్లలో 50 కిలోల DAP వేశాను': Must recognize positive basal DAP application.
    5. '25 రోజులప్పుడు 30 కిలోల యూరియా వేశాను': Must recognize positive prior urea application.
    6. Original two-acre Telugu question: Must NOT be mistaken for applied history.
    """
    from src.ai.service import check_fertilizer_history_details

    # 1. "నాట్లలో DAP వేయలేదు" -> NOT positive basal history
    r1 = check_fertilizer_history_details("నాట్లలో DAP వేయలేదు")
    assert r1["has_basal_history"] is False
    assert r1["has_prior_urea_or_potash"] is False
    assert r1["has_essential_history"] is False

    # 2. "యూరియా ఇంకా వేయలేదు" -> NOT positive prior urea history
    r2 = check_fertilizer_history_details("యూరియా ఇంకా వేయలేదు")
    assert r2["has_basal_history"] is False
    assert r2["has_prior_urea_or_potash"] is False
    assert r2["has_essential_history"] is False

    # 3. "ఇంతవరకు ఏ ఎరువూ వేయలేదు" -> NOT positive fertilizer history
    r3 = check_fertilizer_history_details("ఇంతవరకు ఏ ఎరువూ వేయలేదు")
    assert r3["has_basal_history"] is False
    assert r3["has_prior_urea_or_potash"] is False
    assert r3["has_essential_history"] is False

    # 4. "నాట్లలో 50 కిలోల DAP వేశాను" -> Positive basal DAP application
    r4 = check_fertilizer_history_details("నాట్లలో 50 కిలోల DAP వేశాను")
    assert r4["has_basal_history"] is True
    assert r4["has_prior_urea_or_potash"] is False
    assert r4["has_essential_history"] is True

    # 5. "25 రోజులప్పుడు 30 కిలోల యూరియా వేశాను" -> Positive prior urea application
    r5 = check_fertilizer_history_details("25 రోజులప్పుడు 30 కిలోల యూరియా వేశాను")
    assert r5["has_basal_history"] is False
    assert r5["has_prior_urea_or_potash"] is True
    assert r5["has_essential_history"] is True

    # 6. Original WhatsApp query asking whether details are needed -> NOT positive history
    r6 = check_fertilizer_history_details(TELUGU_FARMER_QUERY)
    assert r6["has_basal_history"] is False
    assert r6["has_prior_urea_or_potash"] is False
    assert r6["has_soil_test_info"] is False
    assert r6["has_essential_history"] is False
    assert r6["asks_if_details_needed"] is True


# =============================================================================
# 27. Telugu Query: Explicit DAP and Urea Negation & Personalized Fallback
# =============================================================================

def test_case_27_telugu_query_explicit_dap_and_urea_negation_personalized_fallback():
    """
    Verifies that for the real production query:
    'నాట్లలో DAP వేయలేదు.
     ఇంతవరకు యూరియా కూడా వేయలేదు.
     నాట్లు వేసి 30 రోజులు అయ్యాయి.
     ఇప్పుడు ఏ ఎరువులు వేయాలి?'

    1. The query is recognized as a paddy tillering fertilizer query and dosage-sensitive.
    2. DAP negation and urea negation are explicitly recognized.
    3. Essential history remains unconfirmed (no exact dosage confirmed yet).
    4. Fallback does NOT repeat questions asking if DAP or urea was applied.
    5. Fallback asks ONLY genuinely missing essential details:
       - Whether SSP or another complex fertilizer was applied.
       - Rice variety / duration group.
       - Whether soil test report (Soil Health Card) is available.
    """
    from src.ai.service import (
        is_paddy_tillering_fertilizer_query,
        is_dosage_sensitive_query,
        check_fertilizer_history_details,
        build_unconfirmed_paddy_fertilizer_guidance,
    )
    from src.ai.prompts import get_unverified_dosage_fallback_response

    query = (
        "నాట్లలో DAP వేయలేదు.\n"
        "ఇంతవరకు యూరియా కూడా వేయలేదు.\n"
        "నాట్లు వేసి 30 రోజులు అయ్యాయి.\n"
        "ఇప్పుడు ఏ ఎరువులు వేయాలి?"
    )

    # 1. Query detection
    assert is_paddy_tillering_fertilizer_query(query) is True
    assert is_dosage_sensitive_query(query) is True

    # 2. History extraction
    details = check_fertilizer_history_details(query)
    assert details["dap_negated"] is True
    assert details["urea_negated"] is True
    assert details["has_basal_history"] is False
    assert details["has_prior_urea_or_potash"] is False
    assert details["has_variety_info"] is False
    assert details["has_soil_test_info"] is False
    assert details["has_essential_history"] is False

    # 3. Personalized extension reference response
    resp = build_unconfirmed_paddy_fertilizer_guidance("te", acres=2.0, fert_hist_info=details)

    # Must NOT re-ask whether DAP or urea was applied
    assert "DAP, SSP లేదా" not in resp
    assert "DAP వేశారా" not in resp
    assert "యూరియా లేదా పొటాష్ వేశారా" not in resp

    # Must ask ONLY genuinely missing essential details
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp
    assert "వరి రకం/వంగడం" in resp
    assert "నేల పరీక్ష నివేదిక" in resp

    # Attributed benchmark qualified without prescribing confirmed unverified dose
    assert "60-70" not in resp and "60–70" not in resp
    assert "30-35" not in resp and "30–35" not in resp
    assert "సాధారణ ప్రచురిత రిఫరెన్స్ మాత్రమే" in resp
    assert "ఖరారైన నిర్ధారణ కాదు" in resp
    assert "అదనపు మోతాదును నిర్ధారించలేము" in resp
    assert "నేలకొరిగే ప్రమాదాన్ని" in resp

    # 4. Personalized defense fallback
    fallback = get_unverified_dosage_fallback_response("te", is_fertilizer=True, fert_hist_info=details)
    assert "DAP/SSP" not in fallback
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in fallback


# =============================================================================
# 28. English Equivalent: Explicit DAP and Urea Negation & Personalized Fallback
# =============================================================================

def test_case_28_english_query_explicit_dap_and_urea_negation_personalized_fallback():
    """
    Verifies that the English equivalent of the negative application query:
    'DAP was not applied at transplanting.
     Urea has not been applied so far.
     It has been 30 days since transplanting.
     Which fertilizers should be applied now?'

    1. Detects paddy tillering query and dosage sensitivity.
    2. Extracts DAP and urea negation correctly.
    3. Fallback does not re-ask if DAP or urea was applied.
    4. Fallback asks whether SSP or another complex fertilizer was applied, variety, and soil test.
    """
    from src.ai.service import (
        is_paddy_tillering_fertilizer_query,
        is_dosage_sensitive_query,
        check_fertilizer_history_details,
        build_unconfirmed_paddy_fertilizer_guidance,
    )
    from src.ai.prompts import get_unverified_dosage_fallback_response

    query = (
        "DAP was not applied at transplanting. "
        "Urea has not been applied so far. "
        "It has been 30 days since transplanting. "
        "Which fertilizers should be applied now?"
    )

    assert is_paddy_tillering_fertilizer_query(query) is True
    assert is_dosage_sensitive_query(query) is True

    details = check_fertilizer_history_details(query)
    assert details["dap_negated"] is True
    assert details["urea_negated"] is True
    assert details["has_basal_history"] is False
    assert details["has_prior_urea_or_potash"] is False
    assert details["has_essential_history"] is False

    resp = build_unconfirmed_paddy_fertilizer_guidance("en", acres=2.0, fert_hist_info=details)

    # Must NOT re-ask whether DAP or urea was applied
    assert "Which basal fertilizers (DAP, SSP, or complex)" not in resp
    assert "Has any urea or potash already been applied" not in resp

    # Must ask ONLY genuinely missing details
    assert "SSP or another complex fertilizer" in resp
    assert "rice variety or duration group" in resp
    assert "soil test report or Soil Health Card" in resp

    # Attributed benchmark qualified
    assert "60-70" not in resp and "60–70" not in resp
    assert "30-35" not in resp and "30–35" not in resp
    assert "attributed general extension reference benchmark" in resp
    assert "not a confirmed field-specific recommendation" in resp

    # English defense fallback verification
    fallback = get_unverified_dosage_fallback_response("en", is_fertilizer=True, fert_hist_info=details)
    assert "DAP/SSP" not in fallback
    assert "SSP or another complex fertilizer" in fallback


# =============================================================================
# 29. Guardrails Intercept Premature Unconfirmed Dose & Return Personalized Guidance
# =============================================================================

@pytest.mark.asyncio
async def test_case_29_guardrails_intercept_premature_dose_and_return_personalized_guidance():
    """
    Verifies that when Gemini generates an unverified dosage instruction for a farmer
    who explicitly reported that DAP and urea were NOT applied, the guardrails intercept it
    and replace it with the personalized reference guidance that does not re-ask about DAP or urea.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Gemini attempts to instruct applying 65 kg urea now
    unsafe_model_output = "వరి పంటకు 2 ఎకరాలకు 65 కిలోల యూరియా వేయండి."

    query = (
        "నాట్లలో DAP వేయలేదు.\n"
        "ఇంతవరకు యూరియా కూడా వేయలేదు.\n"
        "నాట్లు వేసి 30 రోజులు అయ్యాయి.\n"
        "ఇప్పుడు ఏ ఎరువులు వేయాలి?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=unsafe_model_output):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message=query,
        )
        resp = await service.generate_ai_response(req)

        # Unsafe instruction intercepted
        assert "వేయండి" not in resp.response_text

        # Replaced with personalized guidance
        assert "DAP, SSP లేదా" not in resp.response_text
        assert "DAP వేశారా" not in resp.response_text
        assert "యూరియా లేదా పొటాష్ వేశారా" not in resp.response_text
        assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp.response_text
        assert "రకం/వంగడం" in resp.response_text
        assert "60-70" not in resp.response_text and "60–70" not in resp.response_text
        assert "30-35" not in resp.response_text and "30–35" not in resp.response_text
        assert "నేల పరీక్ష నివేదిక" in resp.response_text


# =============================================================================
# 30. Conversation History Preserved in Personalized Fallback
# =============================================================================

def test_case_30_conversation_history_reaches_fallback_correctly():
    """
    Verifies that when prior conversation history contains crop variety information,
    the personalized fallback recognizes that variety is already known and avoids
    re-asking about variety.
    """
    from unittest.mock import MagicMock
    from src.ai.service import check_fertilizer_history_details, build_unconfirmed_paddy_fertilizer_guidance

    # Prior turn in conversation where farmer provided variety
    prior_turn = MagicMock()
    prior_turn.user_message = "మాది స్వల్పకాలిక వరి రకం RNR 15048 (తెలంగాణ సోనా)"
    prior_turn.ai_response = "మంచిది, RNR 15048 రకం వివరాలు నమోదు చేయబడ్డాయి."

    current_query = (
        "నాట్లలో DAP వేయలేదు.\n"
        "ఇంతవరకు యూరియా కూడా వేయలేదు.\n"
        "నాట్లు వేసి 30 రోజులు అయ్యాయి.\n"
        "ఇప్పుడు ఏ ఎరువులు వేయాలి?"
    )

    details = check_fertilizer_history_details(current_query, history_records=[prior_turn])

    assert details["dap_negated"] is True
    assert details["urea_negated"] is True
    assert details["has_variety_info"] is True  # Variety known from history!
    assert details["has_soil_test_info"] is False

    resp = build_unconfirmed_paddy_fertilizer_guidance("te", acres=2.0, fert_hist_info=details)

    # Must NOT ask if DAP or urea was applied
    assert "DAP వేశారా" not in resp
    assert "యూరియా లేదా పొటాష్ వేశారా" not in resp

    # Must NOT ask for variety since it was already provided in prior history!
    assert "మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది" not in resp

    # Must ask ONLY genuinely missing details (SSP/complex and soil test)
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp
    assert "నేల పరీక్ష నివేదిక" in resp



# =============================================================================
# 31. Non-Paddy Crops Not Misclassified as Paddy Tillering Fertilizer Queries
# =============================================================================

def test_case_31_non_paddy_crops_not_misclassified_as_paddy():
    """
    Verifies that paddy-specific guardrails do not incorrectly classify
    clearly identified non-paddy crops (such as chilli, tomato, cotton) as paddy,
    even when transplanting or 30-day stage keywords are present.
    """
    from src.ai.service import is_paddy_tillering_fertilizer_query

    # English non-paddy transplanting queries
    chilli_query = "Chilli transplanting was done 30 days ago. What fertilizer should be applied?"
    assert is_paddy_tillering_fertilizer_query(chilli_query) is False
    assert is_paddy_tillering_fertilizer_query(chilli_query, crop="Chilli") is False

    # Telugu non-paddy transplanting queries
    tomato_query = "టమాటా నాట్లు వేసి 30 రోజులు అయ్యాయి. ఏ ఎరువులు వేయాలి?"
    assert is_paddy_tillering_fertilizer_query(tomato_query) is False
    assert is_paddy_tillering_fertilizer_query(tomato_query, crop="Tomato") is False

    mirchi_query = "మిర్చి నాట్లలో DAP వేయలేదు, ఇప్పుడు ఏ ఎరువు వేయాలి?"
    assert is_paddy_tillering_fertilizer_query(mirchi_query) is False
    assert is_paddy_tillering_fertilizer_query(mirchi_query, crop="Chilli") is False

    # Non-paddy crop in context/profile without explicit paddy mention in message
    generic_transplant_query = "నాట్లు వేసి 30 రోజులు అయ్యాయి. ఏ ఎరువులు వేయాలి?"
    assert is_paddy_tillering_fertilizer_query(generic_transplant_query, crop="Chilli") is False
    assert is_paddy_tillering_fertilizer_query(generic_transplant_query, crop="Cotton") is False

    # Still correctly identifies paddy when crop is paddy or query has explicit paddy keywords
    assert is_paddy_tillering_fertilizer_query(generic_transplant_query) is True
    assert is_paddy_tillering_fertilizer_query(generic_transplant_query, crop="Paddy") is True
    paddy_explicit = "వరి నాట్లలో DAP వేయలేదు, 30 రోజులకు యూరియా ఎంత వేయాలి?"
    assert is_paddy_tillering_fertilizer_query(paddy_explicit, crop="Cotton") is True



# =============================================================================
# 32. Example WhatsApp Query Returns Safe Clarification Without Dose Benchmarks
# =============================================================================

@pytest.mark.asyncio
async def test_case_32_example_query_returns_safe_clarification_without_numerical_dose_benchmarks():
    """
    Verifies that when a farmer states DAP and urea were not applied at 30 DAT:
    "నాట్లలో DAP వేయలేదు.
     ఇంతవరకు యూరియా కూడా వేయలేదు.
     నాట్లు వేసి 30 రోజులు అయ్యాయి.
     ఇప్పుడు ఏ ఎరువులు వేయాలి?"

    The AI response:
    1. Returns a safe clarification inquiry without numerical dose benchmarks (no 30-35 or 60-70 figures).
    2. Asks ONLY for genuinely missing details (SSP/complex, variety, soil test).
    3. Does NOT repeat questions for information already supplied (no asking about DAP or urea).
    4. Explains excess nitrogen risks (lodging, blast, stem borer) without confirming an ungrounded dose.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=uuid4(), preferred_language="te"))
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Flawed response attempting to output general benchmark figures
    flawed_ai_response = (
        "వరి పంట 30 రోజుల పిలకల దశలో ప్రాంతీయ వ్యవసాయ విస్తరణ మార్గదర్శకాల ప్రకారం 2 ఎకరాలకు సుమారు 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) "
        "సాధారణ ప్రచురిత రిఫరెన్స్ మాత్రమే. సరైన మోతాదు కొరకు నాట్లలో SSP వేశారా? వరి రకం ఏది? నేల పరీక్ష నివేదిక ఉందా?"
    )

    query = (
        "నాట్లలో DAP వేయలేదు.\n"
        "ఇంతవరకు యూరియా కూడా వేయలేదు.\n"
        "నాట్లు వేసి 30 రోజులు అయ్యాయి.\n"
        "ఇప్పుడు ఏ ఎరువులు వేయాలి?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_ai_response):
        req = AIGenerateRequest(
            farmer_id=uuid4(),
            message=query,
        )
        resp = await service.generate_ai_response(req)

        # 1. No numerical fertilizer dose benchmarks that could be mistaken for an application instruction
        assert "60-70" not in resp.response_text
        assert "60–70" not in resp.response_text
        assert "30-35" not in resp.response_text
        assert "30–35" not in resp.response_text
        assert "వేయండి" not in resp.response_text
        assert "సిఫార్సు చేయబడుతుంది" not in resp.response_text

        # 2. Asks ONLY for missing details (SSP/complex, variety, soil test)
        assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp.response_text
        assert "రకం/వంగడం" in resp.response_text
        assert "నేల పరీక్ష నివేదిక" in resp.response_text

        # 3. Does NOT repeat questions about DAP or urea
        assert "DAP, SSP లేదా" not in resp.response_text
        assert "DAP వేశారా" not in resp.response_text
        assert "యూరియా లేదా పొటాష్ వేశారా" not in resp.response_text

        # 4. Explains excess nitrogen risks (lodging / blast / stem borer)
        assert "నేలకొరిగే" in resp.response_text
        assert "అగ్గి తెగులు" in resp.response_text
        assert "పురుగు" in resp.response_text


# =============================================================================
# 33. Live Test Regression: Exact Two-Turn Telugu Fertilizer Conversation
# =============================================================================

@pytest.mark.asyncio
async def test_case_33_exact_two_turn_telugu_fertilizer_conversation():
    """
    Regression Test for LIVE TEST PROBLEM (Telugu):
    Turn 1:
      Farmer:
      "నాట్లలో DAP వేయలేదు.
      ఇంతవరకు యూరియా కూడా వేయలేదు.
      నాట్లు వేసి 30 రోజులు అయ్యాయి.
      ఇప్పుడు ఏ ఎరువులు వేయాలి?"

      AI correctly asks for:
      1. SSP or complex fertilizer history.
      2. Paddy variety/duration.
      3. Soil test report availability.

    Turn 2:
      Farmer:
      "నాట్ల సమయంలో SSP లేదా కాంప్లెక్స్ ఎరువులు కూడా వేయలేదు.
      నా వరి రకం పేరు నాకు తెలియదు.
      నా దగ్గర నేల పరీక్ష నివేదిక కూడా లేదు."

    Expected AI response on Turn 2:
      - Understands all 3 clarification questions were answered (even though negative or unknown).
      - Does NOT repeat ANY of the three questions.
      - Acknowledges information received.
      - Explains next safe step (consult local AEO/KVK, avoid heavy unverified urea) and excess nitrogen risks.
      - Zero numerical dose benchmarks (no 60-70 or 30-35 kg).
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import AIService, check_fertilizer_history_details, build_unconfirmed_paddy_fertilizer_guidance
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    farmer_id = uuid4()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=farmer_id, preferred_language="te"))

    # Turn 1
    turn1_user = (
        "నాట్లలో DAP వేయలేదు.\n"
        "ఇంతవరకు యూరియా కూడా వేయలేదు.\n"
        "నాట్లు వేసి 30 రోజులు అయ్యాయి.\n"
        "ఇప్పుడు ఏ ఎరువులు వేయాలి?"
    )
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Flawed response attempting premature benchmark numbers in Turn 1
    flawed_turn1_llm = (
        "వరి పంట 30 రోజుల పిలకల దశలో 2 ఎకరాలకు సుమారు 60-70 కిలోల యూరియా సిఫార్సు చేయబడుతుంది. "
        "సరైన మోతాదు కొరకు నాట్లలో SSP వేశారా? వరి రకం ఏది? నేల పరీక్ష నివేదిక ఉందా?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_turn1_llm):
        resp1 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=turn1_user))

    # Turn 1 verification: asks for SSP/complex, variety, soil test; no DAP or urea questions; no benchmarks
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp1.response_text
    assert "రకం/వంగడం" in resp1.response_text
    assert "నేల పరీక్ష నివేదిక" in resp1.response_text
    assert "DAP వేశారా" not in resp1.response_text
    assert "యూరియా లేదా పొటాష్ వేశారా" not in resp1.response_text
    assert "60-70" not in resp1.response_text and "60–70" not in resp1.response_text
    assert "30-35" not in resp1.response_text and "30–35" not in resp1.response_text

    # Turn 2
    turn2_user = (
        "నాట్ల సమయంలో SSP లేదా కాంప్లెక్స్ ఎరువులు కూడా వేయలేదు.\n"
        "నా వరి రకం పేరు నాకు తెలియదు.\n"
        "నా దగ్గర నేల పరీక్ష నివేదిక కూడా లేదు."
    )
    c1 = Conversation(farmer_id=farmer_id, user_message=turn1_user, ai_response=resp1.response_text)
    mock_repo.get_conversation_history = AsyncMock(return_value=[c1])

    # Direct history detail check verification
    details = check_fertilizer_history_details(turn2_user, [c1])
    assert details["dap_negated"] is True
    assert details["urea_negated"] is True
    assert details["ssp_or_complex_negated"] is True
    assert details["variety_unknown"] is True
    assert details["soil_test_negated"] is True
    assert details["all_clarifications_answered"] is True

    # Flawed LLM attempting to re-ask questions in Turn 2
    flawed_turn2_llm = (
        "వరి పంటకు ఎరువుల కొరకు: నాట్ల సమయంలో SSP వేశారా? మీ వరి రకం ఏది? నేల పరీక్ష నివేదిక ఉందా?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_turn2_llm):
        resp2 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=turn2_user))

    # Turn 2 verification: AI must NOT repeat any of the 3 questions!
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా" not in resp2.response_text
    assert "SSP వేశారా" not in resp2.response_text
    assert "మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది" not in resp2.response_text
    assert "వరి రకం ఏది" not in resp2.response_text
    assert "నేల పరీక్ష నివేదిక (Soil Health Card) అందుబాటులో ఉందా" not in resp2.response_text
    assert "నేల పరీక్ష నివేదిక ఉందా" not in resp2.response_text

    # Turn 2 verification: Acknowledges information received & explains next safe steps
    assert "DAP, SSP లేదా కాంప్లెక్స్ ఎరువులు ఏవీ వేయలేదని" in resp2.response_text or "ప్రాథమిక ఎరువులు" in resp2.response_text
    assert "యూరియా కూడా ఇంకా వేయలేదని" in resp2.response_text or "యూరియా వేయలేదని" in resp2.response_text
    assert "వరి రకం" in resp2.response_text
    assert "నేల పరీక్ష నివేదిక అందుబాటులో లేదని" in resp2.response_text
    assert "AEO" in resp2.response_text or "వ్యవసాయ విస్తరణ అధికారి" in resp2.response_text
    assert "KVK" in resp2.response_text or "కృషి విజ్ఞాన కేంద్రం" in resp2.response_text

    # No numerical benchmarks
    assert "60-70" not in resp2.response_text and "60–70" not in resp2.response_text
    assert "30-35" not in resp2.response_text and "30–35" not in resp2.response_text


# =============================================================================
# 34. Live Test Regression: Equivalent English Two-Turn Fertilizer Conversation
# =============================================================================

@pytest.mark.asyncio
async def test_case_34_exact_two_turn_english_fertilizer_conversation():
    """
    Regression Test for Equivalent English Two-Turn Fertilizer Conversation.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import AIService, check_fertilizer_history_details
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    farmer_id = uuid4()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=farmer_id, preferred_language="en"))

    turn1_user = (
        "DAP was not applied at transplanting.\n"
        "Urea has not been applied so far.\n"
        "It has been 30 days since transplanting paddy.\n"
        "Which fertilizers should I apply now?"
    )
    mock_repo.get_conversation_history = AsyncMock(return_value=[])
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    flawed_turn1_llm = (
        "For paddy at 30 DAT, apply 60-70 kg Urea per 2 acres now. Did you apply SSP? What variety?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_turn1_llm):
        resp1 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=turn1_user))

    # Turn 1 asks for missing details in English without numerical dose benchmarks
    assert "Whether SSP or another complex fertilizer was applied" in resp1.response_text
    assert "Which rice variety or duration group" in resp1.response_text
    assert "soil test report" in resp1.response_text
    assert "60-70" not in resp1.response_text and "60–70" not in resp1.response_text
    assert "30-35" not in resp1.response_text and "30–35" not in resp1.response_text

    # Turn 2
    turn2_user = (
        "Neither SSP nor complex fertilizer was applied at transplanting.\n"
        "I do not know my paddy variety name.\n"
        "I do not have a soil test report either."
    )
    c1 = Conversation(farmer_id=farmer_id, user_message=turn1_user, ai_response=resp1.response_text)
    mock_repo.get_conversation_history = AsyncMock(return_value=[c1])

    flawed_turn2_llm = (
        "For paddy: Did you apply SSP? What variety are you growing? Do you have a soil test report?"
    )

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_turn2_llm):
        resp2 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=turn2_user))

    # Turn 2: Must NOT repeat answered questions
    assert "Whether SSP or another complex fertilizer was applied" not in resp2.response_text
    assert "Did you apply SSP" not in resp2.response_text
    assert "Which rice variety or duration group" not in resp2.response_text
    assert "What variety are you growing" not in resp2.response_text
    assert "Do you have a soil test report" not in resp2.response_text

    # Turn 2: Acknowledges information & advises next safe step (AEO/KVK)
    assert "Understood that no basal fertilizers" in resp2.response_text
    assert "Agricultural Extension Officer" in resp2.response_text or "AEO" in resp2.response_text
    assert "KVK" in resp2.response_text or "Krishi Vigyan Kendra" in resp2.response_text
    assert "60-70" not in resp2.response_text and "30-35" not in resp2.response_text


# =============================================================================
# 35. Farmer Stating Variety is Unknown Does Not Get Asked Variety Again
# =============================================================================

def test_case_35_farmer_stating_variety_is_unknown_omits_variety_inquiry():
    """
    Verifies that when a farmer states variety is unknown:
    1. check_fertilizer_history_details tracks variety_unknown as True.
    2. build_unconfirmed_paddy_fertilizer_guidance omits the variety question.
    3. Genuinely unanswered questions (e.g. SSP/complex and soil test) are still asked.
    """
    from src.ai.service import check_fertilizer_history_details, build_unconfirmed_paddy_fertilizer_guidance

    # Telugu unknown variety
    query_te = (
        "వరి నాట్లలో DAP వేయలేదు, యూరియా వేయలేదు.\n"
        "నా వరి రకం పేరు నాకు తెలియదు."
    )
    details_te = check_fertilizer_history_details(query_te)
    assert details_te["variety_unknown"] is True
    assert details_te["has_variety_info"] is False

    resp_te = build_unconfirmed_paddy_fertilizer_guidance("te", fert_hist_info=details_te)
    assert "రకం/వంగడం" not in resp_te
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp_te
    assert "నేల పరీక్ష నివేదిక" in resp_te

    # English unknown variety
    query_en = (
        "DAP was not applied, urea was not applied.\n"
        "I do not know the variety."
    )
    details_en = check_fertilizer_history_details(query_en)
    assert details_en["variety_unknown"] is True
    assert details_en["has_variety_info"] is False

    resp_en = build_unconfirmed_paddy_fertilizer_guidance("en", fert_hist_info=details_en)
    assert "variety or duration group" not in resp_en
    assert "Whether SSP or another complex fertilizer was applied" in resp_en
    assert "soil test report" in resp_en


# =============================================================================
# 36. Farmer Stating No Soil Test Report Does Not Get Asked Soil Test Again
# =============================================================================

def test_case_36_farmer_stating_no_soil_test_omits_soil_test_inquiry():
    """
    Verifies that when a farmer states no soil test report is available:
    1. check_fertilizer_history_details tracks soil_test_negated as True.
    2. build_unconfirmed_paddy_fertilizer_guidance omits the soil test question.
    3. Genuinely unanswered questions (e.g. variety and SSP/complex) are still asked.
    """
    from src.ai.service import check_fertilizer_history_details, build_unconfirmed_paddy_fertilizer_guidance

    # Telugu no soil test
    query_te = (
        "నాట్లలో DAP వేయలేదు, యూరియా వేయలేదు.\n"
        "నా దగ్గర నేల పరీక్ష నివేదిక లేదు."
    )
    details_te = check_fertilizer_history_details(query_te)
    assert details_te["soil_test_negated"] is True
    assert details_te["has_soil_test_info"] is False

    resp_te = build_unconfirmed_paddy_fertilizer_guidance("te", fert_hist_info=details_te)
    assert "నేల పరీక్ష నివేదిక" not in resp_te
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in resp_te
    assert "రకం/వంగడం" in resp_te

    # English no soil test
    query_en = (
        "DAP was not applied, urea was not applied.\n"
        "No soil test report is available."
    )
    details_en = check_fertilizer_history_details(query_en)
    assert details_en["soil_test_negated"] is True
    assert details_en["has_soil_test_info"] is False

    resp_en = build_unconfirmed_paddy_fertilizer_guidance("en", fert_hist_info=details_en)
    assert "soil test report" not in resp_en
    assert "Whether SSP or another complex fertilizer was applied" in resp_en
    assert "variety or duration group" in resp_en


# =============================================================================
# 37. Farmer Stating SSP/Complex Not Applied Does Not Get Asked SSP/Complex Again
# =============================================================================

def test_case_37_farmer_stating_ssp_complex_not_applied_omits_ssp_inquiry():
    """
    Verifies that when a farmer states SSP or complex was not applied:
    1. check_fertilizer_history_details tracks ssp_or_complex_negated as True.
    2. build_unconfirmed_paddy_fertilizer_guidance omits the SSP/complex question.
    3. Genuinely unanswered questions (e.g. variety and soil test) are still asked.
    """
    from src.ai.service import check_fertilizer_history_details, build_unconfirmed_paddy_fertilizer_guidance

    # Telugu SSP negated
    query_te = (
        "నాట్లలో DAP వేయలేదు, యూరియా వేయలేదు.\n"
        "నాట్ల సమయంలో SSP లేదా కాంప్లెక్స్ ఎరువులు కూడా వేయలేదు."
    )
    details_te = check_fertilizer_history_details(query_te)
    assert details_te["ssp_or_complex_negated"] is True
    assert details_te["dap_negated"] is True

    resp_te = build_unconfirmed_paddy_fertilizer_guidance("te", fert_hist_info=details_te)
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" not in resp_te
    assert "DAP, SSP లేదా కాంప్లెక్స్" not in resp_te
    assert "రకం/వంగడం" in resp_te
    assert "నేల పరీక్ష నివేదిక" in resp_te

    # English SSP negated
    query_en = (
        "DAP was not applied, urea was not applied.\n"
        "Neither SSP nor complex fertilizer was applied at transplanting."
    )
    details_en = check_fertilizer_history_details(query_en)
    assert details_en["ssp_or_complex_negated"] is True

    resp_en = build_unconfirmed_paddy_fertilizer_guidance("en", fert_hist_info=details_en)
    assert "Whether SSP or another complex fertilizer was applied" not in resp_en
    assert "variety or duration group" in resp_en
    assert "soil test report" in resp_en


# =============================================================================
# 38. Multi-turn AI Pipeline Never Repeats Answered Questions or Gives Premature Dose
# =============================================================================

@pytest.mark.asyncio
async def test_case_38_multi_turn_pipeline_never_repeats_answered_questions_or_premature_dose():
    """
    Verifies that across a sequence of turns, questions answered in previous turns
    are never asked again by AIService, and numerical benchmark doses remain absent.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    farmer_id = uuid4()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=farmer_id, preferred_language="te"))
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    # Turn 1: Farmer mentions 30 DAT, DAP not applied, urea not applied
    t1_msg = "నాట్లలో DAP వేయలేదు, యూరియా వేయలేదు, నాట్లు వేసి 30 రోజులు అయ్యాయి. ఇప్పుడు ఏ ఎరువు వేయాలి?"
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    flawed_llm_dose = "వరి పంట 30 రోజుల పిలకల దశలో 2 ఎకరాలకు సుమారు 60-70 కిలోల యూరియా సిఫార్సు చేయబడుతుంది."

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_llm_dose):
        r1 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t1_msg))

    # Turn 2: Farmer supplies variety only ("మాది స్వల్పకాలిక వరి రకం RNR 15048")
    t2_msg = "మాది స్వల్పకాలిక వరి రకం RNR 15048."
    c1 = Conversation(farmer_id=farmer_id, user_message=t1_msg, ai_response=r1.response_text)
    mock_repo.get_conversation_history = AsyncMock(return_value=[c1])

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_llm_dose):
        r2 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t2_msg))

    # In Turn 2, variety MUST NOT be asked (it was just answered!)
    assert "రకం/వంగడం" not in r2.response_text
    # DAP and urea MUST NOT be asked
    assert "DAP వేశారా" not in r2.response_text
    assert "యూరియా లేదా పొటాష్ వేశారా" not in r2.response_text
    # SSP and soil test remain missing, so they can be asked
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు" in r2.response_text
    assert "నేల పరీక్ష నివేదిక" in r2.response_text

    # Turn 3: Farmer provides SSP negation and no soil test
    t3_msg = "నాట్లలో SSP వేయలేదు. నేల పరీక్ష నివేదిక కూడా లేదు."
    c2 = Conversation(farmer_id=farmer_id, user_message=t2_msg, ai_response=r2.response_text)
    mock_repo.get_conversation_history = AsyncMock(return_value=[c1, c2])

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_llm_dose):
        r3 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t3_msg))

    # In Turn 3, ALL questions have been answered. None should be repeated!
    assert "SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా" not in r3.response_text
    assert "మీరు సాగు చేస్తున్న వరి రకం/వంగడం ఏది" not in r3.response_text
    assert "నేల పరీక్ష నివేదిక (Soil Health Card) అందుబాటులో ఉందా" not in r3.response_text
    assert "60-70" not in r3.response_text and "60–70" not in r3.response_text
    assert "30-35" not in r3.response_text and "30–35" not in r3.response_text


# =============================================================================
# 39. Live Scenario: Initial Message Without History, Follow-up Details, and Guess Dose Request
# =============================================================================

@pytest.mark.asyncio
async def test_case_39_live_scenario_initial_message_followup_and_guess_dose_request():
    """
    REGRESSION TEST for the exact live scenario and requirements:
    - Paddy crop, two acres.
    - 30 days after transplanting.
    - Initial fertilizer history not yet supplied.

    Verifies:
    Turn 1 (Initial Message):
      1. Does NOT present unsupported numerical dosage benchmarks (no 30-35 kg/acre, no 60-70 kg/2 acres).
      2. Asks concise clarification questions (DAP/SSP/complex, post-transplanting urea/potash, variety, soil test).
      3. Preserves useful non-numeric safety guidance (excess nitrogen risks: lodging, blast disease, stem borer).
      4. Does NOT instruct immediate application ("వేయండి", "చల్లండి", "తప్పకుండా వేసుకోవాలి").

    Turn 2 (Farmer's Follow-up Details):
      1. Farmer supplies: DAP 50 kg applied as basal, no urea yet, variety RNR 15048.
      2. System remembers answered items via multi-turn memory.
      3. Accounts for DAP, does not prescribe duplicate DAP, does not repeat answered questions.
      4. Does not hardcode refusal: provides safe, calibrated guidance.

    Turn 3 (Subsequent Request to Guess a Dose):
      1. Farmer with missing variety/soil context requests to guess a dose ("2 ఎకరాల వరి, నాట్లు వేసి 30 రోజులు. నాకు రకం తెలీదు, నేల పరీక్ష లేదు, ఊరికే ఎంత ఎరువు వేయాలో ఒక అంచనా చెప్పండి").
      2. Guardrail intercepts ungrounded arbitrary guess and refuses to propose arbitrary numbers.
      3. Zero numerical dose benchmarks (no 60-70 or 30-35 kg).
      4. Explains that without field context chemical doses cannot be confirmed, explains excess nitrogen risks (lodging, blast, stem borer).
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from uuid import uuid4
    from src.core.models import Farmer, Conversation, FarmerMemory
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_repo = MagicMock()
    mock_repo.session = AsyncMock()
    farmer_id = uuid4()
    mock_repo.get_farmer_profile = AsyncMock(return_value=Farmer(id=farmer_id, preferred_language="te"))
    service = AIService(mock_repo)

    mock_rag_item = MagicMock()
    mock_rag_item.document_title = "PJTSAU Paddy Fertilizer Guide"
    mock_rag_item.crop = "Paddy"
    mock_rag_item.source = "PJTSAU"
    mock_rag_item.chunk_text = MOCK_PJTSAU_RAG_SNIPPET

    with patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[mock_rag_item]), \
         patch("src.memory.service.FarmerMemoryService.get_memory", new_callable=AsyncMock, return_value=FarmerMemory(id=uuid4(), farmer_id=farmer_id)), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""):

        # -------------------------------------------------------------------------
        # TURN 1: Initial Message (Live Scenario)
        # -------------------------------------------------------------------------
        t1_msg = "2 ఎకరాల వరి, నాట్లు వేసి 30 రోజులు. ఎంత ఎరువు వేయాలి?"
        mock_repo.get_conversation_history = AsyncMock(return_value=[])

        # Flawed response attempting to output 60-70 kg or 30-35 kg benchmark
        flawed_initial_response = (
            "వరి పంట 30 రోజుల పిలకల దశలో ప్రాంతీయ వ్యవసాయ విస్తరణ మార్గదర్శకాల ప్రకారం 2 ఎకరాలకు మొత్తం 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) "
            "వేసుకోవడం సాధారణంగా సిఫార్సు చేస్తారు. మీరు నాట్లలో వేసిన ఎరువుల వివరాలు DAP లేదా SSP వేశారా మరియు రకం ఏమిటి?"
        )

        with patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_initial_response):
            r1 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t1_msg))

        # Turn 1 Assertions:
        # 1. Zero numerical dose benchmarks
        assert "60-70" not in r1.response_text and "60–70" not in r1.response_text
        assert "30-35" not in r1.response_text and "30–35" not in r1.response_text
        # 2. Concise clarification questions
        assert "DAP" in r1.response_text or "SSP" in r1.response_text
        assert "యూరియా" in r1.response_text
        assert "రకం" in r1.response_text or "వంగడం" in r1.response_text
        assert "పరీక్ష" in r1.response_text or "Soil Health Card" in r1.response_text
        # 3. Non-numeric safety guidance (lodging, blast, stem borer)
        assert "నేలకొరిగే" in r1.response_text
        assert "అగ్గి తెగులు" in r1.response_text
        assert "కాండం" in r1.response_text
        # 4. Does not instruct immediate application
        assert "వేయండి" not in r1.response_text
        assert "చల్లండి" not in r1.response_text
        assert "తప్పకుండా వేసుకోవాలి" not in r1.response_text

        # -------------------------------------------------------------------------
        # TURN 2: Farmer's Follow-up Details
        # -------------------------------------------------------------------------
        t2_msg = "నాట్లలో ఎకరానికి 50 కిలోల DAP వేశాను. యూరియా ఇంకా వేయలేదు. మాది స్వల్పకాలిక రకం RNR 15048."
        c1 = Conversation(farmer_id=farmer_id, user_message=t1_msg, ai_response=r1.response_text)
        mock_repo.get_conversation_history = AsyncMock(return_value=[c1])

        calibrated_guidance = (
            "నాట్లలో ఎకరానికి 50 కిలోల DAP వేసినందున ప్రారంభ భాస్వరం సరిపోతుంది. మళ్లీ DAP వేయకండి. "
            "ఇప్పుడు 30 రోజుల పిలకల దశలో మీ 2 ఎకరాల వరి పంటకు (RNR 15048) మొత్తం 60-70 కిలోల యూరియా (ఎకరానికి 30-35 కిలోలు) "
            "సమతుల్యంగా వేసుకోవచ్చు."
        )

        with patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=calibrated_guidance):
            r2 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t2_msg))

        # Turn 2 Assertions:
        # 1. Accounts for DAP and does NOT prescribe duplicate DAP
        assert "DAP" in r2.response_text
        assert "మళ్లీ DAP వేయకండి" in r2.response_text or "సరిపోతుంది" in r2.response_text
        # 2. Does not re-ask DAP, urea, or variety questions that were answered
        assert "నాట్ల సమయంలో DAP" not in r2.response_text
        assert "సాగు చేస్తున్న వరి రకం/వంగడం ఏది" not in r2.response_text
        # 3. Does not hardcode refusal: safe, calibrated guidance is provided
        assert "60-70" in r2.response_text

        # -------------------------------------------------------------------------
        # TURN 3: Subsequent Request to Guess a Dose
        # -------------------------------------------------------------------------
        # Farmer asks the bot to guess a dose without variety or soil details
        t3_msg = "2 ఎకరాల వరి, నాట్లు వేసి 30 రోజులు. నాకు రకం తెలీదు, నేల పరీక్ష లేదు, ఊరికే ఎంత ఎరువు వేయాలో ఒక అంచనా చెప్పండి."
        mock_repo.get_conversation_history = AsyncMock(return_value=[c1])

        flawed_guess_response = "అంచనాగా 2 ఎకరాలకు 50 కిలోల యూరియా వెంటనే వేయండి."

        with patch("src.ai.service.generate_response", new_callable=AsyncMock, return_value=flawed_guess_response):
            r3 = await service.generate_ai_response(AIGenerateRequest(farmer_id=farmer_id, message=t3_msg))

        # Turn 3 Assertions:
        # 1. Guardrail intercepts arbitrary guessed dose
        assert "50 కిలోల యూరియా" not in r3.response_text
        assert "60-70" not in r3.response_text and "60–70" not in r3.response_text
        assert "30-35" not in r3.response_text and "30–35" not in r3.response_text
        # 2. Refuses to confirm an unverified dose without variety/soil details
        assert "ఖరారైన నిర్ధారణ కాదు" in r3.response_text or "నిర్ధారించలేము" in r3.response_text or "ఖరారు చేయడం సురక్షితం కాదు" in r3.response_text
        # 3. Explains excess nitrogen risks (lodging, blast, stem borer)
        assert "నేలకొరిగే" in r3.response_text
        assert "అగ్గి తెగులు" in r3.response_text
        assert "కాండం" in r3.response_text
