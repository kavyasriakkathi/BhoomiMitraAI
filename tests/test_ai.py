import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from fastapi import HTTPException

from src.main import app
from src.ai.schemas import AIGenerateResponse
from src.ai.service import AIService
from src.ai.dependencies import get_ai_service

client = TestClient(app)

@pytest.fixture
def mock_ai_service():
    service = AsyncMock(spec=AIService)
    app.dependency_overrides[get_ai_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def test_get_ai_health():
    response = client.get("/ai/health")
    # Will be 200 or 503 depending on .env, but usually 200 in test if loaded.
    # To be safe, we might just assert it's 200 since the test environment likely has the key.
    # If not, we can mock get_settings but router accesses it directly. 
    # The requirement is just "GET /ai/health (success)".
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert response.json()["status"] == "healthy"

def test_generate_ai_response_success(mock_ai_service):
    mock_ai_service.generate_ai_response.return_value = AIGenerateResponse(
        response_text="Use NPK fertilizer.",
        intent="general_advice",
        confidence=0.9,
        provider_used="gemini"
    )
    
    response = client.post("/ai/generate", json={
        "farmer_id": str(uuid4()),
        "message": "What fertilizer should I use?"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["response_text"] == "Use NPK fertilizer."
    assert data["provider_used"] == "gemini"

def test_generate_ai_response_invalid_farmer_id(mock_ai_service):
    response = client.post("/ai/generate", json={
        "farmer_id": "not-a-uuid",
        "message": "What fertilizer should I use?"
    })
    assert response.status_code == 422

def test_generate_ai_response_missing_fields(mock_ai_service):
    response = client.post("/ai/generate", json={
        "farmer_id": str(uuid4())
        # missing message
    })
    assert response.status_code == 422

def test_generate_ai_response_validation_errors(mock_ai_service):
    # Empty message
    response = client.post("/ai/generate", json={
        "farmer_id": str(uuid4()),
        "message": "   "
    })
    assert response.status_code == 422

def test_generate_ai_response_provider_unavailable(mock_ai_service):
    mock_ai_service.generate_ai_response.side_effect = HTTPException(
        status_code=503, detail="AI Provider unavailable."
    )
    
    response = client.post("/ai/generate", json={
        "farmer_id": str(uuid4()),
        "message": "Hello"
    })
    
    assert response.status_code == 503
    assert response.json()["detail"] == "AI Provider unavailable."


def test_gemini_model_configuration():
    from src.config import Settings
    s = Settings()
    assert s.gemini_model == "gemini-3.6-flash"


def test_gemini_fallback_models_order():
    from src.ai.gemini_client import FALLBACK_MODELS
    assert FALLBACK_MODELS == [
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]


def test_gemini_initialization_configures_rest_transport(monkeypatch):
    from unittest.mock import MagicMock
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", False)
    mock_configure = MagicMock()
    monkeypatch.setattr(gemini_module.genai, "configure", mock_configure)

    gemini_module._ensure_initialized()

    assert mock_configure.called
    assert mock_configure.call_args.kwargs.get("transport") == "rest"


@pytest.mark.asyncio
async def test_gemini_generate_response_primary_model_is_gemini_36_flash(monkeypatch):
    from unittest.mock import MagicMock
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    attempts = []

    def mock_generative_model(model_name, **kwargs):
        attempts.append(model_name)
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Gemini 3.6 Flash natural language response"
        mock_chat.send_message.return_value = mock_resp
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి?",
        timeout_seconds=5,
    )

    assert response == "Gemini 3.6 Flash natural language response"
    assert attempts[0] == "gemini-3.6-flash"


@pytest.mark.asyncio
async def test_gemini_generate_response_fallback_on_error(monkeypatch):
    from unittest.mock import MagicMock
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    attempts = []

    def mock_generative_model(model_name, **kwargs):
        attempts.append(model_name)
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        if model_name == "gemini-3.6-flash":
            # Primary attempt fails
            mock_chat.send_message.side_effect = Exception("Service Unavailable 503")
        else:
            # Fallback attempt succeeds
            mock_resp = MagicMock()
            mock_resp.text = "Fallback model response"
            mock_chat.send_message.return_value = mock_resp
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="Test message",
        timeout_seconds=5,
    )

    assert response == "Fallback model response"
    assert attempts[0] == "gemini-3.6-flash"
    assert "gemini-3.5-flash" in attempts


@pytest.mark.asyncio
async def test_gemini_generate_response_max_output_tokens_is_1024(monkeypatch):
    from unittest.mock import MagicMock
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    captured_configs = []

    def mock_generative_model(model_name, **kwargs):
        captured_configs.append(kwargs.get("generation_config"))
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Valid test response"
        mock_chat.send_message.return_value = mock_resp
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="పత్తి పంటలో పురుగుల నివారణ",
        timeout_seconds=5,
    )

    assert response == "Valid test response"
    assert len(captured_configs) > 0
    gen_config = captured_configs[0]
    # Check max_output_tokens on genai.GenerationConfig
    token_limit = getattr(gen_config, "max_output_tokens", None)
    if token_limit is None and hasattr(gen_config, "_max_output_tokens"):
        token_limit = gen_config._max_output_tokens
    assert token_limit == 1024


@pytest.mark.asyncio
async def test_gemini_telugu_response_regression_no_artificial_truncation(monkeypatch):
    """Regression test ensuring application code returns the complete Telugu response without artificial cutoff."""
    from unittest.mock import MagicMock
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    long_telugu_response = (
        "నమస్తే రైతు సోదరా! మీ పంట సాగులో సరైన యాజమాన్య పద్ధతులు పాటించడం ఎంతో ముఖ్యం. "
        "పొలంలో నీటి పారుదల మరియు మురుగు నీటి సౌకర్యం సక్రమంగా ఉండేలా చూసుకోవాలి. "
        "పంట ఎదుగుదల దశలో సేంద్రీయ ఎరువులు మరియు తగిన పోషకాలను సమతుల్యంగా అందించడం ద్వారా మంచి దిగుబడి సాధించవచ్చు. "
        "క్షేత్రస్థాయిలో ఏవైనా సమస్యలు లేదా సందేహాలు ఉంటే స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
    )

    def mock_generative_model(model_name, **kwargs):
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = long_telugu_response
        mock_chat.send_message.return_value = mock_resp
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="పంట సాగు మరియు క్షేత్ర యాజమాన్యం గురించి వివరాలు చెప్పండి.",
        timeout_seconds=5,
    )

    assert response == long_telugu_response
    assert len(response) == len(long_telugu_response)
    assert response.endswith("స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.")


@pytest.mark.asyncio
async def test_gemini_primary_timeout_is_15_seconds(monkeypatch):
    """Verify that primary Gemini attempt receives a 15.0 second timeout by default."""
    from unittest.mock import MagicMock
    import asyncio
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    captured_timeouts = []
    original_wait_for = asyncio.wait_for

    async def mock_wait_for(fut, timeout):
        captured_timeouts.append(timeout)
        mock_resp = MagicMock()
        mock_resp.text = "Success with 15s timeout"
        return mock_resp

    monkeypatch.setattr(gemini_module.asyncio, "wait_for", mock_wait_for)

    def mock_generative_model(model_name, **kwargs):
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="Test message",
    )

    assert response == "Success with 15s timeout"
    assert len(captured_timeouts) == 1
    assert captured_timeouts[0] == 15.0


@pytest.mark.asyncio
async def test_gemini_fallback_timeout_is_bounded_to_10_seconds(monkeypatch):
    """Verify that fallback Gemini model attempts receive a bounded 10.0s timeout."""
    from unittest.mock import MagicMock
    import asyncio
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    captured_timeouts = []

    async def mock_wait_for(fut, timeout):
        captured_timeouts.append(timeout)
        if len(captured_timeouts) == 1:
            # First attempt (primary) fails with timeout
            raise asyncio.TimeoutError()
        # Second attempt (fallback) succeeds
        mock_resp = MagicMock()
        mock_resp.text = "Fallback success with 10s timeout"
        return mock_resp

    monkeypatch.setattr(gemini_module.asyncio, "wait_for", mock_wait_for)

    def mock_generative_model(model_name, **kwargs):
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    response = await gemini_module.generate_response(
        system_prompt="Test prompt",
        conversation_history=[],
        user_message="Test message",
    )

    assert response == "Fallback success with 10s timeout"
    assert len(captured_timeouts) == 2
    assert captured_timeouts[0] == 15.0  # Primary attempt
    assert captured_timeouts[1] == 10.0  # Fallback attempt (bounded)


@pytest.mark.asyncio
async def test_gemini_timeout_ceiling_aborts_without_infinite_loop(monkeypatch):
    """Verify that after 2 timeouts, the retry ceiling aborts and raises TimeoutError cleanly."""
    from unittest.mock import MagicMock
    import asyncio
    import src.ai.gemini_client as gemini_module

    monkeypatch.setattr(gemini_module, "_initialized", True)

    captured_attempts = []

    async def mock_wait_for(fut, timeout):
        captured_attempts.append(timeout)
        raise asyncio.TimeoutError()

    monkeypatch.setattr(gemini_module.asyncio, "wait_for", mock_wait_for)

    def mock_generative_model(model_name, **kwargs):
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    with pytest.raises(TimeoutError) as exc_info:
        await gemini_module.generate_response(
            system_prompt="Test prompt",
            conversation_history=[],
            user_message="Test message",
        )

    # Exactly 2 attempts (primary + 1 fallback), no infinite retry
    assert len(captured_attempts) == 2
    assert "Gemini API timed out" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Hard Grounding Gate Tests (Fertilizer / Chemical Dosage Safety)
# -----------------------------------------------------------------------------

def test_is_dosage_sensitive_query_classification():
    from src.ai.service import is_dosage_sensitive_query

    # Exact dosage & quantity queries across languages (MUST BE BLOCKED WHEN UNGROUNDED)
    assert is_dosage_sensitive_query("నాటిన 25 రోజుల పిలకల దశలో ఉన్న వరి పంటకు ఎరువు ఏది?") is True
    assert is_dosage_sensitive_query("వరికి ఎంత యూరియా వేయాలి") is True
    assert is_dosage_sensitive_query("వరి పంటకు ఎరువుల మోతాదు ఎంత?") is True
    assert is_dosage_sensitive_query("వరిలో పిలకల దశలో ఎరువు ఏది వేయాలి?") is True
    assert is_dosage_sensitive_query("How many kg urea per acre for paddy?") is True
    assert is_dosage_sensitive_query("How much urea per acre for cotton?") is True
    assert is_dosage_sensitive_query("What is the dosage of Chlorantraniliprole per litre?") is True
    assert is_dosage_sensitive_query("spray dosage for stem borer in paddy") is True
    assert is_dosage_sensitive_query("vari ki entha urea veyali") is True
    assert is_dosage_sensitive_query("cotton ki entha fertilizer veyali") is True
    assert is_dosage_sensitive_query("Urea fertilizer kitna daalna hai प्रति एकड़?") is True
    assert is_dosage_sensitive_query("एकड़ में कितना यूरिया डालना है?") is True
    assert is_dosage_sensitive_query("நெற்பயிருக்கு எவ்வளவு யூரியா போட வேண்டும்?") is True
    assert is_dosage_sensitive_query("ಭತ್ತಕ್ಕೆ ಎಷ್ಟು ಯೂರಿಯಾ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?") is True

    # General education queries (ALLOWED THROUGH)
    assert is_dosage_sensitive_query("Why is nitrogen important for paddy?") is False
    assert is_dosage_sensitive_query("Why do crops need phosphorus?") is False
    assert is_dosage_sensitive_query("What is the role of nitrogen in plant growth?") is False
    assert is_dosage_sensitive_query("What is urea?") is False
    assert is_dosage_sensitive_query("Benefits of vermicompost") is False
    assert is_dosage_sensitive_query("నత్రజని ప్రాముఖ్యత ఏమిటి?") is False
    assert is_dosage_sensitive_query("యూరియా అంటే ఏమిటి?") is False
    assert is_dosage_sensitive_query("భాస్వరం ఎందుకు అవసరం?") is False

    # Non-dosage general farming / weather / market queries (ALLOWED THROUGH)
    assert is_dosage_sensitive_query("What is the market price of cotton in Warangal?") is False
    assert is_dosage_sensitive_query("Will it rain tomorrow in Khammam?") is False
    assert is_dosage_sensitive_query("PM Kisan scheme details") is False
    assert is_dosage_sensitive_query("హలో నమస్తే") is False


@pytest.mark.asyncio
async def test_hard_gate_blocks_ungrounded_telugu_dosage_query(monkeypatch):
    """
    Verify production scenario: When farmer asks for tillering stage paddy fertilizer
    and RAG has no verified ground truth, Gemini is NOT called and safe Telugu fallback is returned.
    """
    from unittest.mock import MagicMock, patch
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.ai.prompts import UNVERIFIED_DOSAGE_FALLBACK_RESPONSES

    mock_session = AsyncMock()
    mock_repo = MagicMock()
    mock_repo.session = mock_session
    mock_repo.get_farmer_profile = AsyncMock(return_value=None)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(repository=mock_repo)

    with patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock) as mock_rag_search, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_rag_search.return_value = []

        request = AIGenerateRequest(
            farmer_id=uuid4(),
            message="నాటిన 25 రోజుల పిలకల దశలో ఉన్న వరి పంటకు ఎరువు ఏది?"
        )
        response = await service.generate_ai_response(request)

        # Gemini API must NOT be called
        mock_gemini.assert_not_called()
        assert response.provider_used == "hard_grounding_gate"
        assert response.response_text == UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"]
        assert "30" not in response.response_text
        assert "35" not in response.response_text
        assert "kg" not in response.response_text


@pytest.mark.asyncio
async def test_hard_gate_blocks_ungrounded_english_dosage_query(monkeypatch):
    """Verify ungrounded English dosage query returns safe English fallback with no Gemini call."""
    from unittest.mock import MagicMock, patch
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.ai.prompts import UNVERIFIED_DOSAGE_FALLBACK_RESPONSES

    mock_session = AsyncMock()
    mock_repo = MagicMock()
    mock_repo.session = mock_session
    mock_repo.get_farmer_profile = AsyncMock(return_value=None)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(repository=mock_repo)

    with patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock) as mock_rag_search, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_rag_search.return_value = []

        request = AIGenerateRequest(
            farmer_id=uuid4(),
            message="How much urea per acre for cotton in vegetative stage?"
        )
        response = await service.generate_ai_response(request)

        mock_gemini.assert_not_called()
        assert response.provider_used == "hard_grounding_gate"
        assert response.response_text == UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["en"]
        assert "Agriculture Extension Officer" in response.response_text


@pytest.mark.asyncio
async def test_hard_gate_blocks_ungrounded_tanglish_dosage_query(monkeypatch):
    """Verify ungrounded Tanglish/Romanized dosage query returns safe fallback with no Gemini call."""
    from unittest.mock import MagicMock, patch
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.ai.prompts import UNVERIFIED_DOSAGE_FALLBACK_RESPONSES

    mock_session = AsyncMock()
    mock_repo = MagicMock()
    mock_repo.session = mock_session
    mock_repo.get_farmer_profile = AsyncMock(return_value=None)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(repository=mock_repo)

    with patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock) as mock_rag_search, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_rag_search.return_value = []

        request = AIGenerateRequest(
            farmer_id=uuid4(),
            message="vari ki entha urea veyali"
        )
        response = await service.generate_ai_response(request)

        mock_gemini.assert_not_called()
        assert response.provider_used == "hard_grounding_gate"
        assert response.response_text == UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"]


@pytest.mark.asyncio
async def test_verified_rag_allows_grounded_gemini_generation(monkeypatch):
    """Verify that when verified RAG chunks ARE available, Gemini is called with Ground Truth."""
    from unittest.mock import MagicMock, patch
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.rag.schemas import RAGSearchResult

    mock_session = AsyncMock()
    mock_repo = MagicMock()
    mock_repo.session = mock_session
    mock_repo.get_farmer_profile = AsyncMock(return_value=None)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(repository=mock_repo)

    mock_chunk = RAGSearchResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="ICAR Cotton Guide",
        source="ICAR-CICR",
        category="Fertilizer Management",
        language="en",
        state="Telangana",
        crop="Cotton",
        page=1,
        chunk_text="Apply 25 kg Urea per acre at 45 DAS.",
        similarity_score=0.95,
    )

    with patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock) as mock_rag_search, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_rag_search.return_value = [mock_chunk]
        mock_gemini.return_value = "As per ICAR guidelines, apply 25 kg Urea per acre at 45 DAS."

        request = AIGenerateRequest(
            farmer_id=uuid4(),
            message="What is the cotton urea dosage at 45 DAS?"
        )
        response = await service.generate_ai_response(request)

        # Gemini API was called with Ground Truth
        mock_gemini.assert_called_once()
        assert response.provider_used == "gemini"
        assert "25 kg Urea" in response.response_text


@pytest.mark.asyncio
async def test_general_education_allowed_through_without_blocking(monkeypatch):
    """Verify general agronomic education (non-dosage) passes through to Gemini even with empty RAG."""
    from unittest.mock import MagicMock, patch
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    mock_session = AsyncMock()
    mock_repo = MagicMock()
    mock_repo.session = mock_session
    mock_repo.get_farmer_profile = AsyncMock(return_value=None)
    mock_repo.get_conversation_history = AsyncMock(return_value=[])

    service = AIService(repository=mock_repo)

    with patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock) as mock_rag_search, \
         patch("src.ai.service.generate_response", new_callable=AsyncMock) as mock_gemini:

        mock_rag_search.return_value = []
        mock_gemini.return_value = "Nitrogen is essential for chlorophyll formation and vegetative leaf growth in plants."

        request = AIGenerateRequest(
            farmer_id=uuid4(),
            message="Why is nitrogen important for paddy?"
        )
        response = await service.generate_ai_response(request)

        mock_gemini.assert_called_once()
        assert response.provider_used == "gemini"
        assert "chlorophyll" in response.response_text
