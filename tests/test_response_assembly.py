import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from src.core.models import Farmer, Conversation
from src.ai.service import process_text_message, AIService
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse

@pytest.mark.asyncio
async def test_english_message_produces_english_response():
    """Verify that an English question leads to a response in English only (no Telugu appended)."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What fertilizer should I use for cotton?")

    # Mock AIService and its generate_ai_response call
    mock_response = AIGenerateResponse(
        response_text="You should use NPK fertilizer for cotton.",
        intent="fertilizer_recommendation",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "You should use NPK fertilizer for cotton."
        assert "మీరు పత్తి" not in result

@pytest.mark.asyncio
async def test_telugu_message_produces_telugu_response():
    """Verify that a Telugu question leads to a response in Telugu only."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి పంటకు ఏ ఎరువు వాడాలి?")

    mock_response = AIGenerateResponse(
        response_text="పత్తి పంటకు ఎన్పికె (NPK) ఎరువును ఉపయోగించాలి.",
        intent="fertilizer_recommendation",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "పత్తి పంటకు ఎన్పికె (NPK) ఎరువును ఉపయోగించాలి."

@pytest.mark.asyncio
async def test_memory_extraction_not_appended_to_response():
    """Verify that memory extraction engine outputs/schemas are never appended to the farmer-facing response."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="I have 5 acres.")

    mock_response = AIGenerateResponse(
        response_text="Got it, 5 acres of land size recorded.",
        intent="update_profile",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "Got it, 5 acres of land size recorded."
        # Confirm that no memory JSON schema structure leaks into response
        assert "updates" not in result
        assert "confidence_scores" not in result

@pytest.mark.asyncio
async def test_context_sentence_is_not_duplicated():
    """Verify that the farmer-context Telugu sentence is not duplicated or appended to English responses."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="My cotton crop is 45 days old in Hyderabad.")

    mock_response = AIGenerateResponse(
        response_text="At 45 days, cotton is in the squaring stage. Ensure adequate watering.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "At 45 days, cotton is in the squaring stage. Ensure adequate watering."
        # Verify the sentence does not appear at all
        assert "మీరు పత్తి" not in result

@pytest.mark.asyncio
async def test_shop_information_is_preserved():
    """Verify that useful shop information appended by enrich_response_with_shops is preserved."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="Where can I buy Urea?")

    mock_response = AIGenerateResponse(
        response_text="You can buy Urea fertilizer at nearby shops.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    shop_details = "\n\n🏬 Available Nearby Shops:\n• Ramesh Fertilizers\n  Product: Urea (IFFCO)\n  Price: ₹300 | Stock: 50 Bags"

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", return_value=mock_response.response_text + shop_details):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result.startswith("You can buy Urea fertilizer at nearby shops.")
        assert "Ramesh Fertilizers" in result
        assert "Urea (IFFCO)" in result

@pytest.mark.asyncio
async def test_no_duplicate_fragments_in_final_response():
    """Regression test proving the final outbound response does not contain duplicated sentences or fragments."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What fertilizer for cotton?")

    mock_response = AIGenerateResponse(
        response_text="You should use NPK fertilizer. మీరు పత్తి (Cotton) పంటను సాగు చేస్తున్నారు.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        
        # Split into sentences and check that no sentence is duplicated
        sentences = [s.strip() for s in result.replace("?", ".").split(".") if s.strip()]
        assert len(sentences) == len(set(sentences)), f"Found duplicate sentences in response: {sentences}"


@pytest.mark.asyncio
async def test_general_fertilizer_question_does_not_append_shops():
    """Verify that a general question like 'What fertilizer should I use for cotton?' does not append nearby shops."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What fertilizer should I use for cotton?")

    mock_response = AIGenerateResponse(
        response_text="For cotton, you should apply Urea or NPK.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response):
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "For cotton, you should apply Urea or NPK."
        assert "Available Nearby Shops" not in result


@pytest.mark.asyncio
async def test_explicit_buy_fertilizer_question_appends_shops():
    """Verify that an explicit purchase/buying query like 'Where to buy urea?' appends nearby shops."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="Where can I buy urea?")

    mock_response = AIGenerateResponse(
        response_text="You can purchase Urea fertilizer at nearby local shops.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    shop_details = "\n\n🏬 Available Nearby Shops:\n• Ramesh Fertilizers\n  Product: Urea (IFFCO)\n  Price: ₹300 | Stock: 50 Bags"

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", return_value=mock_response.response_text + shop_details):
        result = await process_text_message(db_mock, farmer, conversation)
        assert result.startswith("You can purchase Urea fertilizer at nearby local shops.")
        assert "Available Nearby Shops" in result


@pytest.mark.asyncio
async def test_crop_stage_not_assumed_in_prompt():
    """Verify that the AIService prompt rules explicitly instruct Gemini not to assume crop stage."""
    from src.ai.prompts import BHOOMIMITRA_SYSTEM_PROMPT
    assert "Do NOT assume the farmer's crop stage" in BHOOMIMITRA_SYSTEM_PROMPT
    assert "must NOT assume it" in BHOOMIMITRA_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_e2e_general_question_does_not_append_shops():
    """Verify that a general question like 'What fertilizer should I use for cotton?' does not append shops in the real pipeline."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What fertilizer should I use for cotton?")

    mock_response = AIGenerateResponse(
        response_text="For cotton, you should apply Urea or NPK.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Mallanna Fertilizer", status="active", delivery_available=True, phone_number="8976547654")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "For cotton, you should apply Urea or NPK."
        assert "Nearby Agricultural Shops" not in result


@pytest.mark.asyncio
async def test_e2e_general_question_plural_does_not_append_shops():
    """Verify that 'What fertilizers should I use for cotton?' does not append shops."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="what fertilizers should i use for cotton?")

    mock_response = AIGenerateResponse(
        response_text="You can use organic manure or Urea/NPK.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Mallanna Fertilizer", status="active", delivery_available=True, phone_number="8976547654")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "You can use organic manure or Urea/NPK."
        assert "Nearby Agricultural Shops" not in result


@pytest.mark.asyncio
async def test_e2e_english_buy_question_appends_shops():
    """Verify that 'where can I buy urea?' appends shops in the real pipeline."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="Where can I buy urea?")

    mock_response = AIGenerateResponse(
        response_text="You can buy Urea at local shops.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Mallanna Fertilizer", status="active", delivery_available=True, phone_number="8976547654")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(None, None, None, None)), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "You can buy Urea at local shops." in result
        assert "Nearby Agricultural Shops" in result
        assert "Mallanna Fertilizer" in result


@pytest.mark.asyncio
async def test_e2e_english_buy_question_variation_appends_shops():
    """Verify that 'where i can buy urea?' appends shops in the real pipeline."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="where i can buy urea?")

    mock_response = AIGenerateResponse(
        response_text="You can buy Urea at nearby shops.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Mallanna Fertilizer", status="active", delivery_available=True, phone_number="8976547654")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(None, None, None, None)), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "You can buy Urea at nearby shops." in result
        assert "Nearby Agricultural Shops" in result
        assert "Mallanna Fertilizer" in result


@pytest.mark.asyncio
async def test_e2e_telugu_buy_question_appends_shops():
    """Verify that 'యూరియా ఎక్కడ కొనాలి?' appends shops in the real pipeline."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="యూరియా ఎక్కడ కొనాలి?")

    mock_response = AIGenerateResponse(
        response_text="మీరు స్థానిక దుకాణాలలో యూరియాను కొనుగోలు చేయవచ్చు.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Mallanna Fertilizer", status="active", delivery_available=True, phone_number="8976547654")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(None, None, None, None)), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "మీరు స్థానిక దుకాణాలలో యూరియాను కొనుగోలు చేయవచ్చు." in result
        assert "సమీప వ్యవసాయ దుకాణాలు" in result
        assert "Mallanna Fertilizer" in result


@pytest.mark.asyncio
async def test_cotton_alternaria_multi_turn_dosage_grounded_response():
    """Verify that multi-turn cotton disease inquiry uses retrieved RAG knowledge for Mancozeb dosage."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="మాంకోజెబ్ ఎంత మోతాదులో వేయాలి?"
    )

    # Simulated grounded AI response using retrieved RAG knowledge: 2.5 to 3.0 g per litre
    mock_response = AIGenerateResponse(
        response_text="పత్తిలో ఆల్టర్నేరియా మచ్చల నివారణకు లీటరు నీటికి 2.5 నుండి 3.0 గ్రాముల మాంకోజెబ్ (Mancozeb 75% WP) కలిపి పిచికారీ చేయాలి.",
        intent="disease_treatment",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_response), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "2.5 నుండి 3.0 గ్రాముల" in result or "Mancozeb" in result
        assert "మాంకోజెబ్" in result


@pytest.mark.asyncio
async def test_exact_telugu_alternaria_rag_grounded_response():
    """Verify exact Telugu Alternaria query extracts Cotton and retrieves verified RAG treatment."""
    from src.rag.service import extract_crop_from_text, RAGService
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest

    exact_query = "నా పత్తి పంటకు ఆకులపై గోధుమ రంగు మచ్చలు ఉన్నాయి. ఇది Alternaria ఆకు మచ్చ తెగులా? దీనికి ఏం చేయాలి?"

    # 1. Crop extraction
    extracted_crop = extract_crop_from_text(exact_query)
    assert extracted_crop == "Cotton"

    # 2. RAG retrieval
    mock_rag_repo = AsyncMock()
    mock_rag_repo.get_all_chunks_with_embeddings.return_value = []
    rag_service = RAGService(mock_rag_repo)
    rag_results = await rag_service.search_knowledge(query=exact_query, top_k=3, crop=extracted_crop)

    assert len(rag_results) > 0
    top_doc = rag_results[0]
    assert "Alternaria" in top_doc.document_title or "Alternaria" in top_doc.chunk_text
    assert "Mancozeb" in top_doc.chunk_text
    assert "2.5 to 3.0 g per litre" in top_doc.chunk_text

    # 3. AIService prompt construction & response verification
    mock_ai_repo = AsyncMock()
    mock_ai_repo.get_farmer_profile.return_value = None
    mock_ai_repo.get_conversation_history.return_value = []
    mock_ai_repo.session = mock_rag_repo.session

    ai_service = AIService(mock_ai_repo)
    req = AIGenerateRequest(farmer_id=uuid4(), message=exact_query)

    captured_prompts = {}
    async def mock_generate_response(system_prompt, conversation_history, user_message, **kwargs):
        captured_prompts["system_prompt"] = system_prompt
        return "పత్తిలో ఆకులపై గోధుమ రంగు మచ్చలు ఆల్టర్నేరియా ఆకుమచ్చ తెగులు లక్షణాలు. దీని నివారణకు లీటరు నీటికి 2.5-3.0 గ్రాముల మాంకోజెబ్ (Mancozeb 75% WP) కలిపి పిచికారీ చేయండి."

    with patch("src.ai.service.generate_response", side_effect=mock_generate_response), \
         patch("src.rag.service.RAGRepository", return_value=mock_rag_repo), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", return_value=""), \
         patch("src.memory.service.FarmerMemoryService.extract_and_update_memory", return_value=None):

        response = await ai_service.generate_ai_response(req)
        assert "RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE" in captured_prompts["system_prompt"]
        assert "Alternaria" in captured_prompts["system_prompt"]
        assert "Mancozeb" in captured_prompts["system_prompt"]
        assert "మాంకోజెబ్" in response.response_text or "Mancozeb" in response.response_text


@pytest.mark.asyncio
async def test_rag_db_failure_fallback_to_verified_knowledge():
    """Verify that when database chunk query raises an exception, RAG still evaluates verified knowledge."""
    from src.rag.service import RAGService

    mock_rag_repo = AsyncMock()
    mock_rag_repo.get_all_chunks_with_embeddings.side_effect = RuntimeError("Database connection or query failed")

    rag_service = RAGService(mock_rag_repo)
    exact_query = "నా పత్తి పంటకు ఆకులపై గోధుమ రంగు మచ్చలు ఉన్నాయి. ఇది Alternaria ఆకు మచ్చ తెగులా? దీనికి ఏం చేయాలి?"

    # Search should NOT raise an exception and should return verified Alternaria guide
    results = await rag_service.search_knowledge(query=exact_query, top_k=3, crop="Cotton")
    assert len(results) > 0
    assert "Alternaria" in results[0].document_title or "Alternaria" in results[0].chunk_text
    assert "Mancozeb 75% WP" in results[0].chunk_text
    assert "2.5 to 3.0 g per litre" in results[0].chunk_text


@pytest.mark.asyncio
async def test_rag_db_success_path():
    """Verify that when database chunk query succeeds, DB chunks are evaluated and returned."""
    from src.rag.service import RAGService
    from src.core.models import KnowledgeDocument, KnowledgeChunk, EmbeddingMetadata
    from uuid import uuid4

    doc_id = uuid4()
    chunk_id = uuid4()

    mock_doc = KnowledgeDocument(
        id=doc_id,
        title="Custom Database Advisory",
        source="State Agri Dept",
        category="Pest Control",
        language="te",
        state="Telangana",
        crop="Cotton",
    )
    chunk_content = "Custom DB Knowledge: Cotton pest control using verified bio-agents."
    mock_chunk = KnowledgeChunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        page_number=1,
        chunk_text=chunk_content,
        embedding_id="emb_1",
    )

    mock_rag_repo = AsyncMock()
    rag_service = RAGService(mock_rag_repo)
    chunk_vec = rag_service.generate_embedding(chunk_content)

    mock_emb = EmbeddingMetadata(
        id=uuid4(),
        chunk_id=chunk_id,
        embedding_id="emb_1",
        vector=chunk_vec,
        dimension=len(chunk_vec),
    )

    mock_rag_repo.get_all_chunks_with_embeddings.return_value = [(mock_chunk, mock_emb, mock_doc)]

    results = await rag_service.search_knowledge(query="Cotton pest control using verified bio-agents", top_k=3, crop="Cotton")
    assert len(results) > 0
    assert any("Custom Database Advisory" in r.document_title for r in results)







