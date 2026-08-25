import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from datetime import datetime
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


@pytest.mark.asyncio
async def test_telugu_market_query_deduplication():
    """Verify 'పత్తి మార్కెట్ ధర ఎంత?' returns structured market block without duplicate Gemini price quotes."""
    from src.core.models import MarketPrice
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి మార్కెట్ ధర ఎంత?")

    # Gemini outputs speculative price text
    mock_ai_resp = AIGenerateResponse(
        response_text="ప్రస్తుతం పత్తి మార్కెట్ ధర క్వింటాలుకు దాదాపు ₹7,000 నుండి ₹7,500 వరకు ఉంది.",
        intent="market_price",
        confidence=0.9,
        provider_used="gemini",
    )

    mock_price = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=datetime(2026, 8, 19),
        unit="Quintal",
        source="local_db",
        created_at=datetime.utcnow(),
    )

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp), \
         patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)

        # 1. Contains authoritative structured block
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result

        # 2. Duplicate speculative price numbers from Gemini are stripped
        assert "₹7,000 నుండి ₹7,500" not in result
        assert "దాదాపు ₹7,000" not in result


@pytest.mark.asyncio
async def test_english_market_query_deduplication():
    """Verify 'What is the cotton market price?' returns structured market block without duplicate Gemini price text."""
    from src.core.models import MarketPrice
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What is the cotton market price?")

    # Gemini outputs speculative English price text
    mock_ai_resp = AIGenerateResponse(
        response_text="The current cotton market price is around ₹7,000 to ₹7,500 per quintal in Telangana mandis.",
        intent="market_price",
        confidence=0.9,
        provider_used="gemini",
    )

    mock_price = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=datetime(2026, 8, 19),
        unit="Quintal",
        source="local_db",
        created_at=datetime.utcnow(),
    )

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp), \
         patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)

        # 1. Contains authoritative structured block
        assert "📊 Cotton Mandi Prices" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result

        # 2. Duplicate speculative price numbers from Gemini are stripped
        assert "around ₹7,000 to ₹7,500" not in result
        assert "is around ₹7,000" not in result


@pytest.mark.asyncio
async def test_dual_crop_advice_and_market_query_preserves_advice():
    """Verify that a dual query preserves agronomic advice while stripping duplicate speculative price quotes."""
    from src.core.models import MarketPrice
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="నా పత్తి పంటకు ఆకుమచ్చ తెగులు ఉంది, మందు ఏమిటి? మరియు పత్తి మార్కెట్ ధర ఎంత?"
    )

    mock_ai_resp = AIGenerateResponse(
        response_text="పత్తిలో ఆకుమచ్చ తెగులు నివారణకు లీటరు నీటికి 2.5 గ్రాముల మాంకోజెబ్ కలిపి పిచికారీ చేయండి. ప్రస్తుతం పత్తి ధర క్వింటాలుకు ₹7,200 ఉంది.",
        intent="disease_and_market",
        confidence=0.9,
        provider_used="gemini",
    )

    mock_price = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=datetime(2026, 8, 19),
        unit="Quintal",
        source="local_db",
        created_at=datetime.utcnow(),
    )

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp), \
         patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)

        # Useful agronomic advice is preserved
        assert "మాంకోజెబ్" in result
        assert "ఆకుమచ్చ తెగులు నివారణకు" in result

        # Authoritative market block is appended
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result

        # Speculative price quote is stripped
        assert "₹7,200" not in result


@pytest.mark.asyncio
async def test_normal_non_market_agricultural_query_unchanged():
    """Verify that a normal non-market agricultural query returns unmodified AI response without price blocks."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరి పంటలో కలుపు నివారణ ఎలా చేయాలి?")

    mock_ai_resp = AIGenerateResponse(
        response_text="వరి పంటలో కలుపు నివారణకు ముందస్తుగా బ్యూటాక్లోర్ పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp):
        result = await process_text_message(db_mock, farmer, conversation)
        assert result == "వరి పంటలో కలుపు నివారణకు ముందస్తుగా బ్యూటాక్లోర్ పిచికారీ చేయండి."
        assert "📊" not in result
        assert "మార్కెట్ ధరలు" not in result


@pytest.mark.asyncio
async def test_multi_intent_query_response_compression():
    """Verify that when multiple enrichments trigger, response is budgeted and does not exceed limit."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి మార్కెట్ ధర ఎంత మరియు వర్షం పడుతుందా?"
    )

    mock_ai_resp = AIGenerateResponse(
        response_text="పత్తి సమాచారం.",
        intent="multi_query",
        confidence=0.9,
        provider_used="gemini",
    )

    price_block = "📊 పత్తి మార్కెట్ ధరలు\nమండి: Warangal Mandi, Telangana\nమోడల్ ధర: ₹7,450/క్వింటాల్కు\nకనిష్ట: ₹7,100 | గరిష్ట: ₹7,650\nతేదీ: 19 Aug 2026\n\n📡 స్థానిక డేటాబేస్"
    weather_block = "🌤️ వాతావరణ సూచన: Warangal\nప్రస్తుత ఉష్ణోగ్రత: 28°C\nవాతావరణం: Partly Cloudy\nతేమ: 65%\nగాలి వేగం: 10.0 km/h\n\n5-రోజుల సూచన:\n• 2026-08-26: Partly Cloudy (29°C)\n• 2026-08-27: Light Rain (27°C)"

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp), \
         patch("src.market.service.enrich_response_with_market_prices", return_value="పత్తి సమాచారం.\n\n" + price_block), \
         patch("src.weather.service.enrich_response_with_weather", return_value="పత్తి సమాచారం.\n\n" + price_block + "\n\n" + weather_block):

        result = await process_text_message(db_mock, farmer, conversation)
        assert len(result) <= 1600
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "🌤️ వాతావరణ సూచన" in result


@pytest.mark.asyncio
async def test_gemini_fallback_with_market_query():
    """Verify that when Gemini is unavailable, fallback is handled gracefully and market prices are appended."""
    from fastapi import HTTPException
    from src.core.models import MarketPrice
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి మార్కెట్ ధర ఎంత?")

    mock_price = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=datetime(2026, 8, 19),
        unit="Quintal",
        source="local_db",
        created_at=datetime.utcnow(),
    )

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    # AIService raises HTTPException 503
    with patch("src.ai.service.AIService.generate_ai_response", side_effect=HTTPException(status_code=503)), \
         patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result
