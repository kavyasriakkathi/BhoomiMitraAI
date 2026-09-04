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
    from datetime import datetime

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి మార్కెట్ ధర ఎంత?")

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
        unit="Quintal",
        price_date=datetime(2026, 8, 19),
        source="local_seed",
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

    db_mock.execute.side_effect = [
        mock_profile_res,
        mock_count_res,
        mock_query_res,
        mock_query_res,
        mock_query_res,
    ]

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock, return_value=mock_ai_resp), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result
        assert "₹7,000" not in result


@pytest.mark.asyncio
async def test_gemini_failure_with_telugu_market_price_query_returns_clean_market_data():
    """
    Regression test: When Gemini API times out or fails on a Telugu market-price query,
    specialized market enrichment still provides valid price data without prepending the
    generic Gemini connection failure fallback message.
    """
    from src.market.service import enrich_response_with_market_prices
    from src.core.models import MarketPrice
    from datetime import datetime

    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి మార్కెట్ ధర ఎంత?"
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
        unit="Quintal",
        price_date=datetime(2026, 8, 19),
        source="local_seed",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars

    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [
        mock_profile_res,  # farmer profile in market enricher
        mock_count_res,    # seed check
        mock_query_res,    # db records
        mock_query_res,
        mock_query_res,
    ]

    # Simulate Gemini failing with TimeoutError / 500 HTTPException
    with patch("src.ai.service.AIService.generate_ai_response", side_effect=RuntimeError("Gemini API timed out after 10s across all attempts")), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)

        # 1. Must contain the specialized market price data
        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result

        # 2. Must NOT contain the generic connection fallback error message
        assert "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది" not in result
        assert "I'm sorry, I'm having trouble connecting right now" not in result


@pytest.mark.asyncio
async def test_gemini_failure_without_specialized_intent_returns_fallback_message():
    """
    Verify that when Gemini fails and NO specialized enrichment applies,
    the generic fallback message is correctly returned.
    """
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="సాధారణ సమాచారం చెప్పండి"
    )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=RuntimeError("Gemini timeout")), \
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp, *args, **kwargs: resp):

        result = await process_text_message(db_mock, farmer, conversation)

        # Must return the localized Telugu fallback message
        assert "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది" in result


# ---------------------------------------------------------------------------
# Multi-Intent Response Optimization Tests
# ---------------------------------------------------------------------------

def test_clean_introductions():
    """Verify that redundant greetings, bot self-introductions, and filler preamble are removed."""
    from src.ai.formatting import clean_introductions

    assert clean_introductions("Hello! You should spray Mancozeb.") == "You should spray Mancozeb."
    assert clean_introductions("Namaste farmer! Here is the information: Apply Urea.") == "Apply Urea."
    assert clean_introductions("నమస్తే రైతు సోదరా! పత్తిలో పురుగు నివారణకు మందు పిచికారీ చేయండి.") == "పత్తిలో పురుగు నివారణకు మందు పిచికారీ చేయండి."
    assert clean_introductions("నేను భూమిమిత్ర. మీరు అడిగిన వివరాలు ఇక్కడ ఉన్నాయి:") == ""


def test_multi_intent_formatting_english():
    """Verify that multi-intent English responses are organized into clear sections and stripped of filler."""
    from src.ai.formatting import format_multi_intent_response

    raw_response = (
        "Hello! I am BhoomiMitra. For cotton bollworm, apply Emamectin Benzoate 5% SG @ 0.4g/litre. "
        "Regarding weather, it will be 32 degrees in Warangal. Cotton price is 7500.\n\n"
        "🌡️ Weather Information (Warangal)\n"
        "🌡️ Temperature: 32.0°C (Feels Like: 34.0°C)\n"
        "☁️ Condition: Clear Sky\n"
        "💧 Humidity: 65%\n"
        "💨 Wind Speed: 12.0 km/h\n"
        "📅 ☀️ Tomorrow's Forecast: Weather is expected to be clear/partly cloudy and dry.\n"
        "📡 OpenWeather (Live)\n\n"
        "🏬 Nearby Agricultural Shops & Availability:\n"
        "• *Kisan Agro Kendra* (5.0 km away)\n"
        "  📦 Product: Urea (IFFCO)\n"
        "  💰 Price: ₹295/Bag | In Stock (100 Bags)\n"
        "  📞 Contact: 9876543210 | Open (08:00 - 20:00)\n"
        "  🚚 Delivery: Available\n\n"
        "ℹ️ Note: Prices and stock levels are subject to local dealer confirmation.\n"
        "Find all shops at: /shops\n\n"
        "📊 Cotton Mandi Prices\n"
        "Market: Warangal Mandi, Telangana\n"
        "Modal Price: ₹7,450/per Quintal\n"
        "Min: ₹7,100 | Max: ₹7,650\n"
        "Date: 25 Aug 2026\n\n"
        "📡 Agmarknet (Live)"
    )

    optimized = format_multi_intent_response(raw_response, language="en")

    # 1. Clear section headers
    assert "🌱 *Crop Advice*" in optimized
    assert "🌡️ *Weather Information* (Warangal)" in optimized
    assert "🏬 *Nearby Shops & Availability*:" in optimized
    assert "📊 Cotton Mandi Prices" in optimized

    # 2. Repeated filler stripped from crop advice
    assert "Hello!" not in optimized
    assert "I am BhoomiMitra" not in optimized
    # Speculative text stripped
    assert "Regarding weather" not in optimized

    # 3. Verified information preserved
    assert "Emamectin Benzoate 5% SG @ 0.4g/litre" in optimized
    assert "Temperature: 32.0°C" in optimized
    assert "Kisan Agro Kendra" in optimized
    assert "₹7,450" in optimized

    # 4. Correct section ordering (Crop Advice before Weather, Weather before Shop, Shop before Market)
    idx_crop = optimized.find("🌱 *Crop Advice*")
    idx_weather = optimized.find("🌡️ *Weather Information*")
    idx_shop = optimized.find("🏬 *Nearby Shops")
    idx_market = optimized.find("📊 Cotton Mandi Prices")

    assert idx_crop < idx_weather < idx_shop < idx_market


def test_multi_intent_formatting_telugu():
    """Verify that multi-intent Telugu responses use clean Telugu section headers."""
    from src.ai.formatting import format_multi_intent_response

    raw_response = (
        "నమస్తే! పత్తిలో ఆకుమచ్చ తెగులు నివారణకు మాంకోజెబ్ 2.5 గ్రా/లీటర్ నీటిలో కలిపి పిచికారీ చేయండి.\n\n"
        "🌡️ వాతావరణ సమాచారం (వరంగల్)\n"
        "🌡️ ఉష్ణోగ్రత: 31.5°C (అనిపిస్తుంది: 33.0°C)\n"
        "☁️ వాతావరణం: ఆకాశం నిర్మలంగా ఉంది (Clear Sky)\n"
        "💧 తేమ (Humidity): 70%\n"
        "💨 గాలి వేగం: 10.0 km/h\n"
        "📅 ☀️ రేపటి అంచనా: వాతావరణం పొడిగా మరియు అనుకూలంగా ఉంటుంది.\n"
        "📡 ఓపెన్వెదర్ (లైవ్)\n\n"
        "📊 పత్తి మార్కెట్ ధరలు\n"
        "మండి: వరంగల్ మార్కెట్, Telangana\n"
        "మోడల్ ధర: ₹7,450/క్వింటాల్కు\n"
        "కనిష్ట: ₹7,100 | గరిష్ట: ₹7,650\n"
        "తేదీ: 25 Aug 2026\n\n"
        "📡 స్థానిక డేటాబేస్"
    )

    optimized = format_multi_intent_response(raw_response, language="te")

    assert "🌱 *పంట సలహా*" in optimized
    assert "🌡️ *వాతావరణ సమాచారం* (వరంగల్)" in optimized
    assert "📊 పత్తి మార్కెట్ ధరలు" in optimized
    assert "నమస్తే!" not in optimized
    assert "మాంకోజెబ్ 2.5 గ్రా/లీటర్" in optimized
    assert "7,450" in optimized


def test_single_intent_preserved_unaltered():
    """Verify that single-intent responses remain identical to maintain backwards compatibility."""
    from src.ai.formatting import format_multi_intent_response

    pure_crop = "For cotton, apply NPK 20:20:0:13 at 50kg per acre."
    assert format_multi_intent_response(pure_crop, language="en") == pure_crop

    pure_market = "📊 Tomato Mandi Prices\nMarket: Kolar\nModal Price: ₹1,500/per Quintal\nMin: ₹1,200 | Max: ₹1,800\nDate: 25 Aug 2026\n\n📡 Agmarknet (Live)"
    assert format_multi_intent_response(pure_market, language="en") == pure_market


@pytest.mark.asyncio
async def test_e2e_multi_intent_process_text_message():
    """Verify end-to-end multi-intent pipeline with crop advisory, weather, and shop enrichments."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="What fertilizer for cotton, where to buy urea, and what is the weather in Warangal?"
    )

    mock_ai_response = AIGenerateResponse(
        response_text="Hello! For cotton, use Urea 50kg/acre during vegetative growth. Regarding weather, it will be sunny in Warangal.",
        intent="multi_intent",
        confidence=0.9,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="Balaji Agro", status="active", delivery_available=True, phone_number="9848012345")
    mock_inventory = Inventory(product_name="Urea", brand="IFFCO", price=295.0, quantity_in_stock=50, unit="Bag")
    mock_search_results = [(mock_shop, mock_inventory)]

    weather_block = (
        "\n\n🌡️ Weather Information (Warangal)\n"
        "🌡️ Temperature: 32.0°C (Feels Like: 34.0°C)\n"
        "☁️ Condition: Clear\n"
        "💧 Humidity: 60%\n"
        "💨 Wind Speed: 10.0 km/h\n"
        "📅 ☀️ Tomorrow's Forecast: Weather is expected to be clear/partly cloudy and dry.\n"
        "📡 OpenWeather (Live)"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_response), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(None, None, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results), \
         patch("src.weather.service.enrich_response_with_weather", side_effect=lambda db, msg, resp, *args, **kwargs: resp + weather_block):

        result = await process_text_message(db_mock, farmer, conversation)

        # Verify structured section headers exist
        assert "🌱 *Crop Advice*" in result
        assert "🌡️ *Weather Information* (Warangal)" in result
        assert "🏬 *Nearby Shops & Availability*:" in result
        assert "Balaji Agro" in result
        assert "Urea 50kg/acre" in result
        assert "Hello!" not in result


@pytest.mark.asyncio
async def test_gemini_timeout_fast_failover_to_specialized_services():
    """
    Verify that when Gemini API times out, the error is caught gracefully and
    the pipeline immediately allows specialized enrichment (e.g. shops/mandi/weather) to respond.
    """
    db_mock = AsyncMock()
    farmer = Farmer(
        id=uuid4(),
        phone_number="919876543210",
        preferred_language="te"
    )
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="యూరియా ఎక్కడ దొరుకుతుంది?"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="శ్రీ బాలాజీ ఆగ్రో", status="active", delivery_available=True, phone_number="9848012345")
    mock_inventory = Inventory(product_name="యూరియా", brand="IFFCO", price=295.0, quantity_in_stock=40, unit="బస్తా")
    mock_search_results = [(mock_shop, mock_inventory)]

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=TimeoutError("Gemini API timed out")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(None, None, "వరంగల్", "తెలంగాణ")), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "శ్రీ బాలాజీ ఆగ్రో" in result
        assert "యూరియా" in result
        assert "295" in result


@pytest.mark.asyncio
async def test_gemini_timeout_general_query_returns_fallback_response():
    """
    Verify that when Gemini times out on a general query with no specialized intents,
    the localized fallback response is returned.
    """
    db_mock = AsyncMock()
    farmer = Farmer(
        id=uuid4(),
        phone_number="919876543210",
        preferred_language="te"
    )
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="సాధారణ సమాచారం చెప్పండి"
    )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=TimeoutError("Gemini API timed out")):
        result = await process_text_message(db_mock, farmer, conversation)
        assert len(result) > 0
        assert "క్షమించండి" in result or "సమస్య" in result or "try again" in result.lower()


def test_gemini_config_timeout_field():
    """Verify that gemini_api_timeout_seconds is configured in Settings."""
    from src.config import get_settings
    settings = get_settings()
    assert hasattr(settings, "gemini_api_timeout_seconds")
    assert settings.gemini_api_timeout_seconds == 5.0


@pytest.mark.asyncio
async def test_multi_intent_all_services_successful_deduplicated():
    """
    Verify that when a farmer asks multi-intent (Crop Advice + Shop + Weather + Schemes),
    the response is organized into clear sections, removes duplicate intros/outros,
    and excludes unrequested domains (Market).
    """
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది? రేపు వర్షం పడుతుందా? రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="నమస్తే రైతు మిత్రమా! పత్తి పంటలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు లీటరు నీటికి కలిపి పిచికారీ చేయండి. మీకు ఇంకా ఏమైనా సహాయం కావాలా?",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory, GovernmentScheme
    mock_shop = Shop(shop_name="శ్రీ బాలాజీ ఆగ్రో", status="active", delivery_available=True, phone_number="9848012345")
    mock_inventory = Inventory(product_name="యూరియా", brand="IFFCO", price=295.0, quantity_in_stock=40, unit="బస్తా")
    mock_search_results = [(mock_shop, mock_inventory)]

    mock_scheme = GovernmentScheme(
        scheme_name="పీఎం కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
        scheme_code="PM_KISAN",
        category="Direct Income Support",
        description="రైతులకు ఏటా ₹6000 ఆర్థిక సహాయం.",
        state="Telangana",
        benefits_summary="ఏడాదికి ₹6,000",
        eligibility_criteria="భూమి ఉన్న రైతులు",
        required_documents="ఆధార్, పట్టాదారు పాస్ బుక్",
        official_portal_url="https://pmkisan.gov.in"
    )

    from src.weather.schemas import WeatherForecastResponse
    mock_weather = WeatherForecastResponse.model_validate({
        "location_name": "Warangal",
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {"temp": 29.5, "feels_like": 31.0, "humidity": 75, "wind_speed": 12.0, "description": "Light Rain", "condition_code": 500},
        "forecast": [{"dt_txt": "2026-08-28 12:00:00", "temp": 28.0, "humidity": 80, "description": "Light Rain", "condition_code": 500}],
        "data_available": True,
        "is_live": True,
        "source_note": "OpenWeather (Live)"
    })

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=mock_weather), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=[mock_scheme]):

        result = await process_text_message(db_mock, farmer, conversation)

        # Check sections present
        assert "🌱 *పంట సలహా*" in result
        assert "🌡️ *వాతావరణ సమాచారం*" in result
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in result
        assert "🏛️ *ప్రభుత్వ పథకాలు*" in result
        assert "📊 *మార్కెట్ ధరలు*" not in result  # Not requested

        # Check content details preserved
        assert "ఎమామెక్టిన్ బెంజోయేట్" in result
        assert "0.4 గ్రాములు" in result
        assert "శ్రీ బాలాజీ ఆగ్రో" in result
        assert "295" in result
        assert "పీఎం కిసాన్" in result

        # Check repetitive intros/outros removed
        assert "నమస్తే" not in result
        assert "మీకు ఇంకా ఏమైనా సహాయం కావాలా" not in result


@pytest.mark.asyncio
async def test_multi_intent_weather_unavailable_shows_clean_notice():
    """Verify that when weather is unavailable in multi-intent, a single concise notice is shown."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి? వరంగల్లో రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తి పంటలో పురుగుల నివారణకు మోనోక్రోటోఫాస్ 1.5 మి.లీ లీటరు నీటికి వాడండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.weather.service.WeatherService.get_weather_for_query", side_effect=Exception("API down")):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "🌱 *పంట సలహా*" in result
        assert "మోనోక్రోటోఫాస్" in result
        assert "🌡️ *వాతావరణ సమాచారం*" in result
        assert "వాతావరణ సమాచారం అందుబాటులో లేదు" in result


@pytest.mark.asyncio
async def test_multi_intent_shops_unavailable_shows_clean_notice():
    """Verify that when shops are unavailable in multi-intent, a single concise shop notice is shown."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="వరి పంటలో తెగులు నివారణ ఎలా? ప్రత్యేక మందు ఎక్కడ దొరుకుతుంది?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="వరి పంటలో తెగులు నివారణకు హెక్సాకోనాజోల్ 2 మి.లీ లీటరు నీటికి వాడండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=[]):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "🌱 *పంట సలహా*" in result
        assert "హెక్సాకోనాజోల్" in result
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in result
        assert "సమీప దుకాణాలు అందుబాటులో లేవు" in result


@pytest.mark.asyncio
async def test_multi_intent_crop_advice_not_included_if_not_requested():
    """Verify that if farmer only asks for Weather + Shops, Crop Advice is NOT included."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా? యూరియా ఎక్కడ దొరుకుతుంది?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="నమస్తే! భూమిమిత్ర రైతులకు ఎల్లప్పుడూ సహాయం చేస్తుంది. పంటలకు సరైన సమయానికి నీరు ఇవ్వండి.",
        intent="general",
        confidence=0.8,
        provider_used="gemini"
    )

    from src.core.models import Shop, Inventory
    mock_shop = Shop(shop_name="శ్రీ బాలాజీ ఆగ్రో", status="active", delivery_available=True, phone_number="9848012345")
    mock_inventory = Inventory(product_name="యూరియా", brand="IFFCO", price=295.0, quantity_in_stock=40, unit="బస్తా")
    mock_search_results = [(mock_shop, mock_inventory)]

    from src.weather.schemas import WeatherForecastResponse
    mock_weather = WeatherForecastResponse.model_validate({
        "location_name": "Warangal",
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {"temp": 30.0, "feels_like": 32.0, "humidity": 70, "wind_speed": 10.0, "description": "Clear sky", "condition_code": 800},
        "forecast": [{"dt_txt": "2026-08-28 12:00:00", "temp": 30.0, "humidity": 70, "description": "Clear sky", "condition_code": 800}],
        "data_available": True,
        "is_live": False,
        "source_note": "Local Data"
    })

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=mock_weather):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "🌡️ *వాతావరణ సమాచారం*" in result
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in result
        assert "🌱 *పంట సలహా*" not in result
        assert "నమస్తే" not in result


@pytest.mark.asyncio
async def test_multi_intent_duplicate_scheme_mention_in_ai_text_stripped():
    """Verify that when AI text includes a brief scheme mention and structured schemes run, the AI mention is stripped."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="వరి పంటలో అగ్గితెగులు నివారణ ఏమిటి? రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="వరిలో అగ్గితెగులు నివారణకు ట్రైసైక్లాజోల్ 0.6 గ్రా/లీ నీటికి వాడండి. అలాగే ప్రభుత్వం పిఎం కిసాన్ పథకం ద్వారా రూ. 6000 సహాయం అందిస్తుంది.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    from src.core.models import GovernmentScheme
    mock_scheme = GovernmentScheme(
        scheme_name="పీఎం కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
        scheme_code="PM_KISAN",
        category="Direct Income Support",
        description="రైతులకు ఏటా ₹6000 ఆర్థిక సహాయం.",
        state="Telangana",
        benefits_summary="ఏడాదికి ₹6,000",
        eligibility_criteria="భూమి ఉన్న రైతులు",
        required_documents="ఆధార్, పట్టాదారు పాస్ బుక్",
        official_portal_url="https://pmkisan.gov.in"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=[mock_scheme]):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "🌱 *పంట సలహా*" in result
        assert "ట్రైసైక్లాజోల్ 0.6 గ్రా/లీ" in result
        assert "🏛️ *ప్రభుత్వ పథకాలు*" in result
        assert "పీఎం కిసాన్ సమ్మాన్ నిధి" in result


@pytest.mark.asyncio
async def test_single_intent_five_sample_queries_backward_compatibility():
    """
    Verify that single-intent queries for each domain remain concise, accurate,
    and retain their specialized content without being inappropriately rewritten.
    """
    from datetime import datetime
    from src.core.models import Shop, Inventory, GovernmentScheme, MarketPrice
    from src.weather.schemas import WeatherForecastResponse
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # 1. Shop single intent
    db_shop = AsyncMock()
    conv_shop = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="యూరియా ఎక్కడ దొరుకుతుంది?")
    mock_ai_shop = AIGenerateResponse(response_text="యూరియా సమీప డీలర్ల వద్ద అందుబాటులో ఉంది.", intent="shop_search", confidence=0.9, provider_used="gemini")
    mock_shop = Shop(shop_name="శ్రీ బాలాజీ ఆగ్రో", status="active", delivery_available=True, phone_number="9848012345")
    mock_inv = Inventory(product_name="యూరియా", brand="IFFCO", price=295.0, quantity_in_stock=40, unit="బస్తా")

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_shop), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=[(mock_shop, mock_inv)]):
        res_shop = await process_text_message(db_shop, farmer, conv_shop)
        assert "శ్రీ బాలాజీ ఆగ్రో" in res_shop
        assert "295" in res_shop

    # 2. Weather single intent
    db_weather = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_weather.execute = AsyncMock(side_effect=[mock_exec, mock_exec, MagicMock(), MagicMock()])

    conv_weather = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?")
    mock_ai_weather = AIGenerateResponse(response_text="వాతావరణ వివరాలు కింద ఇవ్వబడ్డాయి.", intent="weather", confidence=0.9, provider_used="gemini")
    mock_weather = WeatherForecastResponse.model_validate({
        "location_name": "Warangal",
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {"temp": 30.5, "feels_like": 33.0, "humidity": 72, "wind_speed": 11.0, "description": "పాక్షికంగా మేఘావృతమై ఉంది", "condition_code": 802},
        "forecast": [{"dt_txt": "2026-08-28 12:00:00", "temp": 29.0, "humidity": 78, "description": "వర్షం", "condition_code": 500}],
        "data_available": True,
        "is_live": True,
        "source_note": "OpenWeather (Live)"
    })
    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_weather), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=mock_weather):
        res_weather = await process_text_message(db_weather, farmer, conv_weather)
        assert "వాతావరణ సమాచారం" in res_weather or "Warangal" in res_weather
        assert "30.5" in res_weather

    # 3. Crop advice single intent
    db_crop = AsyncMock()
    conv_crop = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి?")
    mock_ai_crop = AIGenerateResponse(
        response_text="పత్తి పంటలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు లీటరు నీటికి కలిపి పిచికారీ చేయాలి.",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini"
    )
    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_crop):
        res_crop = await process_text_message(db_crop, farmer, conv_crop)
        assert "ఎమామెక్టిన్ బెంజోయేట్" in res_crop
        assert "0.4 గ్రాములు" in res_crop

    # 4. Schemes single intent
    db_scheme = AsyncMock()
    conv_scheme = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?")
    mock_ai_scheme = AIGenerateResponse(response_text="ప్రభుత్వ పథకాల వివరాలు కింద ఇవ్వబడ్డాయి.", intent="schemes", confidence=0.9, provider_used="gemini")
    mock_sch = GovernmentScheme(
        scheme_name="పీఎం కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
        scheme_code="PM_KISAN",
        category="Direct Income Support",
        description="రైతులకు ఏటా ₹6000 ఆర్థిక సహాయం.",
        state="Telangana",
        benefits_summary="ఏడాదికి ₹6,000",
        eligibility_criteria="భూమి ఉన్న రైతులు",
        required_documents="ఆధార్, పట్టాదారు పాస్ బుక్",
        official_portal_url="https://pmkisan.gov.in"
    )
    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_scheme), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=[mock_sch]):
        res_scheme = await process_text_message(db_scheme, farmer, conv_scheme)
        assert "పీఎం కిసాన్ సమ్మాన్ నిధి" in res_scheme

    # 5. Market single intent
    db_market = AsyncMock()
    conv_market = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి మార్కెట్ ధర ఎంత?")
    mock_ai_market = AIGenerateResponse(response_text="మార్కెట్ ధరల వివరాలు కింద ఇవ్వబడ్డాయి.", intent="market_price", confidence=0.9, provider_used="gemini")
    mock_mandi = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        state="Telangana",
        district="Warangal",
        market_name="Warangal Mandi",
        min_price=7100.0,
        max_price=7650.0,
        modal_price=7450.0,
        unit="Quintal",
        source="Agmarknet",
        price_date=datetime.utcnow(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_market), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=[mock_mandi]):
        res_market = await process_text_message(db_market, farmer, conv_market)
        assert "7450" in res_market or "7,450" in res_market


@pytest.mark.asyncio
async def test_whatsapp_reply_uses_gemini_36_flash_as_primary(monkeypatch):
    """
    Verify that the natural-language WhatsApp AI reply uses gemini-3.6-flash as the primary model.
    """
    import src.ai.gemini_client as gemini_module
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.core.models import Farmer, Conversation

    monkeypatch.setattr(gemini_module, "_initialized", True)

    models_called = []

    def mock_generative_model(model_name, **kwargs):
        models_called.append(model_name)
        mock_instance = MagicMock()
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "పత్తి పంటలో పురుగుల నివారణకు సరైన పురుగుమందు పిచికారీ చేయాలి."
        mock_chat.send_message.return_value = mock_resp
        mock_instance.start_chat.return_value = mock_chat
        return mock_instance

    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", mock_generative_model)

    db_mock = AsyncMock()
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, preferred_language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer_id, user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి?")

    ai_repo_mock = MagicMock()
    ai_repo_mock.session = db_mock
    ai_repo_mock.get_farmer_profile = AsyncMock(return_value=None)
    ai_repo_mock.get_conversation_history = AsyncMock(return_value=[])
    ai_repo_mock.get_recent_conversations = AsyncMock(return_value=[])

    ai_service = AIService(ai_repo_mock)
    req = AIGenerateRequest(farmer_id=farmer_id, message=conv.user_message)

    with patch("src.ai.service.build_farmer_context", return_value=""), \
         patch("src.memory.service.FarmerMemoryService.format_memory_for_system_prompt", new_callable=AsyncMock, return_value=""), \
         patch("src.rag.service.RAGService.search_knowledge", new_callable=AsyncMock, return_value=[]):
        ai_resp = await ai_service.generate_ai_response(req)

    assert models_called[0] == "gemini-3.6-flash"
    assert "పత్తి పంటలో పురుగుల నివారణకు" in ai_resp.response_text


@pytest.mark.asyncio
async def test_multi_intent_whatsapp_response_with_gemini_36_flash():
    """
    Verify multi-intent WhatsApp message assembly:
    Query: "పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది? వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?"
    Assembled:
    🌱 *పంట సలహా* (Gemini 3.5 Flash)
    🌡️ *వాతావరణ సమాచారం* (WeatherService authoritative)
    🏬 *సమీప వ్యవసాయ దుకాణాలు* (ShopService authoritative)
    """
    from src.core.models import Shop, Inventory
    from src.weather.schemas import WeatherForecastResponse

    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది? వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?"
    )

    mock_ai_resp = AIGenerateResponse(
        response_text="పత్తి పంటలో పురుగుల నివారణకు మోనోక్రోటోఫాస్ లేదా ఎమామెక్టిన్ బెంజోయేట్ వాడాలి.",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini"
    )
    mock_weather = WeatherForecastResponse.model_validate({
        "location_name": "Warangal",
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {"temp": 31.0, "feels_like": 34.0, "humidity": 70, "wind_speed": 10.0, "description": "మేఘావృతం", "condition_code": 802},
        "forecast": [{"dt_txt": "2026-08-28 12:00:00", "temp": 29.5, "humidity": 75, "description": "వర్షం", "condition_code": 500}],
        "data_available": True,
        "is_live": True,
        "source_note": "OpenWeather (Live)"
    })
    mock_shop = Shop(shop_name="రైతు మిత్ర ఆగ్రో సేవా కేంద్రం", status="active", delivery_available=True, phone_number="9876543210")
    mock_inv = Inventory(product_name="యూరియా", brand="KRIBHCO", price=266.5, quantity_in_stock=50, unit="బస్తా")

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai_resp), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=[(mock_shop, mock_inv)]), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=mock_weather):
        final_reply = await process_text_message(db_mock, farmer, conv)

    assert "🌱 *పంట సలహా*" in final_reply
    assert "మోనోక్రోటోఫాస్" in final_reply or "ఎమామెక్టిన్ బెంజోయేట్" in final_reply
    assert "🌡️ *వాతావరణ సమాచారం*" in final_reply
    assert "31.0" in final_reply or "వాతావరణం" in final_reply
    assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in final_reply
    assert "రైతు మిత్ర ఆగ్రో సేవా కేంద్రం" in final_reply
    assert "266.5" in final_reply
    # Ensure unrequested domains (market prices, government schemes) are excluded
    assert "📊 *మార్కెట్ ధరలు*" not in final_reply
    assert "🏛️ *ప్రభుత్వ పథకాలు*" not in final_reply

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
