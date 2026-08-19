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
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp: resp):
        
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
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp: resp):
        
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
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp: resp):
        
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
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp: resp):
        
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
         patch("src.shops.service.enrich_response_with_shops", side_effect=lambda db, msg, resp: resp):
        
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
        assert "Available Nearby Shops" not in result


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
        assert "Available Nearby Shops" not in result


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
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "You can buy Urea at local shops." in result
        assert "Available Nearby Shops" in result
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
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "You can buy Urea at nearby shops." in result
        assert "Available Nearby Shops" in result
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
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=mock_search_results):
        
        result = await process_text_message(db_mock, farmer, conversation)
        assert "మీరు స్థానిక దుకాణాలలో యూరియాను కొనుగోలు చేయవచ్చు." in result
        assert "Available Nearby Shops" in result
        assert "Mallanna Fertilizer" in result



