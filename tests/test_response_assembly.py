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
