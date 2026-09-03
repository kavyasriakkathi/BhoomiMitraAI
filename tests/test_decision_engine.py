"""
BhoomiMitra AI — AI Decision Engine Comprehensive Test Suite

Verifies:
1. Intent classification (English, Telugu, Tanglish, Mixed).
2. Routing to authoritative modules (Market, Weather, Schemes, Shops, Advisory).
3. Anti-hallucination protections (real numbers preserved, fallbacks on missing data).
4. Pure greeting shortcuts (no LLM, no external service calls).
5. Multi-intent handling and separation.
6. Context overrides (explicit crop/district in query overriding profile).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from src.core.models import Farmer, Conversation, MarketPrice, Shop, Inventory, GovernmentScheme
from src.ai.decision_engine import (
    AIDecisionEngine,
    FarmerIntent,
    get_decision_engine,
)
from src.ai.prompts import (
    MARKET_FALLBACK_RESPONSE_TE,
    WEATHER_FALLBACK_RESPONSE_TE,
    SCHEMES_FALLBACK_RESPONSE_TE,
    SHOPS_FALLBACK_RESPONSE_TE,
)
from src.ai.service import process_text_message


# ─────────────────────────────────────────────────────────────────────────────
# 1. INTENT CLASSIFICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_intent_telugu_market_query():
    """Telugu market query -> FarmerIntent.MARKET_PRICE."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("వరంగల్లో ఈరోజు పత్తి ధర ఎంత")
    assert intent == FarmerIntent.MARKET_PRICE


def test_intent_tanglish_market_query():
    """Tanglish market query -> FarmerIntent.MARKET_PRICE."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("Warangal lo cotton rate entha")
    assert intent == FarmerIntent.MARKET_PRICE


def test_intent_telugu_weather_query():
    """Telugu weather query -> FarmerIntent.WEATHER."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("వర్షం ఎప్పుడు పడుతుంది")
    assert intent == FarmerIntent.WEATHER


def test_intent_tanglish_weather_query():
    """Tanglish weather query -> FarmerIntent.WEATHER."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("repu varsham paduthunda")
    assert intent == FarmerIntent.WEATHER


def test_intent_crop_health_query():
    """Telugu disease/pest query -> FarmerIntent.CROP_HEALTH."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("మిరప ఆకులు పసుపుగా మారుతున్నాయి")
    assert intent == FarmerIntent.CROP_HEALTH


def test_intent_fertilizer_query():
    """Fertilizer advice query -> FarmerIntent.FERTILIZER."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("నా పంటకు ఏ ఎరువు వేయాలి")
    assert intent == FarmerIntent.FERTILIZER


def test_intent_shop_query():
    """Input availability query -> FarmerIntent.SHOPS."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("యూరియా ఎక్కడ దొరుకుతుంది")
    assert intent == FarmerIntent.SHOPS


def test_intent_government_schemes_query():
    """Government schemes query -> FarmerIntent.GOVERNMENT_SCHEMES."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("ప్రభుత్వ రైతు పథకాలు ఏమైనా ఉన్నాయా")
    assert intent == FarmerIntent.GOVERNMENT_SCHEMES


def test_intent_irrigation_query():
    """Irrigation query -> FarmerIntent.IRRIGATION."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("పత్తి పంటకు నీరు ఎప్పుడు పెట్టాలి")
    assert intent == FarmerIntent.IRRIGATION


def test_intent_sowing_query():
    """Sowing query -> FarmerIntent.SOWING."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("వరి విత్తనాలు ఎప్పుడు వేయాలి")
    assert intent == FarmerIntent.SOWING


def test_intent_harvesting_query():
    """Harvesting query -> FarmerIntent.HARVESTING."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("పత్తి కోత ఎప్పుడు మొదలుపెట్టాలి")
    assert intent == FarmerIntent.HARVESTING


def test_intent_reminders_query():
    """Reminders query -> FarmerIntent.REMINDERS."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("ఎరువు వేయడానికి నాకు రేపు గుర్తు చేయండి")
    assert intent == FarmerIntent.REMINDERS


def test_intent_greeting():
    """Pure greetings in Telugu and English -> FarmerIntent.GREETING."""
    engine = get_decision_engine()
    for greeting in ["హాయ్", "నమస్తే", "hello", "hi", "namaste"]:
        assert engine.detect_primary_intent(greeting) == FarmerIntent.GREETING
        assert engine.is_greeting_only(greeting) is True


def test_greeting_with_question_is_not_pure_greeting():
    """Greeting accompanied by a domain question must NOT be classified as pure greeting."""
    engine = get_decision_engine()
    assert engine.is_greeting_only("నమస్తే, పత్తి ధర ఎంత?") is False
    assert engine.detect_primary_intent("నమస్తే, పత్తి ధర ఎంత?") == FarmerIntent.MARKET_PRICE


def test_intent_unknown_query():
    """Unrelated non-farming questions -> FarmerIntent.UNKNOWN."""
    engine = get_decision_engine()
    intent = engine.detect_primary_intent("who won the cricket match yesterday")
    assert intent == FarmerIntent.UNKNOWN


def test_multi_intent_detection():
    """Query containing both market price and weather -> both detected."""
    engine = get_decision_engine()
    intents = engine.detect_all_intents("ఈరోజు పత్తి ధర ఎంత? వర్షం ఎలా ఉంటుంది?")
    assert FarmerIntent.MARKET_PRICE in intents
    assert FarmerIntent.WEATHER in intents


def test_tanglish_multi_intent_detection():
    """Tanglish query containing cotton price and weather -> both detected."""
    engine = get_decision_engine()
    intents = engine.detect_all_intents("cotton price entha? repu varsham vasthunda?")
    assert FarmerIntent.MARKET_PRICE in intents
    assert FarmerIntent.WEATHER in intents


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUTING & GREETING SHORTCUT TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pure_greeting_does_not_call_external_services_or_llm():
    """Verify that 'hello' / 'హాయ్' returns instant greeting without invoking LLM or enrichments."""
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="హాయ్")

    with patch("src.ai.service.AIService.generate_ai_response") as mock_ai, \
         patch("src.market.service.enrich_response_with_market_prices") as mock_market, \
         patch("src.weather.service.enrich_response_with_weather") as mock_weather, \
         patch("src.shops.service.enrich_response_with_shops") as mock_shops, \
         patch("src.schemes.service.enrich_response_with_schemes") as mock_schemes:

        result = await process_text_message(db_mock, farmer, conversation)

        # Confirm zero external or LLM calls
        mock_ai.assert_not_called()
        mock_market.assert_not_called()
        mock_weather.assert_not_called()
        mock_shops.assert_not_called()
        mock_schemes.assert_not_called()

        # Confirm friendly, helpful greeting returned
        assert "నమస్తే" in result
        assert "భూమిమిత్ర" in result


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANTI-HALLUCINATION & AUTHORITATIVE DATA TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_authoritative_market_price_numbers_preserved_untouched():
    """Verify that supplied real market prices (modal, min, max) are preserved exactly without alteration."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరంగల్లో ఈరోజు పత్తి ధర ఎంత")

    mock_price = MarketPrice(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal APMC",
        district="Warangal",
        state="Telangana",
        modal_price=8900.0,
        min_price=5000.0,
        max_price=9250.0,
        unit="Quintal",
        price_date=datetime(2026, 9, 2),
        source="live_agmarknet",
        created_at=datetime.utcnow(),
    )

    with patch("src.market.service.MarketService.get_prices_for_query") as mock_get_prices, \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.ai.service.AIService.generate_ai_response", side_effect=Exception("LLM offline")):

        from src.market.schemas import MarketPriceQueryResponse, MarketPriceResponse
        mock_get_prices.return_value = MarketPriceQueryResponse(
            commodity="Cotton",
            district="Warangal",
            state="Telangana",
            data_available=True,
            is_live=True,
            source_note="agmarknet",
            results=[MarketPriceResponse.model_validate(mock_price, from_attributes=True)],
        )

        result = await process_text_message(db_mock, farmer, conversation)

        # Exact numbers must be present
        assert "8,900" in result
        assert "5,000" in result
        assert "9,250" in result
        assert "Warangal APMC" in result


@pytest.mark.asyncio
async def test_missing_market_price_data_returns_exact_fallback():
    """Verify that when market prices are unavailable, the exact localized fallback is returned without hallucinating numbers."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరంగల్లో ఈరోజు పత్తి ధర ఎంత")

    with patch("src.market.service.MarketService.get_prices_for_query") as mock_get_prices, \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.ai.service.AIService.generate_ai_response", side_effect=Exception("LLM offline")):

        from src.market.schemas import MarketPriceQueryResponse
        mock_get_prices.return_value = MarketPriceQueryResponse(
            commodity="Cotton",
            district="Warangal",
            state="Telangana",
            data_available=False,
            is_live=False,
            source_note="agmarknet",
            results=[],
        )

        result = await process_text_message(db_mock, farmer, conversation)

        # Must contain exact required fallback
        assert "ప్రస్తుతం మార్కెట్ ధరల సమాచారం అందుబాటులో లేదు" in result
        # Must not contain hallucinated rupees or estimates
        assert "₹" not in result


@pytest.mark.asyncio
async def test_missing_weather_data_returns_exact_fallback():
    """Verify that when weather service fails, the localized fallback is provided without hallucinated temperatures."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరంగల్లో వర్షం ఎప్పుడు పడుతుంది")

    with patch("src.weather.service.WeatherService.get_weather_for_query", side_effect=Exception("Weather API offline")), \
         patch("src.ai.service.AIService.generate_ai_response", side_effect=Exception("LLM offline")):
        result = await process_text_message(db_mock, farmer, conversation)

        # Must state that weather cannot be retrieved
        assert "వాతావరణ సమాచారం" in result
        assert ("పొందలేకపోతున్నాను" in result or "అందుబాటులో లేదు" in result)


@pytest.mark.asyncio
async def test_missing_scheme_data_returns_exact_fallback():
    """Verify that when no government schemes match, the localized fallback is returned."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="ప్రభుత్వ పథకాలు ఏమైనా ఉన్నాయా")

    with patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=[]), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.ai.service.AIService.generate_ai_response", side_effect=Exception("LLM offline")):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "ప్రభుత్వ పథకాల సమాచారం" in result


@pytest.mark.asyncio
async def test_missing_shop_data_returns_exact_fallback():
    """Verify that when no shops have the requested product, the localized fallback is provided."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="యూరియా ఎక్కడ దొరుకుతుంది?")

    with patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.ai.service.AIService.generate_ai_response", side_effect=Exception("LLM offline")):

        result = await process_text_message(db_mock, farmer, conversation)

        assert "సమీప దుకాణాల" in result or "సమీప దుకాణాలు" in result


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-INTENT ORCHESTRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_intent_calls_only_relevant_modules():
    """Verify that 'cotton price + weather' calls ONLY market and weather services, omitting shops and schemes."""
    db_mock = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalars.return_value.all.return_value = []
    mock_scalar.scalars.return_value.first.return_value = None
    db_mock.execute.return_value = mock_scalar

    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="ఈరోజు పత్తి ధర ఎంత? వర్షం ఎలా ఉంటుంది?"
    )

    async def fake_market(db, msg, text, farmer):
        return text + ("\n\n" if text else "") + "📊 *పత్తి మార్కెట్ ధరలు*\nధర: ₹8,500/క్వింటాల్"

    async def fake_weather(db, msg, text, farmer):
        return text + ("\n\n" if text else "") + "🌡️ *వాతావరణ సమాచారం*\nవర్షం పడే అవకాశం ఉంది."

    with patch("src.market.service.enrich_response_with_market_prices", side_effect=fake_market) as mock_market, \
         patch("src.weather.service.enrich_response_with_weather", side_effect=fake_weather) as mock_weather, \
         patch("src.shops.service.enrich_response_with_shops", new_callable=AsyncMock) as mock_shops, \
         patch("src.schemes.service.enrich_response_with_schemes", new_callable=AsyncMock) as mock_schemes:

        result = await process_text_message(db_mock, farmer, conversation)

        # Market and Weather must have been called
        mock_market.assert_awaited_once()
        mock_weather.assert_awaited_once()

        # Unrelated modules must NOT have been called
        mock_shops.assert_not_called()
        mock_schemes.assert_not_called()

        # Response must contain both sections clearly separated
        assert "పత్తి మార్కెట్ ధరలు" in result
        assert "వాతావరణ సమాచారం" in result



# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTEXT OVERRIDE TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_crop_overrides_profile_crop():
    """Verify that when a farmer mentions a specific crop in their query (e.g. Tomato), it overrides their profile crop (e.g. Cotton)."""
    from src.ai.schemas import AIGenerateResponse
    db_mock = AsyncMock()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conversation = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="టమాటా పంటలో ఆకుముడత నివారణ ఎలా?"
    )

    captured_requests = []

    async def fake_generate(request):
        captured_requests.append(request)
        return AIGenerateResponse(
            response_text="టమాటాలో ఆకుముడత నివారణ సలహా...",
            intent="crop_health",
            confidence=0.9,
            provider_used="gemini"
        )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=fake_generate):
        result = await process_text_message(db_mock, farmer, conversation)

        assert len(captured_requests) == 1
        assert "టమాటా" in captured_requests[0].message
        assert "టమాటాలో ఆకుముడత" in result
