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


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPREHENSIVE 13-LANGUAGE INTENT ROUTING TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected_intent",
    [
        # 1. Market Price (13 languages)
        ("వరంగల్లో ఈరోజు పత్తి ధర ఎంత", FarmerIntent.MARKET_PRICE),
        ("कपास का मंडी भाव कितना है", FarmerIntent.MARKET_PRICE),
        ("What is the market price of cotton in Warangal", FarmerIntent.MARKET_PRICE),
        ("பருத்தி சந்தை விலை என்ன", FarmerIntent.MARKET_PRICE),
        ("ಹತ್ತಿ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಎಷ್ಟು", FarmerIntent.MARKET_PRICE),
        ("പരുത്തി വിപണി വില എത്ര", FarmerIntent.MARKET_PRICE),
        ("कापसाचा बाजारभाव किती आहे", FarmerIntent.MARKET_PRICE),
        ("তুলার বাজার দর কত", FarmerIntent.MARKET_PRICE),
        ("કપાસનો બજાર ભાવ કેટલો છે", FarmerIntent.MARKET_PRICE),
        ("ਕਪਾਹ ਦਾ ਮੰਡੀ ਭਾਅ ਕੀ ਹੈ", FarmerIntent.MARKET_PRICE),
        ("କପା ମଣ୍ଡି ଦର କେତେ", FarmerIntent.MARKET_PRICE),
        ("কপাহৰ বজাৰ দৰ কিমান", FarmerIntent.MARKET_PRICE),
        ("کپاس کا منڈی ریٹ کیا ہے", FarmerIntent.MARKET_PRICE),

        # 2. Weather (13 languages)
        ("రేపు వర్షం పడుతుందా", FarmerIntent.WEATHER),
        ("कल बारिश होगी क्या मौसम कैसा रहेगा", FarmerIntent.WEATHER),
        ("Will it rain tomorrow in Warangal", FarmerIntent.WEATHER),
        ("நாளை மழை பெய்யுமா வானிலை எப்படி", FarmerIntent.WEATHER),
        ("ನಾಳೆ ಮಳೆ ಬರುತ್ತಾ ಹವಾಮಾನ ಹೇಗೆ", FarmerIntent.WEATHER),
        ("നാളെ മഴ പെയ്യുമോ കാലാവസ്ഥ പ്രവചനം", FarmerIntent.WEATHER),
        ("उद्या पाऊस पडेल का हवामान अंदाज", FarmerIntent.WEATHER),
        ("কাল কি বৃষ্টি হবে আবহাওয়া কেমন", FarmerIntent.WEATHER),
        ("કાલે વરસાદ પડશે કે નહિ હવામાન", FarmerIntent.WEATHER),
        ("ਕੱਲ੍ਹ ਮੀਂਹ ਪਵੇਗਾ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਰਹੇਗਾ", FarmerIntent.WEATHER),
        ("କାଲି ବର୍ଷା ହେବ କି ପାଣିପାଗ ସୂଚନା", FarmerIntent.WEATHER),
        ("কাইলৈ বৰষুণ হ'ব নেকি বতৰৰ খবৰ", FarmerIntent.WEATHER),
        ("کل بارش ہوگی کیا موسم کا حال بتائیں", FarmerIntent.WEATHER),

        # 3. Government Schemes (13 languages)
        ("రైతు భరోసా ప్రభుత్వ పథకం వివరాలు", FarmerIntent.GOVERNMENT_SCHEMES),
        ("पीएम किसान सम्मान निधि सरकारी योजना की जानकारी", FarmerIntent.GOVERNMENT_SCHEMES),
        ("Tell me about PM Kisan government subsidy scheme", FarmerIntent.GOVERNMENT_SCHEMES),
        ("விவசாய மானியம் அரசு திட்டம் விவரங்கள்", FarmerIntent.GOVERNMENT_SCHEMES),
        ("ಕಿಸಾನ್ ಸಬ್ಸಿಡಿ ಸರ್ಕಾರಿ ಯೋಜನೆ ವಿವರ", FarmerIntent.GOVERNMENT_SCHEMES),
        ("വിള ഇൻഷുറൻസ് സർക്കാർ പദ്ധതി", FarmerIntent.GOVERNMENT_SCHEMES),
        ("पीक विमा शेतकरी शासकीय योजना माहिती", FarmerIntent.GOVERNMENT_SCHEMES),
        ("কৃষক বন্ধু সরকারি অনুদান প্রকল্প", FarmerIntent.GOVERNMENT_SCHEMES),
        ("ખેડૂત સબસિડી સરકારી યોજના ની માહિતી", FarmerIntent.GOVERNMENT_SCHEMES),
        ("ਫ਼ਸਲ ਬੀਮਾ ਸਰਕਾਰੀ ਸਹਾਇਤਾ ਸਕੀਮ", FarmerIntent.GOVERNMENT_SCHEMES),
        ("କାଳିଆ ଯୋଜନା ସରକାରୀ ସହାୟତା ବିବରଣୀ", FarmerIntent.GOVERNMENT_SCHEMES),
        ("শস্য বীমা চৰকাৰী সাহায্য আঁচনি", FarmerIntent.GOVERNMENT_SCHEMES),
        ("پی ایم کسان سرکاری اسکیم کی تفصیلات", FarmerIntent.GOVERNMENT_SCHEMES),

        # 4. Shops / Input Availability (13 languages)
        ("యూరియా ఎరువుల దుకాణం ఎక్కడ ఉంది", FarmerIntent.SHOPS),
        ("यूरिया खाद की दुकान कहाँ मिलेगी", FarmerIntent.SHOPS),
        ("Where can I buy urea fertilizer from nearby shops", FarmerIntent.SHOPS),
        ("உரக்கடை எங்கு உள்ளது மருந்து வாங்க", FarmerIntent.SHOPS),
        ("ಗೊಬ್ಬರದ ಅಂಗಡಿ ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ", FarmerIntent.SHOPS),
        ("വളക്കട എവിടെ ലഭിക്കും സമീപത്തെ കട", FarmerIntent.SHOPS),
        ("खताचे दुकान कुठे मिळेल औषध खरेदी", FarmerIntent.SHOPS),
        ("সারের দোকান কোথায় পাওয়া যাবে", FarmerIntent.SHOPS),
        ("ખાતરની દુકાન ક્યાં મળશે એગ્રો સેન્ટર", FarmerIntent.SHOPS),
        ("ਖਾਦ ਦੀ ਦੁਕਾਨ ਕਿੱਥੇ ਮਿਲੇਗੀ ਡੀਲਰ", FarmerIntent.SHOPS),
        ("ସାର ଦୋକାନ କେଉଁଠି ମିଳିବ କିଣିବା ପାଇଁ", FarmerIntent.SHOPS),
        ("সাৰৰ দোকান ক'ত পোৱা যাব", FarmerIntent.SHOPS),
        ("کھاد کی دکان کہاں ملے گی قریبی ڈیلر", FarmerIntent.SHOPS),

        # 5. Crop Health / Pest / Disease (13 languages)
        ("మిరప పంటలో ఆకుముడత తెగులు నివారణ", FarmerIntent.CROP_HEALTH),
        ("कपास में गुलाबी सुंडी कीट रोकथाम", FarmerIntent.CROP_HEALTH),
        ("How to control bollworm pest and blight disease", FarmerIntent.CROP_HEALTH),
        ("பருத்தி பயிரில் புழு நோய் கட்டுப்பாடு", FarmerIntent.CROP_HEALTH),
        ("ಬೆಳೆಯಲ್ಲಿ ಕೀಟ ರೋಗ ಬಾಧೆ ನಿಯಂತ್ರಣ", FarmerIntent.CROP_HEALTH),
        ("വിളകളിൽ പുഴു രോഗം കീടനിയന്ത്രണം", FarmerIntent.CROP_HEALTH),
        ("कापूस पिकावर बोंडअळी रोग नियंत्रण फवारणी", FarmerIntent.CROP_HEALTH),
        ("তুলা ফসলে পোকা রোগ দমন স্প্রে", FarmerIntent.CROP_HEALTH),
        ("કપાસમાં ગુલાબી ઈયળ જીવાત રોગ નિયંત્રણ", FarmerIntent.CROP_HEALTH),
        ("ਫ਼ਸਲ ਵਿੱਚ ਸੁੰਡੀ ਬਿਮਾਰੀ ਕੀਟਨਾਸ਼ਕ ਸਪਰੇਅ", FarmerIntent.CROP_HEALTH),
        ("ଫସଲରେ ପୋକ ରୋଗ ନିୟନ୍ତ୍ରଣ କୀଟନାଶକ", FarmerIntent.CROP_HEALTH),
        ("খেতিত পোক পৰুৱা ৰোগ নিয়ন্ত্ৰণ", FarmerIntent.CROP_HEALTH),
        ("فصل میں کیڑے اور بیماری کی روکتھام", FarmerIntent.CROP_HEALTH),

        # 6. Fertilizer / Nutrients (13 languages)
        ("వరి పంటకు ఏ ఎరువుల మోతాదు వేయాలి", FarmerIntent.FERTILIZER),
        ("गेहूं में यूरिया खाद कितनी मात्रा में डालें", FarmerIntent.FERTILIZER),
        ("What is the recommended NPK fertilizer schedule", FarmerIntent.FERTILIZER),
        ("நெல் பயிருக்கு எவ்வளவு உரம் இட வேண்டும்", FarmerIntent.FERTILIZER),
        ("ಭತ್ತದ ಬೆಳೆಗೆ ಎಷ್ಟು ರಸಗೊಬ್ಬರ ಪ್ರಮಾಣ ಹಾಕಬೇಕು", FarmerIntent.FERTILIZER),
        ("നെല്ലിന് എത്ര രാസവളം പ്രയോഗിക്കണം", FarmerIntent.FERTILIZER),
        ("कापूस पिकाला किती रासायनिक खत द्यावे", FarmerIntent.FERTILIZER),
        ("ধান চাষে কতটা রাসায়নিক সার প্রয়োগ করব", FarmerIntent.FERTILIZER),
        ("કપાસમાં કેટલું રાસાયણિક ખાતર આપવું", FarmerIntent.FERTILIZER),
        ("ਕਣਕ ਨੂੰ ਕਿੰਨੀ ਰਸਾਇਣਕ ਖਾਦ ਪਾਈਏ", FarmerIntent.FERTILIZER),
        ("ଧାନ ଫସଲରେ କେତେ ରାସାୟନିକ ସାର ପ୍ରୟୋଗ କରିବି", FarmerIntent.FERTILIZER),
        ("ধান খেতিত কিমান ৰাসায়নিক সাৰ প্ৰয়োগ কৰিম", FarmerIntent.FERTILIZER),
        ("گندم کی فصل میں کتنی کھاد استعمال کریں", FarmerIntent.FERTILIZER),

        # 7. Irrigation / Water (13 languages)
        ("పత్తి పంటకు నీరు ఎప్పుడు పెట్టాలి", FarmerIntent.IRRIGATION),
        ("कपास में ड्रिप सिंचाई से पानी कब दें", FarmerIntent.IRRIGATION),
        ("When to provide drip irrigation watering", FarmerIntent.IRRIGATION),
        ("பருத்தி பயிருக்கு சொட்டு நீர்ப்பாசனம் எப்போது", FarmerIntent.IRRIGATION),
        ("ಹತ್ತಿ ಬೆಳೆಗೆ ಹನಿ ನೀರಾವರಿ ಯಾವಾಗ ನೀಡಬೇಕು", FarmerIntent.IRRIGATION),
        ("തുള്ളി നന ജലസേചനം എപ്പോൾ നൽകണം", FarmerIntent.IRRIGATION),
        ("कापूस पिकाला ठिबक सिंचन पाणी कधी द्यावे", FarmerIntent.IRRIGATION),
        ("ড্রিপ সেচ দিয়ে কখন পানি দিতে হবে", FarmerIntent.IRRIGATION),
        ("કપાસમાં ટપક પદ્ધતિ થી ક્યારે પાણી આપવું", FarmerIntent.IRRIGATION),
        ("ਫ਼ਸਲ ਨੂੰ ਤੁਪਕਾ ਸਿੰਚਾਈ ਨਾਲ ਕਦੋਂ ਪਾਣੀ ਲਾਈਏ", FarmerIntent.IRRIGATION),
        ("ବୁନ୍ଦା ଜଳସେଚନ ମାଧ୍ୟମରେ କେବେ ପାଣି ଦେବି", FarmerIntent.IRRIGATION),
        ("টোপাল জলসিঞ্চন কেতিয়া পানী দিব লাগে", FarmerIntent.IRRIGATION),
        ("فصل میں ڈرپ آبپاشی سے کب پانی دیں", FarmerIntent.IRRIGATION),

        # 8. Sowing / Planting (13 languages)
        ("విత్తనాల విత్తే సమయం మరియు నాట్లు", FarmerIntent.SOWING),
        ("कपास की बुवाई का समय और बीज उपचार", FarmerIntent.SOWING),
        ("What is the best sowing time and seed rate", FarmerIntent.SOWING),
        ("விதைப்பு நேரம் மற்றும் விதை நேர்த்தி", FarmerIntent.SOWING),
        ("ಬಿತ್ತನೆ ಸಮಯ ಮತ್ತು ಬೀಜೋಪಚಾರ ಮಾಹಿತಿ", FarmerIntent.SOWING),
        ("വിത്ത് വിതയ്ക്കൽ സമയം ഞാറുനടീൽ", FarmerIntent.SOWING),
        ("पेरणीची वेळ आणि बीजप्रक्रिया कशी करावी", FarmerIntent.SOWING),
        ("বীজ বপনের সময় ও চারা রোপণ পদ্ধতি", FarmerIntent.SOWING),
        ("વાવણીનો સમય અને બીજ માવજત પદ્ધતિ", FarmerIntent.SOWING),
        ("ਬਿਜਾਈ ਦਾ ਸਮਾਂ ਅਤੇ ਬੀਜ ਸੋਧ ਜਾਣਕਾਰੀ", FarmerIntent.SOWING),
        ("ବୁଣିବା ସମୟ ଏବଂ ବିହନ ବିଶୋଧନ", FarmerIntent.SOWING),
        ("বীজ সিঁচা আৰু পুলি ৰোপণৰ সময়", FarmerIntent.SOWING),
        ("فصل کی بوائی کا وقت اور بیج کا علاج", FarmerIntent.SOWING),

        # 9. Harvesting / Storage (13 languages)
        ("పత్తి కోత సమయం మరియు కోయడం", FarmerIntent.HARVESTING),
        ("कपास की फसल कटाई का समय और तुड़ाई", FarmerIntent.HARVESTING),
        ("When is the optimal harvest time and picking", FarmerIntent.HARVESTING),
        ("பருத்தி அறுவடை காலம் எப்போது பறித்தல்", FarmerIntent.HARVESTING),
        ("ಹತ್ತಿ ಕಟಾವಿನ ಸಮಯ ಮತ್ತು ಸುಗ್ಗಿ", FarmerIntent.HARVESTING),
        ("വിളവെടുപ്പ് സമയം എപ്പോൾ കൊയ്യണം", FarmerIntent.HARVESTING),
        ("कापूस वेचणी आणि काढणीची वेळ", FarmerIntent.HARVESTING),
        ("ফসল কাটার সময় ও মাড়াই পদ্ধতি", FarmerIntent.HARVESTING),
        ("કપાસની વીણવું અને કાપણીનો સમય", FarmerIntent.HARVESTING),
        ("ਕਣਕ ਦੀ ਵਾਢੀ ਦਾ ਸਮਾਂ ਅਤੇ ਕਟਾਈ", FarmerIntent.HARVESTING),
        ("ଫସଲ କାଟିବା ସମୟ ଏବଂ ଅମଳ", FarmerIntent.HARVESTING),
        ("শস্য চপোৱাৰ সময় আৰু কটাৰ সময়", FarmerIntent.HARVESTING),
        ("فصل کی کٹائی کا وقت اور چنائی", FarmerIntent.HARVESTING),

        # 10. Reminders (13 languages)
        ("నాకు స్ప్రే గుర్తు చేయండి", FarmerIntent.REMINDERS),
        ("मुझे खाद डालने की याद दिलाना", FarmerIntent.REMINDERS),
        ("Please set reminder for irrigation", FarmerIntent.REMINDERS),
        ("உரம் போட நினைவூட்டல் அமை", FarmerIntent.REMINDERS),
        ("ಔಷಧ ಸಿಂಪಡಿಸಲು ನೆನಪಿಸಿ", FarmerIntent.REMINDERS),
        ("വളം പ്രയോഗിക്കാൻ ഓർമ്മിപ്പിക്കുക", FarmerIntent.REMINDERS),
        ("खत देण्याची आठवण करा", FarmerIntent.REMINDERS),
        ("সার দেওয়ার কথা মনে করিয়ে দাও", FarmerIntent.REMINDERS),
        ("ખાતર આપવાનું યાદ અપાવો", FarmerIntent.REMINDERS),
        ("ਸਪਰੇਅ ਕਰਨ ਦੀ ਯਾਦ ਕਰਵਾਓ", FarmerIntent.REMINDERS),
        ("ସାର ପ୍ରୟୋଗ ମନେ ପକାଇ ଦିଅନ୍ତୁ", FarmerIntent.REMINDERS),
        ("সাৰ দিয়াৰ কথা মনত পেলাই দিব", FarmerIntent.REMINDERS),
        ("اسپرے کرنے کی یاد دلائیں", FarmerIntent.REMINDERS),

        # 11. General Farming / Advisory (13 languages)
        ("మా ఊరి భూమి నేల సారవంతం", FarmerIntent.GENERAL_FARMING),
        ("किसान की जमीन और मिट्टी की पैदावार", FarmerIntent.GENERAL_FARMING),
        ("General farming practices for my agricultural field", FarmerIntent.GENERAL_FARMING),
        ("விவசாயி நிலம் மற்றும் மண் வளம்", FarmerIntent.GENERAL_FARMING),
        ("ರೈತ ಜಮೀನು ಮತ್ತು ಕೃಷಿ ಪದ್ಧತಿ", FarmerIntent.GENERAL_FARMING),
        ("കർഷകൻ കൃഷി നിലം മണ്ണ്", FarmerIntent.GENERAL_FARMING),
        ("शेतकरी जमीन आणि शेती उत्पादन", FarmerIntent.GENERAL_FARMING),
        ("কৃষক জমি এবং চাষাবাদ পদ্ধতি", FarmerIntent.GENERAL_FARMING),
        ("ખેડૂત જમીન અને ખેતી ઉત્પાદન", FarmerIntent.GENERAL_FARMING),
        ("ਕਿਸਾਨ ਜ਼ਮੀਨ ਅਤੇ ਖੇਤੀ ਝਾੜ", FarmerIntent.GENERAL_FARMING),
        ("କୃଷକ ଜମି ଏବଂ ଚାଷ ପଦ୍ଧତି", FarmerIntent.GENERAL_FARMING),
        ("কৃষক মাটি আৰু খেতি পদ্ধতি", FarmerIntent.GENERAL_FARMING),
        ("کسان کی زمین اور کھیتی باڑی", FarmerIntent.GENERAL_FARMING),

        # 12. Greetings (13 languages)
        ("నమస్తే భూమిమిత్ర", FarmerIntent.GREETING),
        ("नमस्ते भूमिमित्र", FarmerIntent.GREETING),
        ("Hello BhoomiMitra", FarmerIntent.GREETING),
        ("வணக்கம் பூமிமித்ரா", FarmerIntent.GREETING),
        ("ನಮಸ್ಕಾರ ಭೂಮಿಮಿತ್ರ", FarmerIntent.GREETING),
        ("നമസ്കാരം ഭൂമിമിത്ര", FarmerIntent.GREETING),
        ("नमस्कार भूमिमित्र", FarmerIntent.GREETING),
        ("নমস্কার ভূমিমিত্র", FarmerIntent.GREETING),
        ("નમસ્તે ભૂમિમિત્ર", FarmerIntent.GREETING),
        ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਭੂਮੀਮਿੱਤਰ", FarmerIntent.GREETING),
        ("ନମସ୍କାର ଭୂମିମିତ୍ର", FarmerIntent.GREETING),
        ("নমস্কাৰ ভূমিমিত্ৰ", FarmerIntent.GREETING),
        ("سلام بھومی مترا", FarmerIntent.GREETING),

        # 13. Unknown queries
        ("12345 67890 xyz abc", FarmerIntent.UNKNOWN),
        ("what is quantum computing", FarmerIntent.UNKNOWN),
    ]
)
def test_all_13_languages_intent_classification(query, expected_intent):
    """Verify that intent detection works reliably across all 13 supported languages."""
    engine = get_decision_engine()
    detected = engine.detect_primary_intent(query)
    assert detected == expected_intent, f"Failed for query '{query}': expected {expected_intent}, got {detected}"


def test_multilingual_multi_intent_detection():
    """Verify multi-intent classification across different languages."""
    engine = get_decision_engine()

    # Hindi: Market + Weather
    hi_intents = engine.detect_all_intents("कपास का मंडी भाव कितना है और कल बारिश होगी क्या?")
    assert FarmerIntent.MARKET_PRICE in hi_intents
    assert FarmerIntent.WEATHER in hi_intents

    # Tamil: Fertilizer + Irrigation
    ta_intents = engine.detect_all_intents("நெல் பயிருக்கு எவ்வளவு உரம் இட வேண்டும் மற்றும் சொட்டு நீர்ப்பாசனம் எப்போது?")
    assert FarmerIntent.FERTILIZER in ta_intents
    assert FarmerIntent.IRRIGATION in ta_intents

    # Marathi: Crop Health + Shops
    mr_intents = engine.detect_all_intents("कापूस पिकावर बोंडअळी रोग नियंत्रण फवारणी औषध खताचे दुकान कुठे मिळेल?")
    assert FarmerIntent.CROP_HEALTH in mr_intents
    assert FarmerIntent.SHOPS in mr_intents

    # Bengali: Sowing + Schemes
    bn_intents = engine.detect_all_intents("বীজ বপনের সময় এবং কৃষক বন্ধু সরকারি অনুদান প্রকল্প")
    assert FarmerIntent.SOWING in bn_intents
    assert FarmerIntent.GOVERNMENT_SCHEMES in bn_intents


def test_hindi_vs_marathi_intent_routing_regression():
    """Verify Hindi and Marathi specific agricultural queries route cleanly."""
    engine = get_decision_engine()

    # Marathi specific terms
    assert engine.detect_primary_intent("कापसाचा बाजारभाव किती आहे") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("कापूस पिकाला ठिबक सिंचन पाणी कधी द्यावे") == FarmerIntent.IRRIGATION
    assert engine.detect_primary_intent("पेरणीची वेळ आणि बीजप्रक्रिया कशी करावी") == FarmerIntent.SOWING
    assert engine.detect_primary_intent("कापूस वेचणी आणि काढणीची वेळ") == FarmerIntent.HARVESTING
    assert engine.detect_primary_intent("खताचे दुकान कुठे मिळेल") == FarmerIntent.SHOPS

    # Hindi specific terms
    assert engine.detect_primary_intent("कपास का मंडी भाव कितना है") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("कपास में ड्रिप सिंचाई से पानी कब दें") == FarmerIntent.IRRIGATION
    assert engine.detect_primary_intent("कपास की बुवाई का समय और बीज उपचार") == FarmerIntent.SOWING
    assert engine.detect_primary_intent("कपास की फसल कटाई का समय और तुड़ाई") == FarmerIntent.HARVESTING
    assert engine.detect_primary_intent("यूरिया खाद की दुकान कहाँ मिलेगी") == FarmerIntent.SHOPS


def test_bengali_vs_assamese_intent_routing_regression():
    """Verify Bengali and Assamese specific agricultural queries route cleanly."""
    engine = get_decision_engine()

    # Assamese specific terms
    assert engine.detect_primary_intent("কপাহৰ বজাৰ দৰ কিমান") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("কাইলৈ বৰষুণ হ'ব নেকি বতৰৰ খবৰ") == FarmerIntent.WEATHER
    assert engine.detect_primary_intent("শস্য বীমা চৰকাৰী সাহায্য আঁচনি") == FarmerIntent.GOVERNMENT_SCHEMES
    assert engine.detect_primary_intent("টোপাল জলসিঞ্চন কেতিয়া পানী দিব লাগে") == FarmerIntent.IRRIGATION
    assert engine.detect_primary_intent("বীজ সিঁচা আৰু পুলি ৰোপণৰ সময়") == FarmerIntent.SOWING

    # Bengali specific terms
    assert engine.detect_primary_intent("তুলার বাজার দর কত") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("কাল কি বৃষ্টি হবে আবহাওয়া কেমন") == FarmerIntent.WEATHER
    assert engine.detect_primary_intent("কৃষক বন্ধু সরকারি অনুদান প্রকল্প") == FarmerIntent.GOVERNMENT_SCHEMES
    assert engine.detect_primary_intent("ড্রিপ সেচ দিয়ে কখন পানি দিতে হবে") == FarmerIntent.IRRIGATION
    assert engine.detect_primary_intent("বীজ বপনের সময় ও চারা রোপণ পদ্ধতি") == FarmerIntent.SOWING


def test_telugu_and_tanglish_intent_routing_regression():
    """Verify Telugu native and Tanglish romanized queries route reliably without degradation."""
    engine = get_decision_engine()

    # Telugu native
    assert engine.detect_primary_intent("వరంగల్లో ఈరోజు పత్తి ధర ఎంత") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("రేపు వర్షం పడుతుందా") == FarmerIntent.WEATHER
    assert engine.detect_primary_intent("రైతు భరోసా పథకం అర్హత ఏమిటి") == FarmerIntent.GOVERNMENT_SCHEMES
    assert engine.detect_primary_intent("పత్తి పంటలో ఆకుముడత నివారణ") == FarmerIntent.CROP_HEALTH
    assert engine.detect_primary_intent("వరి పంటకు ఏ ఎరువు వేయాలి") == FarmerIntent.FERTILIZER
    assert engine.detect_primary_intent("విత్తనాల విత్తే సమయం ఎప్పుడు") == FarmerIntent.SOWING
    assert engine.detect_primary_intent("పత్తి కోత సమయం ఎప్పుడు") == FarmerIntent.HARVESTING
    assert engine.detect_primary_intent("నాకు స్ప్రే గుర్తు చేయండి") == FarmerIntent.REMINDERS

    # Tanglish romanized
    assert engine.detect_primary_intent("cotton market rate entha undi") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("repu varsham paduthunda") == FarmerIntent.WEATHER
    assert engine.detect_primary_intent("rythu bharosa pathakam details") == FarmerIntent.GOVERNMENT_SCHEMES
    assert engine.detect_primary_intent("purugula mandu spray cheyali") == FarmerIntent.CROP_HEALTH
    assert engine.detect_primary_intent("urea fertilizer dose entha") == FarmerIntent.FERTILIZER
    assert engine.detect_primary_intent("neeru eppudu pettali") == FarmerIntent.IRRIGATION
    assert engine.detect_primary_intent("vithanala vithadam time") == FarmerIntent.SOWING
    assert engine.detect_primary_intent("panta kotha time eppudu") == FarmerIntent.HARVESTING
    assert engine.detect_primary_intent("remind cheyandi irrigation kosam") == FarmerIntent.REMINDERS


def test_mixed_language_intent_routing():
    """Verify mixed-language queries (English words embedded in Indic context) route correctly."""
    engine = get_decision_engine()

    assert engine.detect_primary_intent("Cotton crop lo leaf spot ki ఏ మందు వాడాలి?") == FarmerIntent.CROP_HEALTH
    assert engine.detect_primary_intent("Warangal mandi lo cotton rate ఎంత ఉంది?") == FarmerIntent.MARKET_PRICE
    assert engine.detect_primary_intent("Urea fertilizer kitna daalna hai?") == FarmerIntent.FERTILIZER
    assert engine.detect_primary_intent("PM Kisan yojana apply kaise kare?") == FarmerIntent.GOVERNMENT_SCHEMES
