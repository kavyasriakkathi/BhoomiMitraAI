"""
BhoomiMitra AI — Real-World Farmer Conversation Integration Test Harness

Simulates realistic WhatsApp farmer conversations across:
1. Telugu single-intent & multi-intent queries
2. English single-intent & multi-intent queries
3. Gemini timeouts & failovers to specialized services
4. Specialized service unavailability (Weather, Shop, Market, Scheme)
5. Compound failure scenarios
6. Agricultural safety & pesticide hallucination prevention

All tests use mocks for external third-party APIs (OpenWeather, Agmarknet, Meta WhatsApp, Google Gemini)
and run fast and deterministically.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from src.core.models import Farmer, Conversation, Shop, Inventory, GovernmentScheme, MarketPrice
from src.ai.service import process_text_message
from src.ai.schemas import AIGenerateResponse
from src.weather.schemas import WeatherForecastResponse


# ==============================================================================
# Helper Mock Builders
# ==============================================================================

def make_sample_farmer(language: str = "te", district: str = "Warangal", state: str = "Telangana") -> Farmer:
    farmer = Farmer(
        id=uuid4(),
        phone_number="919876543210",
        preferred_language=language,
        is_active=True,
    )
    return farmer


def make_sample_shop():
    shop = Shop(
        id=uuid4(),
        shop_name="శ్రీ బాలాజీ ఆగ్రో ఏజెన్సీస్",
        owner_name="రాజేష్",
        phone_number="9848012345",
        address="మార్కెట్ రోడ్, వరంగల్",
        district="Warangal",
        state="Telangana",
        latitude=17.9689,
        longitude=79.5941,
        status="active",
        delivery_available=True,
        opening_time="08:00",
        closing_time="20:00",
    )
    inventory = Inventory(
        id=uuid4(),
        shop_id=shop.id,
        product_name="యూరియా (Urea 45kg)",
        brand="IFFCO",
        price=266.5,
        quantity_in_stock=50,
        minimum_stock_level=5,
        unit="బస్తా",
        available=True,
    )
    return [(shop, inventory)]


def make_sample_weather(district: str = "Warangal") -> WeatherForecastResponse:
    from datetime import timedelta
    tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")
    return WeatherForecastResponse.model_validate({
        "location_name": district,
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {
            "temp": 30.5,
            "feels_like": 33.0,
            "humidity": 72,
            "wind_speed": 11.0,
            "description": "పాక్షికంగా మేఘావృతమై ఉంది",
            "condition_code": 802,
        },
        "forecast": [
            {
                "dt_txt": tomorrow_str,
                "temp": 29.0,
                "humidity": 78,
                "description": "వర్షం (Rain)",
                "condition_code": 500,
            }
        ],
        "data_available": True,
        "is_live": True,
        "source_note": "OpenWeather (Live)",
    })


def make_sample_schemes() -> list:
    return [
        GovernmentScheme(
            id=uuid4(),
            scheme_name="పీఎం కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
            scheme_code="PM_KISAN",
            category="Direct Income Support",
            description="రైతులకు ఏటా ₹6000 ఆర్థిక సహాయం.",
            state="Telangana",
            benefits_summary="ఏడాదికి ₹6,000",
            eligibility_criteria="భూమి ఉన్న చిన్న, సన్నకారు రైతులు",
            required_documents="ఆధార్ కార్డ్, పట్టాదారు పాస్ బుక్, బ్యాంక్ ఖాతా",
            official_portal_url="https://pmkisan.gov.in",
            min_land_acres=0.1,
            is_active=True,
        )
    ]


def make_sample_mandi() -> list:
    return [
        MarketPrice(
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
            updated_at=datetime.utcnow(),
        )
    ]


# ==============================================================================
# SECTION 1: Realistic Telugu Farmer Scenarios
# ==============================================================================

@pytest.mark.asyncio
async def test_telugu_fertilizer_availability():
    """Query: 'యూరియా ఎక్కడ దొరుకుతుంది?' -> Authoritative nearby shop details."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="యూరియా ఎక్కడ దొరుకుతుంది?")

    mock_ai = AIGenerateResponse(
        response_text="యూరియా సమీప డీలర్ల వద్ద అందుబాటులో ఉంది.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "శ్రీ బాలాజీ ఆగ్రో ఏజెన్సీస్" in reply
        assert "266.5" in reply
        assert "9848012345" in reply


@pytest.mark.asyncio
async def test_telugu_weather_forecast():
    """Query: 'వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?' -> Authoritative weather forecast."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?")

    mock_ai = AIGenerateResponse(
        response_text="వాతావరణ సమాచారం కింద ఇవ్వబడింది.",
        intent="weather",
        confidence=0.9,
        provider_used="gemini",
    )

    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "వాతావరణ సమాచారం" in reply
        assert "30.5" in reply
        assert "వర్షం" in reply


@pytest.mark.asyncio
async def test_telugu_crop_pest_advice():
    """Query: 'పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి?' -> Safe grounded crop advisory."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి?")

    mock_ai = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు లీటరు నీటికి కలిపి పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "ఎమామెక్టిన్ బెంజోయేట్" in reply
        assert "0.4 గ్రాములు" in reply


@pytest.mark.asyncio
async def test_telugu_government_schemes():
    """Query: 'రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?' -> Authoritative scheme details."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?")

    mock_ai = AIGenerateResponse(
        response_text="ప్రభుత్వ పథకాల వివరాలు కింద ఉన్నాయి.",
        intent="schemes",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=make_sample_schemes()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "పీఎం కిసాన్" in reply
        assert "₹6,000" in reply
        assert "https://pmkisan.gov.in" in reply


@pytest.mark.asyncio
async def test_telugu_market_price():
    """Query: 'పత్తి మార్కెట్ ధర ఎంత?' -> Authoritative mandi prices."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="పత్తి మార్కెట్ ధర ఎంత?")

    mock_ai = AIGenerateResponse(
        response_text="మార్కెట్ ధరలు క్రింద ఇవ్వబడ్డాయి.",
        intent="market_price",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=make_sample_mandi()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "పత్తి మార్కెట్ ధరలు" in reply
        assert "Warangal Mandi" in reply
        assert "7,450" in reply


@pytest.mark.asyncio
async def test_telugu_exact_four_intent_query():
    """
    Combined query:
    'పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది? వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా? రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?'
    Verify:
    - Crop advice appears once
    - Weather appears once
    - Shops appear once
    - Schemes appear once
    - Market prices do NOT appear (unrequested)
    - Greetings and closings removed
    """
    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది? వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా? రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="నమస్తే రైతు సోదరా! పత్తి పంటలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు లీటరు నీటికి కలిపి పిచికారీ చేయాలి. మీకు ఇంకా ఏమైనా సహాయం కావాలా?",
        intent="multi_intent",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=make_sample_schemes()):

        reply = await process_text_message(db_mock, farmer, conv)

        # 1. Section presence
        assert "🌱 *పంట సలహా*" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in reply
        assert "🏛️ *ప్రభుత్వ పథకాలు*" in reply

        # 2. Market prices NOT present
        assert "📊 *మార్కెట్ ధరలు*" not in reply

        # 3. Content accuracy
        assert "ఎమామెక్టిన్ బెంజోయేట్" in reply
        assert "శ్రీ బాలాజీ ఆగ్రో ఏజెన్సీస్" in reply
        assert "పీఎం కిసాన్" in reply

        # 4. Filler removed
        assert "నమస్తే" not in reply
        assert "సహాయం కావాలా" not in reply


# ==============================================================================
# SECTION 2: Realistic English Farmer Scenarios
# ==============================================================================

@pytest.mark.asyncio
async def test_english_fertilizer_availability():
    """Query: 'Where can I buy urea?' -> English shop block."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="en")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="Where can I buy urea?")

    mock_ai = AIGenerateResponse(
        response_text="You can buy Urea fertilizer at local agricultural stores.",
        intent="shop_search",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "Nearby Agricultural Shops" in reply
        assert "శ్రీ బాలాజీ ఆగ్రో ఏజెన్సీస్" in reply
        assert "266.5" in reply


@pytest.mark.asyncio
async def test_english_weather_forecast():
    """Query: 'What is the weather forecast in Warangal?' -> English weather block."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="en")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What is the weather forecast in Warangal?")

    mock_ai = AIGenerateResponse(
        response_text="Here is the weather information for Warangal.",
        intent="weather",
        confidence=0.9,
        provider_used="gemini"
    )

    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "Weather Information" in reply
        assert "Temperature: 30.5°C" in reply


@pytest.mark.asyncio
async def test_english_crop_advisory():
    """Query: 'What should I do for bollworm pests in cotton?' -> English crop advice."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="en")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What should I do for bollworm pests in cotton?")

    mock_ai = AIGenerateResponse(
        response_text="For bollworm in cotton, spray Emamectin Benzoate 5% SG @ 0.4g per litre of water.",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "Emamectin Benzoate 5% SG @ 0.4g per litre" in reply


@pytest.mark.asyncio
async def test_english_government_schemes():
    """Query: 'What government schemes are available for farmers?' -> English schemes block."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="en")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What government schemes are available for farmers?")

    mock_ai = AIGenerateResponse(
        response_text="Government scheme details are listed below.",
        intent="schemes",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=make_sample_schemes()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "Government Schemes Available For You" in reply
        assert "PM-KISAN" in reply
        assert "Benefits" in reply


@pytest.mark.asyncio
async def test_english_market_prices():
    """Query: 'What is the market price of cotton?' -> English mandi prices block."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="en")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, user_message="What is the market price of cotton?")

    mock_ai = AIGenerateResponse(
        response_text="Market price details are listed below.",
        intent="market_price",
        confidence=0.95,
        provider_used="gemini"
    )

    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_ai), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=make_sample_mandi()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "Cotton Mandi Prices" in reply
        assert "Warangal Mandi" in reply
        assert "7,450" in reply


@pytest.mark.asyncio
async def test_english_combined_multi_intent_query():
    """Combined English multi-intent query format and deduplication."""
    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="en")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="What spray for pests in cotton, where to buy urea, what is the weather in Warangal, and what government schemes are available?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="Hello farmer! For bollworm pests in cotton, apply Emamectin Benzoate 5% SG @ 0.4g/litre. Please let me know if you need anything else.",
        intent="multi_intent",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=make_sample_schemes()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *Crop Advice*" in reply
        assert "🌡️ *Weather Information*" in reply
        assert "🏬 *Nearby Shops & Availability*:" in reply
        assert "🏛️ *Government Schemes*" in reply
        assert "📊 *Market Prices*" not in reply

        assert "Emamectin Benzoate" in reply
        assert "Hello farmer!" not in reply
        assert "Please let me know if you need anything else" not in reply


# ==============================================================================
# SECTION 3: Failure, Timeout, & Unavailability Scenarios
# ==============================================================================

@pytest.mark.asyncio
async def test_gemini_timeout_with_specialized_services_graceful_continuation():
    """
    When Gemini times out on a query asking for Weather + Shops + Schemes,
    specialized services continue and provide their authoritative data without leaking errors.
    """
    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="యూరియా ఎక్కడ దొరుకుతుంది? వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా? రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?"
    )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=TimeoutError("Gemini API timed out after 5.0s")), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=make_sample_schemes()):

        reply = await process_text_message(db_mock, farmer, conv)

        # Specialized data returned
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in reply
        assert "🏛️ *ప్రభుత్వ పథకాలు*" in reply

        # No error leakage
        assert "Traceback" not in reply
        assert "TimeoutError" not in reply
        assert "503" not in reply
        assert "Gemini" not in reply


@pytest.mark.asyncio
async def test_weather_unavailable_notice():
    """When weather is requested but unavailable, exactly one concise notice is shown."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.weather.service.WeatherService.get_weather_for_query", side_effect=Exception("OpenWeather API down")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "ఈ ప్రాంతానికి ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు" in reply


@pytest.mark.asyncio
async def test_shop_unavailable_notice():
    """When shop/product is requested but no shop is in stock/registered, concise notice is shown."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. యూరియా ఎక్కడ దొరుకుతుంది?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=[]):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in reply
        assert "ప్రస్తుతం ఈ ఉత్పత్తికి సమీప దుకాణాలు అందుబాటులో లేవు" in reply


@pytest.mark.asyncio
async def test_market_unavailable_notice():
    """When market price is requested in multi-intent for unlisted crop, concise honest notice is given."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. అరటి మార్కెట్ ధర ఎంత?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=[]):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "📊 *మార్కెట్ ధరలు*" in reply
        assert "ఈ పంటకు మార్కెట్ ధరలు అందుబాటులో లేవు" in reply



@pytest.mark.asyncio
async def test_multiple_simultaneous_service_failures_clean_handling():
    """When Gemini, Weather, and Shops all fail simultaneously, pipeline returns graceful fallback."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="సాధారణ సమాచారం చెప్పండి"
    )

    with patch("src.ai.service.AIService.generate_ai_response", side_effect=RuntimeError("AI server unreachable")):
        reply = await process_text_message(db_mock, farmer, conv)

        # Must return the safe localized fallback message
        assert "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది" in reply
        assert "Traceback" not in reply
        assert "RuntimeError" not in reply


# ==============================================================================
# SECTION 4: Additional Multi-Intent & Edge Cases (Phase 6 Matrix)
# ==============================================================================

@pytest.mark.asyncio
async def test_telugu_crop_plus_weather():
    """Query: 'పత్తి పంటలో పురుగులు వస్తున్నాయి, వరంగల్లో రేపు వర్షం పడుతుందా?' -> Crop + Weather."""
    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి, వరంగల్లో రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు లీటరు నీటికి కలిపి పిచికారీ చేయాలి.",
        intent="multi_intent",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" not in reply
        assert "📊 *మార్కెట్ ధరలు*" not in reply
        assert "🏛️ *ప్రభుత్వ పథకాలు*" not in reply


@pytest.mark.asyncio
async def test_telugu_crop_plus_shop():
    """Query: 'పత్తి పంటలో పురుగులు వస్తున్నాయి, యూరియా ఎక్కడ దొరుకుతుంది?' -> Crop + Shop."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి, యూరియా ఎక్కడ దొరుకుతుంది?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన పిచికారీ చేయండి.",
        intent="multi_intent",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.shops.service._resolve_farmer_location", new_callable=AsyncMock, return_value=(17.96, 79.59, "Warangal", "Telangana")), \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", return_value=make_sample_shop()):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" not in reply
        assert "📊 *మార్కెట్ ధరలు*" not in reply


@pytest.mark.asyncio
async def test_telugu_crop_plus_market_plus_weather():
    """Query: 'పత్తికి పురుగులు వచ్చాయి, మార్కెట్ ధర ఎంత, రేపు వర్షం పడుతుందా?' -> Crop + Market + Weather."""
    db_mock = AsyncMock()
    mock_mem = MagicMock()
    mock_mem.district = "Warangal"
    mock_mem.state = "Telangana"
    mock_mem.gps_coordinates = {"latitude": 17.9689, "longitude": 79.5941}
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = mock_mem
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తికి పురుగులు వచ్చాయి, మార్కెట్ ధర ఎంత, రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ పిచికారీ చేయండి.",
        intent="multi_intent",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=make_sample_mandi()), \
         patch("src.weather.service.WeatherService.get_weather_for_query", new_callable=AsyncMock, return_value=make_sample_weather("Warangal")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "పత్తి మార్కెట్ ధరలు" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "🏬 *సమీప వ్యవసాయ దుకాణాలు*" not in reply
        assert "🏛️ *ప్రభుత్వ పథకాలు*" not in reply


@pytest.mark.asyncio
async def test_schemes_unavailable_notice():
    """When schemes are requested in multi-intent but no scheme matches, concise notice is given."""
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. రైతులకు ఏ పథకాలు ఉన్నాయి?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన చర్యలు చేపట్టండి.",
        intent="multi_intent",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.schemes.repository.SchemeRepository.seed_default_schemes_if_empty", new_callable=AsyncMock), \
         patch("src.schemes.repository.SchemeRepository.get_all_active", new_callable=AsyncMock, return_value=[]):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🏛️ *ప్రభుత్వ పథకాలు*" in reply
        assert "పథకాలు అందుబాటులో లేవు" in reply


@pytest.mark.asyncio
async def test_external_api_timeout_handling():
    """When an external API (like OpenWeather) times out via httpx.TimeoutException in multi-intent."""
    import httpx
    db_mock = AsyncMock()
    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. వరంగల్లో రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన చర్యలు చేపట్టండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.weather.service.WeatherService.get_weather_for_query", side_effect=httpx.TimeoutException("OpenWeather timed out")):

        reply = await process_text_message(db_mock, farmer, conv)

        assert "🌱 *పంట సలహా*" in reply
        assert "🌡️ *వాతావరణ సమాచారం*" in reply
        assert "ఈ ప్రాంతానికి ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు" in reply
        assert "Traceback" not in reply


@pytest.mark.asyncio
async def test_weather_unknown_location_fallback():
    """When weather is requested but location cannot be resolved."""
    db_mock = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_exec)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="రేపు వర్షం పడుతుందా?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="వాతావరణ సమాచారం తెలుసుకోవడానికి మీ జిల్లా లేదా గ్రామం పేరు చెప్పండి.",
        intent="weather",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "జిల్లా" in reply or "వాతావరణ" in reply


def test_empty_or_whitespace_message_handling():
    """Empty or whitespace user messages should be gracefully handled."""
    from src.gateway.router import _extract_message
    from src.gateway.schemas import WhatsAppMessage

    # Empty text message
    msg = WhatsAppMessage(
        from_="919876543210",
        id="wamid.EMPTY123",
        timestamp="1700000000",
        type="text",
        text=None
    )
    parsed = _extract_message(msg)
    assert parsed is None


def test_invalid_webhook_payload_handling():
    """Invalid webhook payloads should return status: ignored without crashing."""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    # Post invalid JSON
    res = client.post("/webhook/whatsapp", content=b"invalid json payload", headers={"Content-Type": "application/json"})
    assert res.status_code == 200
    assert res.json()["status"] == "ignored"


@pytest.mark.asyncio
async def test_duplicate_whatsapp_message_deduplication():
    """Stage 1 deduplication returns immediately if message_id exists."""
    from src.gateway.service import process_message_pipeline
    from src.gateway.schemas import ParsedIncomingMessage

    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DUPLICATE_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Hello duplicate test",
    )

    # Mock DB context where message already exists
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = Conversation(id=uuid4(), message_id="wamid.DUPLICATE_TEST")
    mock_db.execute.return_value = mock_result

    class MockContextManager:
        async def __aenter__(self):
            return mock_db
        async def __aexit__(self, exc_type, exc, tb):
            pass

    with patch("src.gateway.service.AsyncSessionLocal", return_value=MockContextManager()), \
         patch("src.ai.service.AIService.generate_ai_response") as mock_ai:

        await process_message_pipeline(parsed)

        # AI should never be invoked for duplicate message
        mock_ai.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_duplicate_message_integrity_error():
    """Stage 4 catches IntegrityError / UniqueConstraint race conditions and skips processing."""
    from src.gateway.service import process_message_pipeline
    from src.gateway.schemas import ParsedIncomingMessage
    from sqlalchemy.exc import IntegrityError

    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.CONCURRENT_RACE",
        timestamp="1700000000",
        message_type="text",
        text_content="Hello race condition test",
    )

    # First check returns None (Stage 1 passes), but commit raises IntegrityError (Stage 4)
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    mock_db.commit.side_effect = IntegrityError("duplicate key", params=None, orig=Exception("unique constraint"))

    class MockContextManager:
        async def __aenter__(self):
            return mock_db
        async def __aexit__(self, exc_type, exc, tb):
            pass

    with patch("src.gateway.service.AsyncSessionLocal", return_value=MockContextManager()), \
         patch("src.ai.service.AIService.generate_ai_response") as mock_ai:

        await process_message_pipeline(parsed)

        # AI should never be called because IntegrityError aborted pipeline cleanly
        mock_ai.assert_not_called()


# ==============================================================================
# SECTION 5: Crop Advice Safety & Anti-Hallucination Regression Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_safety_unknown_pest():
    """When a farmer asks about an unknown/unidentifiable pest, AI must NOT guess arbitrary chemicals."""
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="నా చేనులో ఏదో వింత నల్లటి పురుగులు కనపడుతున్నాయి, ఏ మందు కొట్టాలి?"
    )

    # Mock Gemini adhering to strict safety rules: asking for symptoms and suggesting AEO/KVK
    safe_response = (
        "పురుగుల రకం ఖచ్చితంగా తెలియనప్పుడు రసాయన మందులు వాడకూడదు. "
        "దయచేసి ఆ పురుగుల లక్షణాలు లేదా ఆకులపై జరిగిన నష్టాన్ని వివరించండి, "
        "లేదా స్థానిక వ్యవసాయ అధికారిని (AEO) సంప్రదించండి."
    )
    mock_gemini = AIGenerateResponse(
        response_text=safe_response,
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini):
        reply = await process_text_message(db_mock, farmer, conv)

        # Must promote safety / clarification / AEO consultation
        assert "వ్యవసాయ అధికారి" in reply or "లక్షణాలు" in reply
        # Must NOT contain arbitrary chemical inventions
        assert "Chlorpyrifos 50%" not in reply
        assert "Monocrotophos" not in reply


@pytest.mark.asyncio
async def test_safety_unknown_disease():
    """When a farmer reports vague disease symptoms without known diagnosis, AI asks follow-ups instead of guessing treatments."""
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="ఆకులు వింతగా మారుతున్నాయి తెగులు ఏమిటో చెప్పండి"
    )

    safe_response = (
        "ఆకులు ఏ రంగులోకి మారుతున్నాయి (పసుపు, గోధుమ లేదా ఎరుపు)? "
        "ఆకులపై మచ్చలు లేదా ముడుతలు ఉన్నాయా? సరైన నివారణ కోసం వివరాలు తెలపండి."
    )
    mock_gemini = AIGenerateResponse(
        response_text=safe_response,
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "వివరాలు" in reply or "రంగు" in reply or "మచ్చలు" in reply
        # No arbitrary chemical dosage guessed
        assert "500 ml" not in reply
        assert "2.0 kg" not in reply


@pytest.mark.asyncio
async def test_safety_farmer_incomplete_info_no_crop():
    """When farmer asks for remedy without specifying crop, AI asks which crop they are growing."""
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="నా పంటలో పురుగులు వచ్చాయి మందు చెప్పండి"
    )

    safe_response = (
        "మీరు ఏ పంట (పత్తి, వరి, మిరప మొదలైనవి) సాగు చేస్తున్నారు? "
        "మరియు పురుగుల లక్షణాలు ఏమిటి? పంట పేరు చెబితే సరైన సలహా ఇవ్వగలను."
    )
    mock_gemini = AIGenerateResponse(
        response_text=safe_response,
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "ఏ పంట" in reply or "పంట పేరు" in reply
        # Must not guess a random pesticide without crop context
        assert "Dichlorvos" not in reply


@pytest.mark.asyncio
async def test_safety_dosage_only_query():
    """When farmer asks only 'ఎంత కొట్టాలి?' without naming the chemical or crop, AI asks for clarification."""
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="ఎంత కొట్టాలి?"
    )

    safe_response = (
        "మీరు ఏ మందు (రసాయనం/పురుగుమందు) మరియు ఏ పంట కోసం మోతాదు అడుగుతున్నారో తెలపండి. "
        "సరైన సమాచారం ఇస్తేనే సురక్షితమైన మోతాదు చెప్పగలను."
    )
    mock_gemini = AIGenerateResponse(
        response_text=safe_response,
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini):
        reply = await process_text_message(db_mock, farmer, conv)

        assert "ఏ మందు" in reply or "పంట" in reply or "మోతాదు" in reply
        # Must not invent a random quantity
        assert "2.5 ml per litre" not in reply


@pytest.mark.asyncio
async def test_safety_image_diagnosis_insufficient_information():
    """When image is blurry / insufficient for certain diagnosis, multimodal pipeline provides cautious advice and requests clearer photo."""
    from src.ai.service import process_image_message
    import json

    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="ఈ ఫోటో చూసి రోగం చెప్పండి"
    )

    # Multimodal JSON output adhering to IMAGE DIAGNOSIS SAFETY RULES
    multimodal_mock_json = json.dumps({
        "disease_name": "అస్పష్టమైన లక్షణాలు (Uncertain / Low Confidence)",
        "confidence_score": 0.45,
        "severity": "low",
        "symptoms": "ఫోటో అస్పష్టంగా ఉంది. స్పష్టమైన తెగులు లక్షణాలు కనిపించడం లేదు.",
        "treatment_recommendation": "స్పష్టమైన ఫోటో తీసి మళ్ళీ పంపండి లేదా స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
        "friendly_whatsapp_reply": "ఫోటో స్పష్టంగా లేదు, కాబట్టి తెగులును ఖచ్చితంగా నిర్ధారించలేము. దయచేసి ఆకు పైభాగం మరియు వెనుక భాగాన్ని దగ్గరగా ఫోటో తీసి పంపండి."
    })

    with patch("src.ai.gemini_client.generate_multimodal_response", new_callable=AsyncMock, return_value=multimodal_mock_json):
        reply = await process_image_message(
            db=db_mock,
            farmer=farmer,
            conversation=conv,
            image_bytes=b"fake_blurry_image_data",
            mime_type="image/jpeg"
        )

        assert "స్పష్టంగా లేదు" in reply or "ఖచ్చితంగా" in reply
        assert "ఫోటో" in reply


@pytest.mark.asyncio
async def test_telugu_expert_escalation_e2e():
    """When a farmer requests human officer/expert help, system generates ticket and returns escalation contact details."""
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = make_sample_farmer(language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="నాకు వ్యవసాయ అధికారితో మాట్లాడాలి నిపుణుడి సహాయం కావాలి"
    )

    mock_gemini = AIGenerateResponse(
        response_text="మీ సమస్యను వ్యవసాయ అధికారికి పంపించడం జరిగింది.",
        intent="escalation",
        confidence=0.95,
        provider_used="gemini"
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.escalation.service.enrich_response_with_escalation") as mock_esc:
        mock_esc.return_value = (
            "మీ సమస్యను వ్యవసాయ అధికారికి పంపించడం జరిగింది.\n\n"
            "🎫 *సహాయ నిపుణుల సంప్రదింపు టికెట్*\n"
            "టికెట్ నంబర్: #ESC-98234\n"
            "స్థానిక వ్యవసాయ విస్తరణ అధికారి (AEO) త్వరలో మిమ్మల్ని సంప్రదిస్తారు."
        )

        reply = await process_text_message(db_mock, farmer, conv)

        assert "టికెట్" in reply or "వ్యవసాయ అధికారి" in reply
        assert "ESC-" in reply
