"""
BhoomiMitra AI — Weather Integration Tests

Tests the weather module client, caching, Telugu/English formatting,
intent maps, service logic, and FastAPI endpoints.
NO real API keys or external connections needed.
"""
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.weather.schemas import WeatherForecastResponse
from src.weather.service import WeatherService
from src.weather.dependencies import get_weather_service

client = TestClient(app)
NOW = datetime.utcnow()


@pytest.fixture
def mock_weather_service():
    service = AsyncMock(spec=WeatherService)
    app.dependency_overrides[get_weather_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_weather_response(location_name="Warangal", is_live=False, has_rain=False) -> WeatherForecastResponse:
    tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")
    forecast_item = {
        "dt_txt": tomorrow_str,
        "temp": 28.5,
        "humidity": 70,
        "description": "Light Rain" if has_rain else "Clear sky",
        "condition_code": 500 if has_rain else 800,
    }

    return WeatherForecastResponse.model_validate({
        "location_name": location_name,
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {
            "temp": 30.2,
            "feels_like": 32.5,
            "humidity": 65,
            "wind_speed": 12.0,
            "description": "Partly Cloudy",
            "condition_code": 802,
        },
        "forecast": [forecast_item],
        "data_available": True,
        "data_freshness_minutes": 5.0,
        "source_note": "OpenWeather API (Live)" if is_live else "Simulated Weather (Local Fallback)",
        "is_live": is_live,
    })


# ---------------------------------------------------------------------------
# 1. GET /weather/forecast — returns 200 with forecast data
# ---------------------------------------------------------------------------

def test_get_weather_forecast_endpoint(mock_weather_service):
    """GET /weather/forecast returns 200 with valid weather data."""
    mock_weather_service.get_weather_for_query.return_value = _mock_weather_response(location_name="Warangal")

    response = client.get("/weather/forecast?district=Warangal&state=Telangana")
    assert response.status_code == 200
    data = response.json()
    assert data["location_name"] == "Warangal"
    assert data["data_available"] is True
    assert data["current"]["temp"] == 30.2
    assert len(data["forecast"]) == 1


# ---------------------------------------------------------------------------
# 2. GET /weather/forecast — with coordinates
# ---------------------------------------------------------------------------

def test_get_weather_forecast_coords_endpoint(mock_weather_service):
    """GET /weather/forecast with lat/lon returns 200 with weather data."""
    mock_weather_service.get_weather_for_query.return_value = _mock_weather_response(location_name="Farm (17.97, 79.59)")

    response = client.get("/weather/forecast?latitude=17.9689&longitude=79.5941")
    assert response.status_code == 200
    data = response.json()
    assert "Farm" in data["location_name"]
    assert data["current"]["humidity"] == 65


# ---------------------------------------------------------------------------
# 3. Intent Detection — English
# ---------------------------------------------------------------------------

def test_weather_intent_detection_english():
    """Detects weather intent in English query."""
    from src.weather.service import WEATHER_KEYWORDS_EN

    query = "is it going to rain tomorrow in my village?"
    query_lower = query.lower()
    has_intent = any(kw in query_lower for kw in WEATHER_KEYWORDS_EN)
    assert has_intent, "Should detect rain forecast intent in English"


# ---------------------------------------------------------------------------
# 4. Intent Detection — Telugu
# ---------------------------------------------------------------------------

def test_weather_intent_detection_telugu():
    """Detects weather intent in Telugu query."""
    from src.weather.service import WEATHER_KEYWORDS_TE

    query = "రేపు వర్షం పడుతుందా?"
    has_intent = any(kw in query for kw in WEATHER_KEYWORDS_TE)
    assert has_intent, "Should detect weather/rain intent in Telugu"


# ---------------------------------------------------------------------------
# 5. Early exit on non-weather query
# ---------------------------------------------------------------------------

def test_no_weather_intent_skipped():
    """Non-weather query returns False for intent check."""
    from src.weather.service import WEATHER_KEYWORDS_EN, WEATHER_KEYWORDS_TE

    query = "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?"
    query_lower = query.lower()
    has_intent = any(kw in query_lower for kw in WEATHER_KEYWORDS_EN)
    has_intent = has_intent or any(kw in query_lower for kw in WEATHER_KEYWORDS_TE)
    assert not has_intent, "Pest diagnosis query should not trigger weather intent"


# ---------------------------------------------------------------------------
# 6. OpenWeatherClient — key-less fallback mock data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openweather_client_no_key_mock():
    """Client returns simulated mock weather response when API key is missing."""
    from src.weather.openweather_client import OpenWeatherClient

    client_obj = OpenWeatherClient(
        api_key="",
        api_url="https://api.openweathermap.org/data/2.5/forecast",
        cache_ttl_seconds=1800,
    )

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()) as mock_set_cache:
        
        result = await client_obj.fetch_weather(district="Warangal", state="Telangana")

    assert result is not None
    assert result["is_live"] is False
    assert "Simulated Weather" in result["source_note"]
    assert result["location_name"] == "Warangal"
    assert "current" in result
    assert len(result["forecast"]) == 40
    mock_set_cache.assert_called_once()


# ---------------------------------------------------------------------------
# 7. OpenWeatherClient — API error returns None gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openweather_client_api_error():
    """Client handles HTTP 500 errors gracefully, returning None."""
    from src.weather.openweather_client import OpenWeatherClient

    client_obj = OpenWeatherClient(
        api_key="fake-api-key",
        api_url="https://api.openweathermap.org/data/2.5/forecast",
        cache_ttl_seconds=1800,
    )

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch("httpx.AsyncClient.get", return_value=mock_response):
        
        result = await client_obj.fetch_weather(district="Warangal", state="Telangana")

    assert result is None


# ---------------------------------------------------------------------------
# 8. Service Formatting — English
# ---------------------------------------------------------------------------

def test_weather_formatting_english_clear():
    """English weather text format with clear/sunny condition."""
    from src.weather.service import WeatherService as WS

    svc = WS.__new__(WS)
    mock_resp = _mock_weather_response(location_name="Warangal", has_rain=False)
    reply = svc.format_whatsapp_reply(mock_resp, language="en")

    assert "Weather Information (Warangal)" in reply
    assert "Temperature: 30.2" in reply
    assert "Humidity: 65%" in reply
    assert "Tomorrow's Forecast: Weather is expected to be clear" in reply


# ---------------------------------------------------------------------------
# 9. Service Formatting — English with Rain Alert
# ---------------------------------------------------------------------------

def test_weather_formatting_english_rain():
    """English weather text format with rain alert."""
    from src.weather.service import WeatherService as WS

    svc = WS.__new__(WS)
    mock_resp = _mock_weather_response(location_name="Warangal", has_rain=True)
    reply = svc.format_whatsapp_reply(mock_resp, language="en")

    assert "Tomorrow's Forecast: Rain is expected in your area." in reply


# ---------------------------------------------------------------------------
# 10. Service Formatting — Telugu with Rain Alert
# ---------------------------------------------------------------------------

def test_weather_formatting_telugu_rain():
    """Telugu weather text format with rain alerts and translations."""
    from src.weather.service import WeatherService as WS

    svc = WS.__new__(WS)
    mock_resp = _mock_weather_response(location_name="Warangal", has_rain=True)
    reply = svc.format_whatsapp_reply(mock_resp, language="te")

    assert "వాతావరణ సమాచారం (Warangal)" in reply
    assert "ఉష్ణోగ్రత: 30.2" in reply
    assert "తేమ (Humidity): 65%" in reply
    assert "రేపటి అంచనా: మీ ప్రాంతంలో వర్షం పడే అవకాశం ఉంది." in reply
    assert "ఉరుములతో కూడిన వర్షం" not in reply  # current is partly cloudy (code 802)


# ---------------------------------------------------------------------------
# 11. OpenWeatherClient — no key in production returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openweather_client_no_key_production():
    """Client returns None in production environment if API key is missing."""
    from src.weather.openweather_client import OpenWeatherClient

    client_obj = OpenWeatherClient(
        api_key="",
        api_url="https://api.openweathermap.org/data/2.5/forecast",
        cache_ttl_seconds=1800,
    )

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch("src.weather.openweather_client.get_settings") as mock_get_settings:
        
        mock_get_settings.return_value.app_env = "production"
        result = await client_obj.fetch_weather(district="Warangal", state="Telangana")

    assert result is None


# ===========================================================================
# INTEGRATION TESTS — enrich_response_with_weather() full pipeline
# ===========================================================================
# These tests verify the enrich_response_with_weather() orchestration
# function end-to-end with mocked DB, no real API key, no Redis, and no
# WhatsApp credentials.
# ===========================================================================


def _make_mock_farmer(farmer_id=None, language="en"):
    """Helper: lightweight mock Farmer with id and preferred_language."""
    farmer = MagicMock()
    farmer.id = farmer_id or uuid4()
    farmer.preferred_language = language
    return farmer


def _make_normalised_weather_dict(location_name="Warangal"):
    """Helper: normalised dict matching OpenWeatherClient.fetch_weather() output."""
    tomorrow_str = (NOW + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")
    return {
        "location_name": location_name,
        "latitude": 17.9689,
        "longitude": 79.5941,
        "current": {
            "temp": 30.2,
            "feels_like": 32.5,
            "humidity": 65,
            "wind_speed": 12.0,
            "description": "Partly Cloudy",
            "condition_code": 802,
        },
        "forecast": [{
            "dt_txt": tomorrow_str,
            "temp": 28.5,
            "humidity": 70,
            "description": "Clear Sky",
            "condition_code": 800,
        }],
        "data_available": True,
        "is_live": False,
        "source_note": "Simulated Weather (Local Fallback)",
    }


def _mock_db_with_location(memory_gps=None, memory_district=None, memory_state=None,
                            profile_district=None, profile_state=None):
    """Helper: create a mock AsyncSession with FarmerMemory and FarmerProfile results."""
    mock_memory = MagicMock()
    mock_memory.gps_coordinates = memory_gps or {}
    mock_memory.district = memory_district
    mock_memory.state = memory_state

    mock_profile = MagicMock()
    mock_profile.district = profile_district
    mock_profile.state = profile_state

    memory_result = MagicMock()
    memory_result.scalar_one_or_none.return_value = mock_memory
    profile_result = MagicMock()
    profile_result.scalar_one_or_none.return_value = mock_profile

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[memory_result, profile_result])
    return db


# ---------------------------------------------------------------------------
# 12. Integration: GPS from FarmerMemory resolves weather
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_gps_from_farmer_memory():
    """Weather enrichment appends forecast when FarmerMemory has GPS coordinates."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="en")
    db = _mock_db_with_location(memory_gps={"latitude": 17.385, "longitude": 78.4867})
    original = "Here is some farming advice."

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=_make_normalised_weather_dict("Hyderabad"),
    ):
        result = await enrich_response_with_weather(
            db, "What is the weather today?", original, farmer
        )

    assert original in result, "Original AI response must be preserved"
    assert "Weather Information" in result, "English weather title must be present"
    assert "30.2" in result, "Temperature from mock data must appear"
    assert result.startswith(original), "Weather block must be appended, not prepended"


# ---------------------------------------------------------------------------
# 13. Integration: District from FarmerProfile resolves weather (Telugu)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_district_from_farmer_profile():
    """Weather enrichment resolves district from FarmerProfile and responds in Telugu."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="te")
    db = _mock_db_with_location(profile_district="Warangal", profile_state="Telangana")
    original = "వ్యవసాయ సలహా ఇక్కడ ఉంది."

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=_make_normalised_weather_dict("Warangal"),
    ):
        result = await enrich_response_with_weather(
            db, "రేపు వర్షం పడుతుందా?", original, farmer
        )

    assert original in result, "Original AI response must be preserved"
    assert "వాతావరణ సమాచారం" in result, "Telugu weather title must be present"
    assert "ఉష్ణోగ్రత" in result, "Telugu temperature label must be present"


# ---------------------------------------------------------------------------
# 14. Integration: No location — asks farmer for district/area
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_no_location_returns_original():
    """When farmer has no location data, enrichment prompts farmer to provide district."""
    from src.weather.service import enrich_response_with_weather

    farmer = _make_mock_farmer(language="en")
    db = _mock_db_with_location()  # No GPS, no district
    original = "Here is some farming advice."

    result = await enrich_response_with_weather(
        db, "Will it rain tomorrow?", original, farmer
    )

    assert original in result
    assert "Please provide your district or area name" in result


# ---------------------------------------------------------------------------
# 15. Integration: API failure does not crash — returns original response with fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_api_failure_returns_original():
    """When weather API raises an exception, enrichment safely returns original response."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer()
    db = _mock_db_with_location(memory_gps={"latitude": 17.385, "longitude": 78.4867})
    original = "Here is some farming advice."

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        side_effect=Exception("Simulated API connection failure"),
    ):
        result = await enrich_response_with_weather(
            db, "What is the weather forecast?", original, farmer
        )

    assert result == original, "Original response must survive an unhandled API exception"


# ---------------------------------------------------------------------------
# 16. Integration: Non-weather query skips enrichment entirely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_non_weather_query_skips():
    """Non-weather query returns AI response unchanged and makes no DB queries."""
    from src.weather.service import enrich_response_with_weather

    farmer = _make_mock_farmer()
    db = AsyncMock()
    original = "నిమ్మ నూనె వాడి తెగులు నివారణ చేయండి."

    result = await enrich_response_with_weather(
        db, "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?", original, farmer
    )

    assert result == original, "Response must be unchanged for non-weather queries"
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# 17. Regression: Extract district from query in English & Telugu
# ---------------------------------------------------------------------------

def test_extract_district_from_weather_query():
    """Extracts district correctly from various colloquial query formats."""
    from src.weather.service import _extract_district_from_query

    assert _extract_district_from_query("వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?") == "Warangal"
    assert _extract_district_from_query("వరంగల్లో వాతావరణం ఎలా ఉంది?") == "Warangal"
    assert _extract_district_from_query("Will it rain tomorrow in Warangal?") == "Warangal"
    assert _extract_district_from_query("What is the temperature in Karimnagar?") == "Karimnagar"
    assert _extract_district_from_query("గుంటూరు జిల్లాలో వర్షం కురుస్తుందా?") == "Guntur"
    assert _extract_district_from_query("weather forecast near vijayawada") == "Krishna"
    assert _extract_district_from_query("రేపు వర్షం పడుతుందా?") is None


# ---------------------------------------------------------------------------
# 18. Regression: Telugu weather query with Warangal returns actual forecast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_query_with_warangal_telugu():
    """Telugu weather query mentioning Warangal returns real forecast data without inventing info."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="te")
    db = _mock_db_with_location()  # Profile has no district, but query has 'వరంగల్'
    original = "వ్యవసాయ సలహా."

    mock_weather = _make_normalised_weather_dict(location_name="Warangal")
    mock_weather["is_live"] = True

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=mock_weather,
    ):
        result = await enrich_response_with_weather(
            db, "వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?", original, farmer
        )

    assert "వాతావరణ సమాచారం (Warangal)" in result
    assert "ఉష్ణోగ్రత: 30.2°C" in result
    assert "తేమ (Humidity): 65%" in result
    assert "గాలి వేగం: 12.0 km/h" in result
    assert "రేపటి అంచనా" in result
    assert "ఓపెన్వెదర్ (లైవ్)" in result


# ---------------------------------------------------------------------------
# 19. Regression: English weather query with Warangal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_query_with_warangal_english():
    """English weather query mentioning Warangal extracts location and formats English response."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="en")
    db = _mock_db_with_location()
    original = "Here is your agricultural advisory."

    mock_weather = _make_normalised_weather_dict(location_name="Warangal")
    mock_weather["is_live"] = True

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=mock_weather,
    ):
        result = await enrich_response_with_weather(
            db, "Will it rain tomorrow in Warangal?", original, farmer
        )

    assert "Weather Information (Warangal)" in result
    assert "Temperature: 30.2°C" in result
    assert "Humidity: 65%" in result
    assert "Tomorrow's Forecast" in result
    assert "OpenWeather (Live)" in result


# ---------------------------------------------------------------------------
# 20. Regression: Weather API unavailable returns honest fallback note
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_api_unavailable_fallback():
    """When weather API returns no data, append honest fallback note without inventing data."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="te")
    db = _mock_db_with_location()
    original = "వ్యవసాయ సలహా."

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await enrich_response_with_weather(
            db, "వరంగల్ ప్రాంతంలో రేపు వర్షం పడుతుందా?", original, farmer
        )

    assert "ఈ ప్రాంతానికి సంబంధించిన వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు" in result
    assert "1800-180-1551" in result


# ---------------------------------------------------------------------------
# 21. Regression: Conversational district follow-up after weather prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_weather_conversational_district_followup():
    """When user replies with just 'Warangal' following a weather question, triggers weather enrichment."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="en")
    db = _mock_db_with_location()
    # AI response from Gemini acknowledges weather
    ai_response = "Here is the weather forecast for Warangal district."

    mock_weather = _make_normalised_weather_dict(location_name="Warangal")

    with patch.object(
        OpenWeatherClient, "fetch_weather",
        new_callable=AsyncMock,
        return_value=mock_weather,
    ):
        result = await enrich_response_with_weather(
            db, "Warangal", ai_response, farmer
        )

    assert "Weather Information (Warangal)" in result
    assert "Temperature: 30.2°C" in result
