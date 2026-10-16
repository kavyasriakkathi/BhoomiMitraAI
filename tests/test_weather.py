"""
BhoomiMitra AI — Weather Integration Tests

Tests the weather module client, caching, Telugu/English formatting,
intent maps, service logic, and FastAPI endpoints.
NO real API keys or external connections needed.
"""
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.weather.schemas import WeatherForecastResponse
from src.weather.service import WeatherService
from src.weather.dependencies import get_weather_service

client = TestClient(app)
NOW = datetime.now(timezone.utc)


@pytest.fixture
def mock_weather_service():
    service = AsyncMock(spec=WeatherService)
    app.dependency_overrides[get_weather_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_weather_response(location_name="Warangal", is_live=False, has_rain=False) -> WeatherForecastResponse:
    tomorrow_str = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")
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


# ---------------------------------------------------------------------------
# 22. Regression: OpenWeather failure cascades to Open-Meteo fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openweather_failed_falls_back_to_open_meteo_live():
    """When OpenWeather API key is present but returns failure (e.g. 401 or timeout),

    the client must fall through to Open-Meteo instead of returning None.
    """
    from src.weather.openweather_client import OpenWeatherClient

    client = OpenWeatherClient(
        api_key="expired_or_invalid_key",
        api_url="https://api.openweathermap.org/data/2.5/forecast",
    )

    open_meteo_mock = {
        "location_name": "16.5123,80.6432",
        "latitude": 16.5123,
        "longitude": 80.6432,
        "current": {
            "temp": 29.5,
            "feels_like": 31.0,
            "humidity": 68,
            "wind_speed": 11.2,
            "description": "Clear Sky",
            "condition_code": 800,
        },
        "forecast": [],
        "data_available": True,
        "is_live": True,
        "source_note": "Live Weather (Open-Meteo)",
    }

    with patch.object(client, "_get_from_cache", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None) as mock_owm, \
         patch.object(client, "_fetch_open_meteo", new_callable=AsyncMock, return_value=open_meteo_mock) as mock_om, \
         patch.object(client, "_set_in_cache", new_callable=AsyncMock):

        data = await client.fetch_weather(latitude=16.5123, longitude=80.6432)

        # 1. OpenWeather was attempted
        mock_owm.assert_awaited_once_with(16.5123, 80.6432, None, None)
        # 2. Open-Meteo was invoked as fallback
        mock_om.assert_awaited_once_with(16.5123, 80.6432, "16.5123,80.6432")
        # 3. Valid live data was produced
        assert data is not None
        assert data["data_available"] is True
        assert data["is_live"] is True
        assert data["current"]["temp"] == 29.5
        assert data["source_note"] == "Live Weather (Open-Meteo)"


# ---------------------------------------------------------------------------
# 23. Regression: _fetch_open_meteo includes required descriptive User-Agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_open_meteo_sends_user_agent_header():
    """_fetch_open_meteo must send a custom User-Agent to avoid datacenter 403 blocks."""
    from src.weather.openweather_client import OpenWeatherClient

    client = OpenWeatherClient(api_key="", api_url="")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "current": {
            "temperature_2m": 27.5,
            "relative_humidity_2m": 60,
            "apparent_temperature": 28.5,
            "weather_code": 0,
            "wind_speed_10m": 9.5,
        },
        "daily": {
            "time": ["2026-10-08", "2026-10-09"],
            "temperature_2m_max": [31.0, 30.5],
            "weather_code": [0, 1],
        },
    }

    recorded_headers = {}

    class FakeAsyncClient:
        def __init__(self, timeout=5.0, headers=None):
            nonlocal recorded_headers
            recorded_headers = headers or {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            return fake_response

    with patch("httpx.AsyncClient", FakeAsyncClient):
        result = await client._fetch_open_meteo(latitude=18.7891, longitude=78.1234, loc_name="18.7891,78.1234")

    assert result is not None
    assert result["data_available"] is True
    assert result["current"]["temp"] == 27.5
    assert "User-Agent" in recorded_headers
    assert recorded_headers["User-Agent"] == "BhoomiMitraAI/1.0 (https://bhoomimitra.org; contact@bhoomimitra.org)"


# ---------------------------------------------------------------------------
# 24. Regression: All providers failing produces honest fallback response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_weather_providers_fail_honest_fallback_pipeline():
    """When both OpenWeather and Open-Meteo fail in production, system returns honest unavailable fallback."""
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient

    farmer = _make_mock_farmer(language="en")
    db = _mock_db_with_location(memory_gps={"latitude": 17.4093, "longitude": 78.6496})

    with patch.object(OpenWeatherClient, "fetch_weather", new_callable=AsyncMock, return_value=None):
        result = await enrich_response_with_weather(
            db, "weather forecast", "", farmer
        )

    assert "Weather forecast is currently unavailable for this location" in result
    assert "1800-180-1551" in result
    assert "🌡️" not in result


# ===========================================================================
# 25. Regression: OpenWeatherMap HTTP 401 triggers fallback without retries
# ===========================================================================

@pytest.mark.asyncio
async def test_openweather_http_401_triggers_fallback_without_retries():
    """
    When OpenWeatherMap returns HTTP 401 Unauthorized (e.g. invalid or inactive key),
    the client must log a clear diagnostic, must NOT retry OWM repeatedly, and must
    immediately fall back to Open-Meteo live weather.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache

    clear_weather_memory_cache()
    client = OpenWeatherClient(
        api_key="invalid_test_api_key",
        api_url="https://api.openweathermap.org/data/2.5/forecast",
    )

    owm_call_count = 0

    class FakeAsyncClient:
        def __init__(self, timeout=5.0, headers=None):
            self.timeout = timeout
            self.headers = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None):
            nonlocal owm_call_count
            resp = MagicMock()
            if "openweathermap.org" in url:
                owm_call_count += 1
                resp.status_code = 401
                resp.json.return_value = {"cod": 401, "message": "Invalid API key."}
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "current": {
                        "temperature_2m": 31.2,
                        "relative_humidity_2m": 58,
                        "apparent_temperature": 33.0,
                        "weather_code": 0,
                        "wind_speed_10m": 11.2,
                    },
                    "daily": {
                        "time": ["2026-10-16", "2026-10-17"],
                        "temperature_2m_max": [33.5, 32.0],
                        "weather_code": [0, 80],
                    },
                }
            return resp

    with patch("httpx.AsyncClient", FakeAsyncClient), \
         patch.object(client, "_get_from_cache", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_set_in_cache", new_callable=AsyncMock):

        data = await client.fetch_weather(latitude=17.4093, longitude=78.6496)

    # 1. OpenWeatherMap was called exactly ONCE (zero repeated retries for 401)
    assert owm_call_count == 1
    # 2. Live fallback to Open-Meteo succeeded
    assert data is not None
    assert data["data_available"] is True
    assert data["is_live"] is True
    assert data["current"]["temp"] == 31.2
    assert "Open-Meteo" in data["source_note"]


# ===========================================================================
# 26. Regression: Open-Meteo HTTP 429 handled safely with bounded backoff
# ===========================================================================

@pytest.mark.asyncio
async def test_open_meteo_http_429_handled_safely_no_retry_storm():
    """
    When Open-Meteo returns HTTP 429 Too Many Requests:
    - If Retry-After is large (> 2s), abort immediately with no retry storm.
    - If Retry-After is small or absent, retry at most once with bounded sleep.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache

    clear_weather_memory_cache()
    # Use configured API key to simulate production/configured mode (avoiding keyless non-prod mock generator)
    client = OpenWeatherClient(api_key="configured_test_key", api_url="")

    # Case A: Retry-After = 60s (large) -> zero retry attempts
    om_call_count = 0

    class FakeClientLargeRetry:
        def __init__(self, timeout=5.0, headers=None):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, params=None):
            nonlocal om_call_count
            om_call_count += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.headers = {"Retry-After": "60"}
            return resp

    with patch("httpx.AsyncClient", FakeClientLargeRetry), \
         patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_get_from_cache", new_callable=AsyncMock, return_value=None):
        data_a = await client.fetch_weather(latitude=17.4093, longitude=78.6496)

    assert om_call_count == 1
    assert data_a is None

    # Case B: Retry-After = 1s -> retries at most once
    from src.weather.openweather_client import clear_open_meteo_cooldown
    clear_open_meteo_cooldown()
    om_retry_count = 0

    class FakeClientSmallRetry:
        def __init__(self, timeout=5.0, headers=None):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, params=None):
            nonlocal om_retry_count
            om_retry_count += 1
            resp = MagicMock()
            if om_retry_count == 1:
                resp.status_code = 429
                resp.headers = {"Retry-After": "1"}
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "current": {
                        "temperature_2m": 28.0,
                        "relative_humidity_2m": 65,
                        "apparent_temperature": 29.5,
                        "weather_code": 1,
                        "wind_speed_10m": 8.0,
                    },
                    "daily": {
                        "time": ["2026-10-16", "2026-10-17"],
                        "temperature_2m_max": [30.0, 29.0],
                        "weather_code": [1, 2],
                    },
                }
            return resp

    with patch("httpx.AsyncClient", FakeClientSmallRetry), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_get_from_cache", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_set_in_cache", new_callable=AsyncMock):

        data_b = await client.fetch_weather(latitude=17.4093, longitude=78.6496)

    assert om_retry_count == 2
    mock_sleep.assert_awaited_once_with(1.0)
    assert data_b is not None
    assert data_b["current"]["temp"] == 28.0


# ===========================================================================
# 27. Regression: Valid fallback data produces complete weather response
# ===========================================================================

@pytest.mark.asyncio
async def test_valid_fallback_data_produces_weather_response():
    """
    Verifies that when OpenWeatherMap fails (401), valid fallback Open-Meteo data
    produces a complete, verified WeatherForecastResponse and localized WhatsApp message.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    from src.weather.service import WeatherService

    clear_weather_memory_cache()
    client = OpenWeatherClient(api_key="bad_key", api_url="https://api.openweathermap.org/data/2.5/forecast")

    open_meteo_live = {
        "location_name": "17.4093,78.6496",
        "latitude": 17.409296,
        "longitude": 78.649556,
        "current": {
            "temp": 29.8,
            "feels_like": 31.4,
            "humidity": 62,
            "wind_speed": 10.5,
            "description": "Clear Sky",
            "condition_code": 800,
        },
        "forecast": [
            {
                "dt_txt": (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00"),
                "temp": 31.0,
                "humidity": 65,
                "description": "Rain",
                "condition_code": 500,
            }
        ],
        "data_available": True,
        "data_freshness_minutes": 0.0,
        "is_live": True,
        "source_note": "Live Weather (Open-Meteo)",
    }

    with patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_fetch_open_meteo", new_callable=AsyncMock, return_value=open_meteo_live), \
         patch.object(client, "_get_from_cache", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_set_in_cache", new_callable=AsyncMock):

        svc = WeatherService(client)
        resp = await svc.get_weather_for_query(latitude=17.409296, longitude=78.649556)

        assert resp.data_available is True
        assert resp.is_live is True
        assert resp.current.temp == 29.8
        assert resp.latitude == 17.409296
        assert resp.longitude == 78.649556

        # Formatted reply in Telugu
        reply_te = svc.format_whatsapp_reply(resp, language="te")
        assert "వాతావరణ సమాచారం" in reply_te
        assert "29.8°C" in reply_te
        assert "ఓపెన్-మెటియో (లైవ్)" in reply_te
        assert "వర్షం పడే అవకాశం ఉంది" in reply_te


# ===========================================================================
# 28. Regression: Cached data freshness is checked and disclosed
# ===========================================================================

@pytest.mark.asyncio
async def test_cached_data_freshness_checked_and_disclosed():
    """
    Verifies that when both live providers fail:
    1. Recent cached data (< 6 hours) is reused safely.
    2. Cached data freshness is computed accurately (data_freshness_minutes).
    3. WhatsApp message clearly discloses that the data is cached from X minutes ago, not live.
    4. Data older than 6 hours is NOT presented as current; safe fallback is used instead.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    from src.weather.service import WeatherService

    clear_weather_memory_cache()
    client = OpenWeatherClient(api_key="bad_key", api_url="")

    # A. 45-minute-old cached data
    stale_payload = {
        "location_name": "Ghatkesar Farm",
        "latitude": 17.409296,
        "longitude": 78.649556,
        "current": {
            "temp": 30.0,
            "feels_like": 32.0,
            "humidity": 60,
            "wind_speed": 9.0,
            "description": "Clear Sky",
            "condition_code": 800,
        },
        "forecast": [],
        "data_available": True,
        "_cached_at": (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat(),
        "is_live": True,
        "source_note": "OpenWeatherMap API (Live)",
    }

    # Populate in-memory cache directly
    cache_key = client._cache_key(17.409296, 78.649556, None, None)
    from src.weather.openweather_client import _MEMORY_CACHE
    _MEMORY_CACHE[cache_key] = (datetime.now(timezone.utc) - timedelta(minutes=45), stale_payload)

    # Simulate live providers failing (OWM 401, Open-Meteo 429)
    with patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_fetch_open_meteo", new_callable=AsyncMock, return_value=None):

        svc = WeatherService(client)
        resp = await svc.get_weather_for_query(latitude=17.409296, longitude=78.649556)

        assert resp.data_available is True
        assert resp.is_live is False
        assert resp.data_freshness_minutes is not None
        assert 44.0 <= resp.data_freshness_minutes <= 46.0

        # Formatted reply explicitly identifies stale data and discloses age
        reply_te = svc.format_whatsapp_reply(resp, language="te")
        assert "45 నిమిషాల క్రితం" in reply_te
        assert "తాజా లైవ్ కాదు" in reply_te

        reply_en = svc.format_whatsapp_reply(resp, language="en")
        assert "45 mins ago" in reply_en
        assert "not live" in reply_en

    # B. Cache older than 6 hours (e.g. 8 hours) -> NOT reused, returns honest unavailable
    clear_weather_memory_cache()
    very_old_payload = dict(stale_payload)
    very_old_payload["_cached_at"] = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    _MEMORY_CACHE[cache_key] = (datetime.now(timezone.utc) - timedelta(hours=8), very_old_payload)

    with patch.object(client, "_call_api", new_callable=AsyncMock, return_value=None), \
         patch.object(client, "_fetch_open_meteo", new_callable=AsyncMock, return_value=None):

        resp_expired = await svc.get_weather_for_query(latitude=17.409296, longitude=78.649556)
        assert resp_expired.data_available is False
        reply_expired = svc.format_whatsapp_reply(resp_expired, language="te")
        assert "వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు" in reply_expired


# ===========================================================================
# 29. Regression: Both providers failing returns safe Telugu fallback
# ===========================================================================

@pytest.mark.asyncio
async def test_both_providers_failing_returns_safe_telugu_fallback():
    """
    When OpenWeatherMap returns 401 and Open-Meteo returns 429 with no cache,
    enrich_response_with_weather returns the safe, honest Telugu fallback.
    """
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import clear_weather_memory_cache

    clear_weather_memory_cache()
    farmer = _make_mock_farmer(language="te")
    db = _mock_db_with_location(memory_gps={"latitude": 17.409296, "longitude": 78.649556})

    class BothFailingClient:
        def __init__(self, timeout=5.0, headers=None):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, params=None):
            resp = MagicMock()
            if "openweathermap.org" in url:
                resp.status_code = 401
            else:
                resp.status_code = 429
                resp.headers = {"Retry-After": "60"}
            return resp

    with patch("httpx.AsyncClient", BothFailingClient), \
         patch("src.config.get_settings") as mock_settings:
        s = MagicMock()
        s.openweather_api_key = "configured_live_key"
        s.openweather_api_url = "https://api.openweathermap.org/data/2.5/forecast"
        s.weather_cache_ttl_seconds = 1800
        s.openweather_api_timeout_seconds = 5.0
        s.redis_url = "redis://localhost:6379/0"
        s.app_env = "production"
        mock_settings.return_value = s

        result = await enrich_response_with_weather(
            db, "రేపు మా ఊర్లో వర్షం పడుతుందా?", "", farmer
        )

    assert "వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు" in result
    assert "1800-180-1551" in result


# ===========================================================================
# 30. Regression: GPS coordinates passed correctly through pipeline
# ===========================================================================

@pytest.mark.asyncio
async def test_gps_coordinates_passed_correctly_through_pipeline():
    """
    Verifies that coordinates 17.409296, 78.649556 supplied via FarmerMemory
    or query text are extracted accurately and forwarded to weather clients.
    """
    from src.weather.service import enrich_response_with_weather
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache

    clear_weather_memory_cache()
    farmer = _make_mock_farmer(language="te")
    db = _mock_db_with_location(memory_gps={"latitude": 17.409296, "longitude": 78.649556})

    captured_coords = {}

    async def mock_fetch(self, latitude=None, longitude=None, district=None, state=None):
        nonlocal captured_coords
        captured_coords["lat"] = latitude
        captured_coords["lon"] = longitude
        return {
            "location_name": f"{latitude:.4f},{longitude:.4f}",
            "latitude": latitude,
            "longitude": longitude,
            "current": {
                "temp": 32.0,
                "feels_like": 34.0,
                "humidity": 55,
                "wind_speed": 12.0,
                "description": "Clear Sky",
                "condition_code": 800,
            },
            "forecast": [],
            "data_available": True,
            "data_freshness_minutes": 0.0,
            "is_live": True,
            "source_note": "Live Weather (Open-Meteo)",
        }

    with patch.object(OpenWeatherClient, "fetch_weather", mock_fetch):
        # 1. Coordinate resolution from memory GPS
        res = await enrich_response_with_weather(db, "వాతావరణం ఎలా ఉంది?", "", farmer)
        assert captured_coords["lat"] == 17.409296
        assert captured_coords["lon"] == 78.649556
        assert "32.0°C" in res

        # 2. Coordinate resolution from query text directly
        empty_db = _mock_db_with_location(memory_gps=None, profile_district=None)
        await enrich_response_with_weather(empty_db, "వాతావరణం 17.409296, 78.649556 వద్ద ఎలా ఉంది?", "", farmer)
        assert round(captured_coords["lat"], 6) == 17.409296
        assert round(captured_coords["lon"], 6) == 78.649556


# ------------------------------------------------------------------
# Regressions: Secret Redaction & Rate Limit Cooldown Protection
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_secret_redaction_httpx_logger_never_exposes_api_key(caplog):
    """
    REGRESSION: Ensure httpx request logging never exposes OPENWEATHER_API_KEY / appid
    in logs, while preserving provider hostname, path, and HTTP status code.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    from src.core.logging import SecretRedactingFilter, apply_secret_redacting_filter
    import logging
    import httpx

    clear_weather_memory_cache()
    fake_secret = "TEST_SECRET_KEY_EXPOSURE_CHECK_XYZ123456"
    test_client = OpenWeatherClient(api_key=fake_secret, api_url="https://api.openweathermap.org/data/2.5/forecast")

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.INFO)
    apply_secret_redacting_filter(httpx_logger)

    def mock_handler(request: httpx.Request):
        assert fake_secret in str(request.url)  # Secret is sent over HTTP wire
        return httpx.Response(401, json={"cod": 401, "message": "Invalid API key."})

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with caplog.at_level(logging.INFO):
            res = await test_client._call_api(latitude=17.409296, longitude=78.649556, district=None, state=None)
            assert res is None

    # Verify that fake_secret NEVER appears anywhere in any log record or captured log text
    assert fake_secret not in caplog.text
    for record in caplog.records:
        assert fake_secret not in record.getMessage()

    # Verify safe diagnostics are present
    assert "api.openweathermap.org" in caplog.text
    assert "/data/2.5/forecast" in caplog.text
    assert "401" in caplog.text


@pytest.mark.asyncio
async def test_secret_redaction_exception_message_never_exposes_api_key(caplog):
    """
    REGRESSION: Ensure exceptions containing sensitive query parameters mask raw secrets as '***'.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    import logging
    import httpx

    clear_weather_memory_cache()
    fake_secret = "EXCEPTION_SECRET_KEY_CHECK_98765"
    test_client = OpenWeatherClient(api_key=fake_secret, api_url="https://api.openweathermap.org/data/2.5/forecast")

    async def mock_get(*args, **kwargs):
        raise httpx.RequestError(f"Connection failed to https://api.openweathermap.org/data/2.5/forecast?lat=17.4&appid={fake_secret}")

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        with caplog.at_level(logging.INFO):
            res = await test_client._call_api(latitude=17.409296, longitude=78.649556, district=None, state=None)
            assert res is None

    assert fake_secret not in caplog.text
    assert "appid=***" in caplog.text


@pytest.mark.asyncio
async def test_openweather_401_logs_provider_response_without_retrying(caplog):
    """
    REGRESSION: HTTP 401 Unauthorized logs safe provider details without retrying or guessing.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    import logging
    import httpx

    clear_weather_memory_cache()
    fake_secret = "KEY_PENDING_ACTIVATION_CHECK_111"
    test_client = OpenWeatherClient(api_key=fake_secret, api_url="https://api.openweathermap.org/data/2.5/forecast")

    call_count = 0
    def mock_handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"cod": 401, "message": "Your API key is not activated yet."})

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with caplog.at_level(logging.INFO):
            res = await test_client._call_api(latitude=17.409296, longitude=78.649556, district=None, state=None)
            assert res is None

    assert call_count == 1  # No repeated retries
    assert "Your API key is not activated yet" in caplog.text
    assert fake_secret not in caplog.text


@pytest.mark.asyncio
async def test_open_meteo_429_persisted_activates_cooldown_preventing_storms(caplog):
    """
    REGRESSION: When Open-Meteo 429 persists after 1 bounded retry, activates cooldown to prevent retry storms.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    import logging
    import httpx

    clear_weather_memory_cache()
    test_client = OpenWeatherClient(api_key="", api_url="")

    call_count = 0
    def mock_handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "0.1"})

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with caplog.at_level(logging.INFO):
            # First call: makes 1 initial request + 1 bounded retry (call_count == 2), both return 429
            res1 = await test_client._fetch_open_meteo(latitude=17.409, longitude=78.649, loc_name="Farm")
            assert res1 is None
            assert call_count == 2
            assert "Open-Meteo HTTP 429 persisted after bounded retry" in caplog.text

            # Immediate subsequent call: should be blocked by cooldown without making any new HTTP requests
            res2 = await test_client._fetch_open_meteo(latitude=17.409, longitude=78.649, loc_name="Farm")
            assert res2 is None
            assert call_count == 2  # Still 2, no new HTTP request was attempted
            assert "Open-Meteo is currently in rate-limit cooldown" in caplog.text


@pytest.mark.asyncio
async def test_open_meteo_retry_after_large_activates_cooldown_immediately(caplog):
    """
    REGRESSION: If Open-Meteo returns Retry-After > 2.0s, aborts retry immediately and sets cooldown.
    """
    from src.weather.openweather_client import OpenWeatherClient, clear_weather_memory_cache
    import logging
    import httpx

    clear_weather_memory_cache()
    test_client = OpenWeatherClient(api_key="", api_url="")

    call_count = 0
    def mock_handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "60"})

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with caplog.at_level(logging.INFO):
            res = await test_client._fetch_open_meteo(latitude=17.409, longitude=78.649, loc_name="Farm")
            assert res is None
            assert call_count == 1  # Did NOT retry because 60s > 2.0s
            assert "Retry-After (60.0s) exceeds threshold" in caplog.text

            # Next call during cooldown is blocked
            res2 = await test_client._fetch_open_meteo(latitude=17.409, longitude=78.649, loc_name="Farm")
            assert res2 is None
            assert call_count == 1
