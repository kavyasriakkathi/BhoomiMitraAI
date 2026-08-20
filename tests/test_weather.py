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
    tomorrow_str = (NOW + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")
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
