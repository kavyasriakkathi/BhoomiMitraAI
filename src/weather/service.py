"""
BhoomiMitra AI — Weather Service

Business logic for processing weather queries and forecast templates.
Integrates directly into the AI pipeline (ai/service.py).
"""
import re
from typing import Optional, List
from datetime import datetime, timedelta

from src.core.logging import logger
from src.weather.openweather_client import OpenWeatherClient
from src.weather.schemas import (
    WeatherCondition,
    WeatherForecastItem,
    WeatherForecastResponse,
)

# ------------------------------------------------------------------
# Weather Intent Keywords
# ------------------------------------------------------------------
WEATHER_KEYWORDS_EN = {
    "weather", "forecast", "rain", "raining", "rainy", "temperature",
    "wind", "humidity", "climate", "degree", "hot", "cold", "will it rain"
}
WEATHER_KEYWORDS_TE = {
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "కురుస్తుందా",
    "ఉష్ణోగ్రత", "గాలి", "తేమ", "ఎండ", "చలి", "వాతావరణ అంచనా", "మంచు"
}

# ------------------------------------------------------------------
# Static Labels for Telugu / English Replies
# ------------------------------------------------------------------
_TE_LABELS = {
    "title": "🌡️ వాతావరణ సమాచారం ({location})",
    "temp": "ఉష్ణోగ్రత",
    "feels_like": "అనిపిస్తుంది",
    "wind": "గాలి వేగం",
    "humidity": "తేమ (Humidity)",
    "condition": "వాతావరణం",
    "source_live": "ఓపెన్వెదర్ (లైవ్)",
    "source_local": "స్థానిక వాతావరణ డేటా",
    "rain_alert": "🌧️ రేపటి అంచనా: మీ ప్రాంతంలో వర్షం పడే అవకాశం ఉంది. దయచేసి పంటలపై తగిన రక్షణ చర్యలు తీసుకోండి.",
    "clear_alert": "☀️ రేపటి అంచనా: వాతావరణం పొడిగా మరియు అనుకూలంగా ఉంటుంది.",
    "no_data": "క్షమించండి, ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు.",
}

_EN_LABELS = {
    "title": "🌡️ Weather Information ({location})",
    "temp": "Temperature",
    "feels_like": "Feels Like",
    "wind": "Wind Speed",
    "humidity": "Humidity",
    "condition": "Condition",
    "source_live": "OpenWeather (Live)",
    "source_local": "Local Weather Data",
    "rain_alert": "🌧️ Tomorrow's Forecast: Rain is expected in your area. Please take necessary protective measures for your crops.",
    "clear_alert": "☀️ Tomorrow's Forecast: Weather is expected to be clear/partly cloudy and dry.",
    "no_data": "Sorry, weather information is currently unavailable.",
}


class WeatherService:
    def __init__(self, client: OpenWeatherClient):
        self.client = client

    # ------------------------------------------------------------------
    # Public: Query forecast data
    # ------------------------------------------------------------------

    async def get_weather_for_query(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        district: Optional[str] = None,
        state: Optional[str] = None,
    ) -> WeatherForecastResponse:
        """Query weather forecast for a location, returning a validated response."""
        data = await self.client.fetch_weather(
            latitude=latitude,
            longitude=longitude,
            district=district,
            state=state,
        )

        if not data or not data.get("data_available"):
            return WeatherForecastResponse(
                location_name=district or "Unknown",
                current=WeatherCondition(
                    temp=0.0,
                    feels_like=0.0,
                    humidity=0,
                    wind_speed=0.0,
                    description="Unknown",
                    condition_code=800,
                ),
                forecast=[],
                data_available=False,
                source_note="No weather provider available.",
                is_live=False,
            )

        # Parse normalized dictionary to schema responses
        current_data = data["current"]
        forecast_items = []
        for f in data.get("forecast", []):
            forecast_items.append(WeatherForecastItem(
                dt_txt=f["dt_txt"],
                temp=f["temp"],
                humidity=f["humidity"],
                description=f["description"],
                condition_code=f["condition_code"],
            ))

        return WeatherForecastResponse(
            location_name=data["location_name"],
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            current=WeatherCondition(
                temp=current_data["temp"],
                feels_like=current_data["feels_like"],
                humidity=current_data["humidity"],
                wind_speed=current_data["wind_speed"],
                description=current_data["description"],
                condition_code=current_data["condition_code"],
            ),
            forecast=forecast_items,
            data_available=True,
            source_note=data["source_note"],
            is_live=data["is_live"],
        )

    # ------------------------------------------------------------------
    # Public: WhatsApp message formatter
    # ------------------------------------------------------------------

    def format_whatsapp_reply(self, response: WeatherForecastResponse, language: str = "en") -> str:
        """Format the forecast response into a friendly WhatsApp text block."""
        labels = _TE_LABELS if language == "te" else _EN_LABELS

        if not response.data_available:
            return labels["no_data"]

        # Translate weather condition description for Telugu
        condition_desc = response.current.description
        if language == "te":
            condition_desc = self.translate_condition(response.current.condition_code, condition_desc)

        # Determine rain forecast for tomorrow
        tomorrow_date = (datetime.utcnow() + timedelta(days=1)).date()
        will_rain_tomorrow = False

        for f_item in response.forecast:
            try:
                # Parse "YYYY-MM-DD HH:MM:SS"
                f_date = datetime.strptime(f_item.dt_txt.strip(), "%Y-%m-%d %H:%M:%S").date()
                if f_date == tomorrow_date:
                    # Condition code in the 5xx (Rain) or 2xx (Thunderstorm) range indicates rain
                    if 200 <= f_item.condition_code < 600:
                        will_rain_tomorrow = True
                        break
            except Exception:
                continue

        tomorrow_alert = labels["rain_alert"] if will_rain_tomorrow else labels["clear_alert"]

        lines = [
            labels["title"].format(location=response.location_name),
            f"\n🌡️ {labels['temp']}: {response.current.temp:.1f}°C ({labels['feels_like']}: {response.current.feels_like:.1f}°C)",
            f"☁️ {labels['condition']}: {condition_desc}",
            f"💧 {labels['humidity']}: {response.current.humidity}%",
            f"💨 {labels['wind']}: {response.current.wind_speed:.1f} km/h",
            f"\n📅 {tomorrow_alert}",
            f"\n📡 {labels['source_live'] if response.is_live else labels['source_local']}"
        ]

        return "\n".join(lines)

    @staticmethod
    def translate_condition(code: int, default_desc: str) -> str:
        """Map OpenWeatherMap condition code to friendly Telugu description."""
        if 200 <= code < 300:
            return "ఉరుములతో కూడిన వర్షం (Thunderstorm)"
        if 300 <= code < 400:
            return "చిరుజల్లులు (Drizzle)"
        if 500 <= code < 600:
            return "వర్షం (Rain)"
        if 600 <= code < 700:
            return "మంచు (Snow)"
        if 700 <= code < 800:
            return "పొగమంచు (Mist/Fog)"
        if code == 800:
            return "ఆకాశం నిర్మలంగా ఉంది (Clear Sky)"
        if 800 < code < 900:
            return "పాక్షికంగా మేఘావృతమై ఉంది (Cloudy)"
        return default_desc


# ------------------------------------------------------------------
# Pipeline integration function — mirrors enrich_response_with_market_prices()
# Called from ai/service.py inside a try/except block.
# ------------------------------------------------------------------

async def enrich_response_with_weather(
    db,
    query_text: str,
    ai_response: str,
    farmer,
) -> str:
    """
    Detect weather-forecast intent in the farmer's query.
    If detected, append a formatted weather forecast block to the AI response.

    Always returns the original ai_response unchanged if:
    - No weather intent is detected
    - No location information is resolved
    - Any error occurs
    """
    query_lower = query_text.lower()

    # Step 1: Detect weather intent
    has_weather_intent = any(kw in query_lower for kw in WEATHER_KEYWORDS_EN)
    if not has_weather_intent:
        has_weather_intent = any(kw in query_lower for kw in WEATHER_KEYWORDS_TE)

    if not has_weather_intent:
        return ai_response

    # Step 2: Resolve Location (Priority-ordered)
    latitude = None
    longitude = None
    district = None
    state = None
    language = getattr(farmer, "preferred_language", "en") or "en"

    try:
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        from src.memory.models import FarmerMemory

        # 1. Check FarmerMemory for GPS coordinates
        memory_result = await db.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer.id)
        )
        memory = memory_result.scalar_one_or_none()

        if memory and memory.gps_coordinates:
            gps_coords = memory.gps_coordinates
            # Validate coordinates are floats
            try:
                lat = float(gps_coords.get("latitude") or 0.0)
                lon = float(gps_coords.get("longitude") or 0.0)
                if lat != 0.0 and lon != 0.0:
                    latitude = lat
                    longitude = lon
            except (ValueError, TypeError):
                pass

        # 2. Check FarmerProfile for district/state
        profile_result = await db.execute(
            select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
        )
        profile = profile_result.scalar_one_or_none()

        if profile and profile.district:
            district = profile.district
            state = profile.state

        # 3. Check FarmerMemory for district/state if profile has none
        if not district and memory and memory.district:
            district = memory.district
            state = memory.state

        # 4. Fallback: Parse query text for known district names (optional future enhancement, skip for now)

    except Exception as loc_err:
        logger.warning(f"[WEATHER ENRICH] Failed to resolve farmer location: {loc_err}")

    # If no location information is resolved, return unchanged.
    if (latitude is None or longitude is None) and not district:
        logger.info("[WEATHER ENRICH] Weather intent detected but no location resolved. Bypassing.")
        return ai_response

    # Step 3: Fetch Weather Forecast
    try:
        from src.config import get_settings
        settings = get_settings()

        client = OpenWeatherClient(
            api_key=settings.openweather_api_key,
            api_url=settings.openweather_api_url,
            cache_ttl_seconds=settings.weather_cache_ttl_seconds,
        )
        svc = WeatherService(client)

        weather_data = await svc.get_weather_for_query(
            latitude=latitude,
            longitude=longitude,
            district=district,
            state=state,
        )

        if weather_data.data_available:
            logger.info(f"[WEATHER ENRICH] Appending weather data for location '{weather_data.location_name}'.")
            weather_block = svc.format_whatsapp_reply(weather_data, language=language)
            return ai_response + "\n\n" + weather_block

        logger.info("[WEATHER ENRICH] Weather data unavailable. Returning original response.")
        return ai_response

    except Exception as exc:
        logger.warning(f"[WEATHER ENRICH] Weather enrichment failed: {exc}")
        return ai_response
