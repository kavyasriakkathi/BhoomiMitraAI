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
    "wind", "humidity", "climate", "degree", "hot", "cold", "will it rain",
    "precipitation", "cloudy", "storm", "thunderstorm", "showers", "sun", "sunny",
}
WEATHER_KEYWORDS_TE = {
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "వానలు", "కురుస్తుందా",
    "పడుతుంది", "పడుతుందా", "కురుస్తుంది", "ఉష్ణోగ్రత", "గాలి", "తేమ", "ఎండ",
    "చలి", "వాతావరణ అంచనా", "మంచు", "తుఫాను", "జల్లులు", "మేఘాలు",
}

# Known Telangana & Andhra Pradesh Districts/Cities for Query Extraction
_KNOWN_DISTRICTS = {
    # Telangana
    "warangal": "Warangal",
    "hanamkonda": "Warangal",
    "వరంగల్": "Warangal",
    "హనుమకొండ": "Warangal",
    "karimnagar": "Karimnagar",
    "కరీంనగర్": "Karimnagar",
    "khammam": "Khammam",
    "ఖమ్మం": "Khammam",
    "guntur": "Guntur",
    "గుంటూరు": "Guntur",
    "nizamabad": "Nizamabad",
    "నిజామాబాద్": "Nizamabad",
    "nalgonda": "Nalgonda",
    "నల్గొండ": "Nalgonda",
    "mahabubnagar": "Mahabubnagar",
    "మహబూబ్‌నగర్": "Mahabubnagar",
    "medak": "Medak",
    "మెదక్": "Medak",
    "adilabad": "Adilabad",
    "ఆదిలాబాద్": "Adilabad",
    "rangareddy": "Rangareddy",
    "రంగారెడ్డి": "Rangareddy",
    "hyderabad": "Hyderabad",
    "హైదరాబాద్": "Hyderabad",
    "siddipet": "Siddipet",
    "సిద్దిపేట": "Siddipet",
    "suryapet": "Suryapet",
    "సూర్యాపేట": "Suryapet",
    "jagtial": "Jagtial",
    "జగిత్యాల": "Jagtial",
    "korutla": "Jagtial",
    "కోరుట్ల": "Jagtial",
    "mancherial": "Mancherial",
    "మంచిర్యాల": "Mancherial",
    "bhadradri": "Bhadradri Kothagudem",
    "భద్రాద్రి": "Bhadradri Kothagudem",
    "kothagudem": "Bhadradri Kothagudem",
    "కొత్తగూడెం": "Bhadradri Kothagudem",
    "vikarabad": "Vikarabad",
    "వికారాబాద్": "Vikarabad",
    "sangareddy": "Sangareddy",
    "సంగారెడ్డి": "Sangareddy",
    "kamareddy": "Kamareddy",
    "కామారెడ్డి": "Kamareddy",
    "rajanna sircilla": "Rajanna Sircilla",
    "సిరిసిల్ల": "Rajanna Sircilla",
    "sircilla": "Rajanna Sircilla",
    "peddapalli": "Peddapalli",
    "పెద్దపల్లి": "Peddapalli",
    "wanaparthy": "Wanaparthy",
    "వనపర్తి": "Wanaparthy",
    "jogulamba": "Jogulamba Gadwal",
    "గద్వాల": "Jogulamba Gadwal",
    "gadwal": "Jogulamba Gadwal",
    "nagarkurnool": "Nagarkurnool",
    "నాగర్‌కర్నూల్": "Nagarkurnool",
    "narayanpet": "Narayanpet",
    "నారాయణపేట": "Narayanpet",
    "mulugu": "Mulugu",
    "ములుగు": "Mulugu",
    "jayashankar": "Jayashankar Bhupalpally",
    "భూపాలపల్లి": "Jayashankar Bhupalpally",
    "bhupalpally": "Jayashankar Bhupalpally",
    "janagaon": "Jangaon",
    "జనగామ": "Jangaon",
    "jangaon": "Jangaon",
    "yadadri": "Yadadri Bhuvanagiri",
    "యాదాద్రి": "Yadadri Bhuvanagiri",
    "bhuvanagiri": "Yadadri Bhuvanagiri",
    "భూవనగిరి": "Yadadri Bhuvanagiri",
    "asifabad": "Komaram Bheem Asifabad",
    "ఆసిఫాబాద్": "Komaram Bheem Asifabad",
    "nirmal": "Nirmal",
    "నిర్మల్": "Nirmal",
    "medchal": "Medchal-Malkajgiri",
    "మేడ్చల్": "Medchal-Malkajgiri",

    # Andhra Pradesh
    "krishna": "Krishna",
    "కృష్ణా": "Krishna",
    "vijayawada": "Krishna",
    "విజయవాడ": "Krishna",
    "kurnool": "Kurnool",
    "కర్నూలు": "Kurnool",
    "anantapur": "Anantapur",
    "అనంతపురం": "Anantapur",
    "kadapa": "Kadapa",
    "కడప": "Kadapa",
    "ysr": "Kadapa",
    "nellore": "Nellore",
    "నెల్లూరు": "Nellore",
    "prakasam": "Prakasam",
    "ప్రకాశం": "Prakasam",
    "ongole": "Prakasam",
    "ఒంగోలు": "Prakasam",
    "chittoor": "Chittoor",
    "చిత్తూరు": "Chittoor",
    "tirupati": "Tirupati",
    "తిరుపతి": "Tirupati",
    "visakhapatnam": "Visakhapatnam",
    "విశాఖపట్నం": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "godavari": "Godavari",
    "గోదావరి": "Godavari",
    "kakinada": "Kakinada",
    "కాకినాడ": "Kakinada",
    "rajahmundry": "East Godavari",
    "రాజమండ్రి": "East Godavari",
    "eluru": "Eluru",
    "ఏలూరు": "Eluru",
    "srikakulam": "Srikakulam",
    "శ్రీకాకుళం": "Srikakulam",
    "vizianagaram": "Vizianagaram",
    "విజయనగరం": "Vizianagaram",
    "bapatla": "Bapatla",
    "బాపట్ల": "Bapatla",
    "palnadu": "Palnadu",
    "పల్నాడు": "Palnadu",
    "narasaraopet": "Palnadu",
    "నరసరావుపేట": "Palnadu",
    "nandyal": "Nandyal",
    "నంద్యాల": "Nandyal",
    "machilipatnam": "Krishna",
    "మచిలీపట్నం": "Krishna",
    "konaseema": "Dr. B.R. Ambedkar Konaseema",
    "కోనసీమ": "Dr. B.R. Ambedkar Konaseema",
    "amalapuram": "Dr. B.R. Ambedkar Konaseema",
    "అమలాపురం": "Dr. B.R. Ambedkar Konaseema",
    "anakapalli": "Anakapalli",
    "అనకాపల్లి": "Anakapalli",
    "alluri": "Alluri Sitharama Raju",
    "అల్లూరి": "Alluri Sitharama Raju",
    "parvathipuram": "Parvathipuram Manyam",
    "పార్వతీపురం": "Parvathipuram Manyam",
    "sri sathya sai": "Sri Sathya Sai",
    "పుట్టపర్తి": "Sri Sathya Sai",
    "puttaparthi": "Sri Sathya Sai",
    "annamayya": "Annamayya",
    "అన్నమయ్య": "Annamayya",
    "rayachoty": "Annamayya",
    "రాయచోటి": "Annamayya",
}


def _extract_district_from_query(query_text: str) -> Optional[str]:
    """Extract known district or city from farmer query in English or Telugu."""
    q = query_text.lower()
    for kw, dist_name in _KNOWN_DISTRICTS.items():
        if kw in q:
            return dist_name
    return None


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
    "no_data": "ℹ️ గమనిక: ఈ ప్రాంతానికి సంబంధించిన వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు. దయచేసి స్థానిక వాతావరణ కేంద్రం లేదా కిసాన్ కాల్ సెంటర్ (1800-180-1551) ను సంప్రదించండి.",
    "ask_location": "📍 మీ పంటలకు సంబంధించిన ఖచ్చితమైన వాతావరణ సమాచారం కోసం దయచేసి మీ జిల్లా లేదా ప్రాంతం పేరును తెలపండి (ఉదాహరణకు: వరంగల్, గుంటూరు).",
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
    "no_data": "ℹ️ Note: Weather forecast is currently unavailable for this location. Please check local agromet advisories or the Kisan Call Centre (1800-180-1551).",
    "ask_location": "📍 Please provide your district or area name (e.g., Warangal, Guntur) to get accurate weather forecast information for your crops.",
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
    Detect weather-forecast intent in the farmer's query or conversational follow-up.
    If detected, append a formatted weather forecast block to the AI response.

    Always returns the original ai_response unchanged if:
    - No weather intent is detected
    - Any unhandled error occurs
    """
    query_lower = query_text.lower()
    language = getattr(farmer, "preferred_language", "en") or "en"
    labels = _TE_LABELS if language == "te" else _EN_LABELS

    # Step 1: Detect weather intent
    has_weather_intent = any(kw in query_lower for kw in WEATHER_KEYWORDS_EN) or any(kw in query_text for kw in WEATHER_KEYWORDS_TE)

    # Also detect if farmer just provided a district name as a follow-up to a previous weather question
    query_district = _extract_district_from_query(query_text)
    if not has_weather_intent and query_district:
        ai_lower = ai_response.lower()
        if any(kw in ai_lower for kw in WEATHER_KEYWORDS_EN) or any(kw in ai_response for kw in WEATHER_KEYWORDS_TE):
            has_weather_intent = True

    if not has_weather_intent:
        return ai_response

    # Step 2: Resolve Location (Priority-ordered: Query District -> GPS -> Profile District -> Memory District)
    latitude = None
    longitude = None
    district = query_district
    state = None

    try:
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        from src.memory.models import FarmerMemory

        # 1. Check FarmerMemory for GPS coordinates (only if not explicit query district override)
        if not district:
            memory_result = await db.execute(
                select(FarmerMemory).where(FarmerMemory.farmer_id == farmer.id)
            )
            memory = memory_result.scalar_one_or_none()

            if memory and memory.gps_coordinates:
                gps_coords = memory.gps_coordinates
                try:
                    lat = float(gps_coords.get("latitude") or 0.0)
                    lon = float(gps_coords.get("longitude") or 0.0)
                    if lat != 0.0 and lon != 0.0:
                        latitude = lat
                        longitude = lon
                except (ValueError, TypeError):
                    pass

            # 2. Check FarmerProfile for district/state
            if not district:
                profile_result = await db.execute(
                    select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
                )
                profile = profile_result.scalar_one_or_none()

                if profile and profile.district:
                    district = profile.district.strip()
                    state = profile.state.strip() if profile.state else None

            # 3. Check FarmerMemory for district/state if profile has none
            if not district and memory and memory.district:
                district = memory.district.strip()
                state = memory.state.strip() if memory.state else None

            # 4. Check FarmerMemory village mapping if district still not set
            if not district and memory and memory.village:
                district = _extract_district_from_query(memory.village)
                if district and not state:
                    state = memory.state.strip() if memory.state else "Telangana"

    except Exception as loc_err:
        logger.warning(f"[WEATHER ENRICH] Failed to resolve farmer location: {loc_err}")

    # If no location information is resolved, ask for district
    if (latitude is None or longitude is None) and not district:
        logger.info("[WEATHER ENRICH] Weather intent detected but no location resolved. Appending location prompt.")
        if "location" not in ai_response.lower() and "ప్రాంతం" not in ai_response and "జిల్లా" not in ai_response and "district" not in ai_response.lower():
            return ai_response + "\n\n" + labels["ask_location"]
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

        if weather_data and weather_data.data_available:
            logger.info(f"[WEATHER ENRICH] Appending weather data for location '{weather_data.location_name}'.")
            weather_block = svc.format_whatsapp_reply(weather_data, language=language)
            return ai_response + "\n\n" + weather_block

        logger.info("[WEATHER ENRICH] Weather data unavailable. Appending honest fallback.")
        return ai_response + "\n\n" + labels["no_data"]

    except Exception as exc:
        logger.warning(f"[WEATHER ENRICH] Weather enrichment failed: {exc}")
        return ai_response
