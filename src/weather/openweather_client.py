"""
BhoomiMitra AI — OpenWeatherMap API Client

Fetches current weather and forecasts from OpenWeatherMap.
Integrates Redis cache and graceful mock fallback for key-less setups.
"""
import json
import hashlib
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import httpx

from src.core.logging import logger
from src.config import get_settings


class OpenWeatherClient:
    """
    Async HTTP client for OpenWeatherMap 5-day / 3-hour forecast API.
    Never raises exceptions — handles errors gracefully and returns None.
    """

    def __init__(self, api_key: str, api_url: str, cache_ttl_seconds: int = 1800, timeout_seconds: float = 5.0):
        self.api_key = api_key.strip() if api_key else ""
        self.api_url = api_url
        self.cache_ttl = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_weather(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        district: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch forecast payload from OpenWeatherMap or Redis cache.
        Returns a dictionary structure matching schemas on success, or None on failure.
        """
        # Resolve target location name for logging/caching
        location_label = ""
        if latitude is not None and longitude is not None:
            location_label = f"{latitude:.4f},{longitude:.4f}"
        elif district:
            location_label = f"{district}, {state or ''}".strip(", ")
        else:
            logger.warning("[WEATHER CLIENT] fetch_weather called with no location details.")
            return None

        # 1. Try Cache First
        cached = await self._get_from_cache(latitude, longitude, district, state)
        if cached is not None:
            logger.info(f"[WEATHER CLIENT] Cache HIT for location='{location_label}'")
            return cached

        # 2. Keyless Fallback Mock Data Generator
        if not self.api_key:
            settings = get_settings()
            if settings.app_env != "production":
                logger.info(
                    f"[WEATHER CLIENT] OPENWEATHER_API_KEY not configured — "
                    f"generating mock weather data for '{location_label}' (Non-prod fallback)."
                )
                mock_data = self._generate_mock_data(latitude, longitude, district, state)
                await self._set_in_cache(latitude, longitude, district, state, mock_data)
                return mock_data
            else:
                logger.warning(
                    f"[WEATHER CLIENT] OPENWEATHER_API_KEY not configured in production. "
                    f"Weather query for '{location_label}' returning None."
                )
                return None

        # 3. Call Live API
        live_data = await self._call_api(latitude, longitude, district, state)
        if live_data is not None:
            await self._set_in_cache(latitude, longitude, district, state, live_data)
            return live_data

        return None

    # ------------------------------------------------------------------
    # Internal: API Call
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
    ) -> Optional[Dict]:
        """Make async GET request to OpenWeatherMap."""
        params = {
            "appid": self.api_key,
            "units": "metric",  # Celsius
        }

        if latitude is not None and longitude is not None:
            params["lat"] = str(latitude)
            params["lon"] = str(longitude)
        elif district:
            # e.g. "Warangal, IN" or "Warangal, Telangana, IN"
            loc = f"{district}, {state}, IN" if state else f"{district}, IN"
            params["q"] = loc
        else:
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                logger.info(f"[WEATHER CLIENT] Calling live OpenWeatherMap API: {params}")
                response = await client.get(self.api_url, params=params)

            if response.status_code != 200:
                logger.warning(
                    f"[WEATHER CLIENT] OpenWeatherMap API returned HTTP {response.status_code}. "
                    "Failed to retrieve weather."
                )
                return None

            data = response.json()
            return self._normalise_response(data)

        except httpx.TimeoutException:
            logger.warning(f"[WEATHER CLIENT] Connection timed out ({self.timeout_seconds}s threshold exceeded).")
            return None
        except httpx.RequestError as exc:
            logger.warning(f"[WEATHER CLIENT] HTTP connection error: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"[WEATHER CLIENT] Unexpected error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal: Normalise OpenWeatherMap JSON Payload
    # ------------------------------------------------------------------

    def _normalise_response(self, raw: dict) -> Optional[Dict]:
        """Convert raw OpenWeatherMap forecast payload to normal JSON dict."""
        try:
            forecast_list = raw.get("list", [])
            if not forecast_list:
                return None

            # Primary (first) element serves as current condition
            current_raw = forecast_list[0]
            current_weather = current_raw.get("weather", [{}])[0]

            normalised = {
                "location_name": raw.get("city", {}).get("name", "Unknown Location"),
                "latitude": float(raw.get("city", {}).get("coord", {}).get("lat", 0.0)),
                "longitude": float(raw.get("city", {}).get("coord", {}).get("lon", 0.0)),
                "current": {
                    "temp": float(current_raw.get("main", {}).get("temp", 0.0)),
                    "feels_like": float(current_raw.get("main", {}).get("feels_like", 0.0)),
                    "humidity": int(current_raw.get("main", {}).get("humidity", 0)),
                    "wind_speed": float(current_raw.get("wind", {}).get("speed", 0.0)) * 3.6,  # Convert m/s to km/h
                    "description": current_weather.get("description", "Clear").title(),
                    "condition_code": int(current_weather.get("id", 800)),
                },
                "forecast": [],
                "data_available": True,
                "is_live": True,
                "source_note": "OpenWeatherMap API (Live)",
            }

            # Map the remaining forecast slots
            for item in forecast_list:
                weather_info = item.get("weather", [{}])[0]
                normalised["forecast"].append({
                    "dt_txt": str(item.get("dt_txt", "")),
                    "temp": float(item.get("main", {}).get("temp", 0.0)),
                    "humidity": int(item.get("main", {}).get("humidity", 0)),
                    "description": weather_info.get("description", "").title(),
                    "condition_code": int(weather_info.get("id", 800)),
                })

            return normalised

        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"[WEATHER CLIENT] Failed to parse OWM response: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal: Caching
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
    ) -> str:
        if latitude is not None and longitude is not None:
            raw = f"weather:latlon:{latitude:.2f}:{longitude:.2f}"
        else:
            raw = f"weather:loc:{(district or '').lower()}:{(state or '').lower()}"
        return "weather:" + hashlib.md5(raw.encode()).hexdigest()[:16]

    async def _get_from_cache(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
    ) -> Optional[Dict]:
        try:
            import redis.asyncio as aioredis
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            key = self._cache_key(latitude, longitude, district, state)
            val = await r.get(key)
            await r.aclose()
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.debug(f"[WEATHER CLIENT] Redis get skipped: {exc}")
        return None

    async def _set_in_cache(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
        data: Dict,
    ) -> None:
        try:
            import redis.asyncio as aioredis
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            key = self._cache_key(latitude, longitude, district, state)
            await r.setex(key, self.cache_ttl, json.dumps(data))
            await r.aclose()
            logger.debug(f"[WEATHER CLIENT] Cached weather for key '{key}' (TTL={self.cache_ttl}s)")
        except Exception as exc:
            logger.debug(f"[WEATHER CLIENT] Redis set skipped: {exc}")

    # ------------------------------------------------------------------
    # Internal: Deterministic Mock Data Fallback
    # ------------------------------------------------------------------

    def _generate_mock_data(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
    ) -> Dict:
        """Deterministic simulated weather data based on location details."""
        loc_name = district or "Selected Farm"
        if latitude is not None and longitude is not None:
            loc_name = f"Farm ({latitude:.2f}, {longitude:.2f})"

        # Generate seed from location name characters to vary weather slightly
        char_sum = sum(ord(c) for c in loc_name)
        base_temp = 25.0 + (char_sum % 10)  # Varies between 25.0°C and 34.0°C
        humidity = 50 + (char_sum % 30)     # Varies between 50% and 79%
        wind = 8.0 + (char_sum % 15) * 0.5  # Varies between 8.0 and 15.0 km/h

        # Determine rain or clear based on location seed
        has_rain_tomorrow = (char_sum % 3 == 0)
        cond_desc = "Moderate Rain" if has_rain_tomorrow else "Partly Cloudy"
        cond_code = 501 if has_rain_tomorrow else 802

        now = datetime.utcnow()

        mock_payload = {
            "location_name": loc_name,
            "latitude": latitude or 17.3850,
            "longitude": longitude or 78.4867,
            "current": {
                "temp": base_temp,
                "feels_like": base_temp + 1.5,
                "humidity": humidity,
                "wind_speed": wind,
                "description": cond_desc,
                "condition_code": cond_code,
            },
            "forecast": [],
            "data_available": True,
            "is_live": False,
            "source_note": "Simulated Weather (Local Fallback)",
        }

        # Build 5 days of forecast slots (every 3 hours)
        for i in range(40):
            slot_time = now + timedelta(hours=i * 3)
            # Create variation in temperature throughout the day (cycles temp)
            hour_factor = abs(12 - slot_time.hour) / 12.0
            temp_var = base_temp - (hour_factor * 6.0)

            # Tomorrow's forecast should indicate rain if the seed matched
            is_tomorrow = (slot_time.date() == (now + timedelta(days=1)).date())
            slot_rain = has_rain_tomorrow and is_tomorrow
            slot_desc = "Light Rain" if slot_rain else "Partly Cloudy"
            slot_code = 500 if slot_rain else 802

            mock_payload["forecast"].append({
                "dt_txt": slot_time.strftime("%Y-%m-%d %H:%M:%S"),
                "temp": round(temp_var, 1),
                "humidity": min(95, humidity + (5 if slot_rain else 0)),
                "description": slot_desc,
                "condition_code": slot_code,
            })

        return mock_payload
