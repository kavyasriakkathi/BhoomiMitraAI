"""
BhoomiMitra AI — OpenWeatherMap API Client

Fetches current weather and forecasts from OpenWeatherMap.
Integrates Redis cache, in-memory process fallback, and graceful mock fallback for key-less setups.
Handles HTTP 401 without retries and respects Open-Meteo HTTP 429 rate limits with bounded backoff.
"""
import json
import hashlib
import copy
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

import httpx

from src.core.logging import logger
from src.config import get_settings


# Module-level in-memory cache to guarantee caching even when Redis is down or unconfigured
_MEMORY_CACHE: Dict[str, Tuple[datetime, Dict]] = {}


def clear_weather_memory_cache() -> None:
    """Helper to clear process in-memory weather cache (primarily for tests)."""
    _MEMORY_CACHE.clear()


class OpenWeatherClient:
    """
    Async HTTP client for OpenWeatherMap 5-day / 3-hour forecast API with Open-Meteo live fallback.
    Never raises exceptions — handles errors gracefully and returns None.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        cache_ttl_seconds: int = 1800,
        timeout_seconds: float = 5.0,
        max_stale_seconds: int = 21600,
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.api_url = api_url
        self.cache_ttl = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.max_stale_seconds = max_stale_seconds

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
        Fetch forecast payload from OpenWeatherMap, Open-Meteo, or Redis/in-memory cache.
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

        # 1. Try Fresh Cache First (within normal TTL, e.g. 30 mins)
        cached = await self._get_from_cache(latitude, longitude, district, state, allow_stale=False)
        if cached is not None:
            freshness = cached.get("data_freshness_minutes", 0.0)
            logger.info(f"[WEATHER CLIENT] Cache HIT for location='{location_label}' (freshness={freshness}m)")
            return cached

        # 2. Call OpenWeatherMap Live API if API key is present
        if self.api_key:
            live_data = await self._call_api(latitude, longitude, district, state)
            if live_data is not None:
                await self._set_in_cache(latitude, longitude, district, state, live_data)
                return live_data
            logger.warning(
                f"[WEATHER CLIENT] OpenWeatherMap failed for '{location_label}'. "
                "Falling back to alternative weather provider if available."
            )

        # 3. Live fallback provider (Open-Meteo) when valid GPS coordinates exist
        if latitude is not None and longitude is not None:
            open_meteo_data = await self._fetch_open_meteo(latitude, longitude, location_label)
            if open_meteo_data is not None:
                logger.info(f"[WEATHER CLIENT] Live Open-Meteo weather succeeded for '{location_label}'.")
                await self._set_in_cache(latitude, longitude, district, state, open_meteo_data)
                return open_meteo_data

        # 4. Check for Recent Cached/Stale Weather Data when live providers fail
        stale_cached = await self._get_from_cache(
            latitude, longitude, district, state, allow_stale=True, max_age_seconds=self.max_stale_seconds
        )
        if stale_cached is not None:
            freshness_min = stale_cached.get("data_freshness_minutes", 0.0)
            logger.info(
                f"[WEATHER CLIENT] Reusing recent cached weather data for '{location_label}' "
                f"(age={freshness_min:.1f} mins) following live provider outage."
            )
            stale_cached["is_live"] = False
            stale_cached["source_note"] = f"Cached Weather ({int(freshness_min)}m ago)"
            return stale_cached

        # 5. Keyless Fallback Mock Data Generator (Non-prod only, when no API key configured)
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

        logger.warning(
            f"[WEATHER CLIENT] Weather query for '{location_label}' returning None (all providers exhausted)."
        )
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
        """Make async GET request to OpenWeatherMap. Never print API secrets in logs."""
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
            # Mask sensitive API key in debug logs
            safe_params = {k: ("***" if k == "appid" else v) for k, v in params.items()}
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                logger.info(f"[WEATHER CLIENT] Calling live OpenWeatherMap API: {safe_params}")
                response = await client.get(self.api_url, params=params)

            if response.status_code == 401:
                logger.warning(
                    "[WEATHER CLIENT] OpenWeatherMap API returned HTTP 401 Unauthorized. "
                    "The configured OPENWEATHER_API_KEY is invalid, unauthorized, or inactive. "
                    "Skipping retries and proceeding immediately to fallback provider."
                )
                return None

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

    async def _fetch_open_meteo(
        self,
        latitude: float,
        longitude: float,
        loc_name: str,
    ) -> Optional[Dict]:
        """
        Fetch live weather from Open-Meteo (keyless global weather service).
        Handles HTTP 429 rate limiting gracefully with bounded backoff and no retry storms.
        """
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                f"&timezone=auto"
            )
            headers = {
                "User-Agent": "BhoomiMitraAI/1.0 (https://bhoomimitra.org; contact@bhoomimitra.org)",
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
                resp = await client.get(url)

                if resp.status_code == 429:
                    retry_after_raw = resp.headers.get("Retry-After")
                    retry_after_sec: Optional[float] = None
                    if retry_after_raw:
                        try:
                            retry_after_sec = float(retry_after_raw)
                        except (ValueError, TypeError):
                            pass

                    logger.warning(
                        f"[WEATHER CLIENT] Open-Meteo returned HTTP 429 Too Many Requests. "
                        f"Retry-After header: {retry_after_raw}."
                    )

                    # Bounded backoff: at most 1 retry if Retry-After is absent or small (<= 2.0s)
                    if retry_after_sec is not None and retry_after_sec <= 2.0:
                        backoff = max(0.1, min(retry_after_sec, 2.0))
                        logger.info(f"[WEATHER CLIENT] Open-Meteo rate limit backoff: sleeping {backoff:.2f}s before 1 retry.")
                        import asyncio
                        await asyncio.sleep(backoff)
                        resp = await client.get(url)
                    elif retry_after_sec is None:
                        # Short bounded backoff (0.5s) if no Retry-After header supplied
                        logger.info("[WEATHER CLIENT] Open-Meteo rate limit backoff: sleeping 0.5s before 1 retry.")
                        import asyncio
                        await asyncio.sleep(0.5)
                        resp = await client.get(url)
                    else:
                        logger.warning(
                            f"[WEATHER CLIENT] Open-Meteo Retry-After ({retry_after_sec}s) exceeds threshold. "
                            "Aborting retry to prevent retry storm."
                        )
                        return None

                if resp.status_code != 200:
                    logger.warning(f"[WEATHER CLIENT] Open-Meteo responded with status {resp.status_code}")
                    return None

                data = resp.json()

            curr = data.get("current", {})
            daily = data.get("daily", {})

            code = curr.get("weather_code", 0)
            if code == 0:
                cond_desc, cond_code = "Clear Sky", 800
            elif code in (1, 2, 3):
                cond_desc, cond_code = "Partly Cloudy", 802
            elif code in (45, 48):
                cond_desc, cond_code = "Foggy", 741
            elif code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
                cond_desc, cond_code = "Rain", 500
            elif code in (95, 96, 99):
                cond_desc, cond_code = "Thunderstorm", 200
            else:
                cond_desc, cond_code = "Cloudy", 803

            now = datetime.utcnow()
            forecast = []
            daily_times = daily.get("time", [])
            daily_max = daily.get("temperature_2m_max", [])
            daily_codes = daily.get("weather_code", [])

            for i, d_date in enumerate(daily_times[1:5]):
                t_max = daily_max[i + 1] if i + 1 < len(daily_max) else curr.get("temperature_2m", 28.0)
                c_code = daily_codes[i + 1] if i + 1 < len(daily_codes) else 0
                if c_code == 0:
                    d_desc, d_slot_code = "Clear Sky", 800
                elif c_code in (1, 2, 3):
                    d_desc, d_slot_code = "Partly Cloudy", 802
                elif c_code >= 50:
                    d_desc, d_slot_code = "Rain", 500
                else:
                    d_desc, d_slot_code = "Cloudy", 803

                for h in (9, 12, 15, 18):
                    forecast.append({
                        "dt_txt": f"{d_date} {h:02d}:00:00",
                        "temp": round(float(t_max), 1),
                        "humidity": int(curr.get("relative_humidity_2m", 60)),
                        "description": d_desc,
                        "condition_code": d_slot_code,
                    })

            return {
                "location_name": loc_name,
                "latitude": latitude,
                "longitude": longitude,
                "current": {
                    "temp": round(float(curr.get("temperature_2m", 28.0)), 1),
                    "feels_like": round(float(curr.get("apparent_temperature", 29.0)), 1),
                    "humidity": int(curr.get("relative_humidity_2m", 60)),
                    "wind_speed": round(float(curr.get("wind_speed_10m", 10.0)), 1),
                    "description": cond_desc,
                    "condition_code": cond_code,
                },
                "forecast": forecast,
                "data_available": True,
                "data_freshness_minutes": 0.0,
                "is_live": True,
                "source_note": "Live Weather (Open-Meteo)",
            }
        except Exception as exc:
            logger.warning(f"[WEATHER CLIENT] Open-Meteo live fetch failed: {exc}")
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
                "data_freshness_minutes": 0.0,
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
        allow_stale: bool = False,
        max_age_seconds: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Fetch cached weather data from Redis or in-memory cache.
        If allow_stale is False, only returns data within self.cache_ttl.
        If allow_stale is True, returns data within max_age_seconds (or max_stale_seconds).
        """
        target_ttl = max_age_seconds if (allow_stale and max_age_seconds is not None) else (
            self.max_stale_seconds if allow_stale else self.cache_ttl
        )
        key = self._cache_key(latitude, longitude, district, state)

        # 1. Try Redis first
        try:
            import redis.asyncio as aioredis
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            val = await r.get(key)
            await r.aclose()
            if val:
                cached_data = json.loads(val)
                cached_at_str = cached_data.get("_cached_at")
                if cached_at_str:
                    try:
                        cached_dt = datetime.fromisoformat(cached_at_str)
                        age_sec = (datetime.utcnow() - cached_dt).total_seconds()
                        if age_sec <= target_ttl:
                            cached_data["data_freshness_minutes"] = max(0.0, round(age_sec / 60.0, 1))
                            _MEMORY_CACHE[key] = (cached_dt, copy.deepcopy(cached_data))
                            return cached_data
                    except Exception:
                        return cached_data
                else:
                    return cached_data
        except Exception as exc:
            logger.debug(f"[WEATHER CLIENT] Redis get skipped: {exc}")

        # 2. Try In-Memory Cache
        if key in _MEMORY_CACHE:
            cached_dt, cached_data = _MEMORY_CACHE[key]
            age_sec = (datetime.utcnow() - cached_dt).total_seconds()
            if age_sec <= target_ttl:
                data_copy = copy.deepcopy(cached_data)
                data_copy["data_freshness_minutes"] = max(0.0, round(age_sec / 60.0, 1))
                return data_copy

        return None

    async def _set_in_cache(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str],
        state: Optional[str],
        data: Dict,
    ) -> None:
        """Store weather in both process in-memory cache and Redis."""
        key = self._cache_key(latitude, longitude, district, state)
        now = datetime.utcnow()
        data_to_store = copy.deepcopy(data)
        data_to_store["_cached_at"] = now.isoformat()
        data_to_store["data_freshness_minutes"] = 0.0

        # Update process memory cache
        _MEMORY_CACHE[key] = (now, data_to_store)

        # Update Redis with extended TTL so stale data is accessible on provider outages
        try:
            import redis.asyncio as aioredis
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            redis_ttl = max(self.cache_ttl, self.max_stale_seconds)
            await r.setex(key, redis_ttl, json.dumps(data_to_store))
            await r.aclose()
            logger.debug(f"[WEATHER CLIENT] Cached weather in Redis for key '{key}' (TTL={redis_ttl}s)")
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
            "data_freshness_minutes": 0.0,
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
