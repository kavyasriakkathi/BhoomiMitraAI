"""
BhoomiMitra AI — Agmarknet / data.gov.in API Client

Fetches daily mandi commodity prices from the Government of India's
data.gov.in Agmarknet dataset.

IMPORTANT:
- DATA_GOV_API_KEY is OPTIONAL. If absent, the client returns [] immediately.
- All errors are caught and logged; this client NEVER raises.
- Results are cached in Redis (TTL = market_price_cache_ttl_seconds) to
  minimise API calls. If Redis is unavailable, caching is silently skipped.
"""
import json
import hashlib
import time
from typing import List, Optional
from datetime import datetime

import httpx

from src.core.logging import logger


class AgmarknetClient:
    """
    Async HTTP client for the data.gov.in Agmarknet daily mandi price API.

    Returns a list of price dicts on success, or an empty list on any failure.
    Never raises exceptions — callers must handle the empty-list case.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        cache_ttl_seconds: int = 21600,
        timeout_seconds: float = 5.0,
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.api_url = api_url
        self.cache_ttl = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> List[dict]:
        """
        Fetch mandi prices for a commodity from data.gov.in.

        Returns a list of dicts with keys:
            commodity, market, district, state,
            min_price, max_price, modal_price, arrival_date

        Returns [] if:
          - API key is not configured
          - API returns an error
          - Network/timeout error
          - Response cannot be parsed
        """
        if not self.api_key:
            logger.info(
                "[AGMARKNET] DATA_GOV_API_KEY not configured — "
                "skipping live API call, using local DB fallback."
            )
            return []

        # Try cache first
        cached = await self._get_from_cache(commodity, state, district)
        if cached is not None:
            logger.info(
                f"[AGMARKNET] Cache HIT for commodity='{commodity}' "
                f"state='{state}' district='{district}' ({len(cached)} records)"
            )
            return cached

        # Call the live API
        records = await self._call_api(commodity, state, district)

        # Store in cache regardless of empty (avoids re-calling a dead API repeatedly)
        if records is not None:
            await self._set_in_cache(commodity, state, district, records)

        return records if records is not None else []

    # ------------------------------------------------------------------
    # Internal: API call
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        commodity: str,
        state: Optional[str],
        district: Optional[str],
    ) -> Optional[List[dict]]:
        """
        Make the HTTP call to data.gov.in.
        Returns parsed list of normalised price dicts, or None on any error.
        """
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": 100,
            "filters[commodity]": commodity,
        }
        if state:
            params["filters[state.keyword]"] = state
        if district:
            params["filters[district]"] = district

        start_time = time.time()
        connect_timeout = min(3.0, self.timeout_seconds)
        timeout_config = httpx.Timeout(self.timeout_seconds, connect=connect_timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info(
                    f"[AGMARKNET] Calling live API: commodity='{commodity}' "
                    f"state='{state}' district='{district}' (timeout={self.timeout_seconds}s)"
                )
                response = await client.get(self.api_url, params=params)

            elapsed = time.time() - start_time

            if response.status_code != 200:
                logger.warning(
                    f"[AGMARKNET] API returned HTTP {response.status_code} in {elapsed:.2f}s "
                    f"for commodity='{commodity}'. Using local DB fallback."
                )
                return None

            data = response.json()
            records = data.get("records", [])
            logger.info(
                f"[AGMARKNET] Live API returned {len(records)} records "
                f"for commodity='{commodity}' in {elapsed:.2f}s"
            )
            return self._normalise_records(records)

        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            logger.warning(
                f"[AGMARKNET] API timeout after {elapsed:.2f}s (limit={self.timeout_seconds}s) "
                f"for commodity='{commodity}'. Using local DB fallback."
            )
            return None
        except httpx.RequestError as exc:
            elapsed = time.time() - start_time
            logger.warning(
                f"[AGMARKNET] Network error after {elapsed:.2f}s for commodity='{commodity}': {exc}. "
                "Using local DB fallback."
            )
            return None
        except (ValueError, KeyError) as exc:
            logger.warning(
                f"[AGMARKNET] Failed to parse API response for commodity='{commodity}': {exc}. "
                "Using local DB fallback."
            )
            return None
        except Exception as exc:
            logger.warning(
                f"[AGMARKNET] Unexpected error for commodity='{commodity}': {exc}. "
                "Using local DB fallback."
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Record normalisation
    # ------------------------------------------------------------------

    def _normalise_records(self, raw_records: list) -> List[dict]:
        """
        Map raw data.gov.in Agmarknet fields to the internal schema.

        Expected raw fields (Agmarknet dataset):
            commodity, market, district, state,
            min_price, max_price, modal_price, arrival_date
        """
        normalised = []
        for rec in raw_records:
            try:
                arrival_str = rec.get("arrival_date") or rec.get("date", "")
                price_date = self._parse_date(arrival_str)
                if not price_date:
                    continue  # Skip records without a valid date

                normalised.append({
                    "commodity": str(rec.get("commodity", "")).strip(),
                    "market": str(rec.get("market", "")).strip(),
                    "district": str(rec.get("district", "")).strip(),
                    "state": str(rec.get("state", "")).strip(),
                    "min_price": float(rec.get("min_price", 0) or 0),
                    "max_price": float(rec.get("max_price", 0) or 0),
                    "modal_price": float(rec.get("modal_price", 0) or 0),
                    "arrival_date": price_date,
                })
            except (TypeError, ValueError):
                continue  # Skip malformed records silently

        return normalised

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse Agmarknet date strings. Returns None if unparseable."""
        if not date_str:
            return None
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        logger.debug(f"[AGMARKNET] Could not parse date string: '{date_str}'")
        return None

    # ------------------------------------------------------------------
    # Internal: Redis cache
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        commodity: str,
        state: Optional[str],
        district: Optional[str],
    ) -> str:
        raw = f"market_price:{commodity.lower()}:{(state or '').lower()}:{(district or '').lower()}"
        return "agmarknet:" + hashlib.md5(raw.encode()).hexdigest()[:16]

    async def _get_from_cache(
        self,
        commodity: str,
        state: Optional[str],
        district: Optional[str],
    ) -> Optional[List[dict]]:
        """Return cached records, or None if cache miss / Redis unavailable."""
        try:
            import redis.asyncio as aioredis
            from src.config import get_settings
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            key = self._cache_key(commodity, state, district)
            val = await r.get(key)
            await r.aclose()
            if val:
                return json.loads(val)
            return None
        except Exception as exc:
            logger.debug(f"[AGMARKNET] Redis cache get skipped: {exc}")
            return None

    async def _set_in_cache(
        self,
        commodity: str,
        state: Optional[str],
        district: Optional[str],
        records: List[dict],
    ) -> None:
        """Store records in Redis. Silently skips if Redis is unavailable."""
        try:
            import redis.asyncio as aioredis
            from src.config import get_settings
            settings = get_settings()
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            key = self._cache_key(commodity, state, district)
            # Serialize datetime objects to ISO strings
            serialisable = [
                {**rec, "arrival_date": rec["arrival_date"].isoformat()}
                for rec in records
            ]
            await r.setex(key, self.cache_ttl, json.dumps(serialisable))
            await r.aclose()
            logger.debug(
                f"[AGMARKNET] Cached {len(records)} records for key '{key}' "
                f"(TTL={self.cache_ttl}s)"
            )
        except Exception as exc:
            logger.debug(f"[AGMARKNET] Redis cache set skipped: {exc}")
