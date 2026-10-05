"""
BhoomiMitra AI — Google Places Discovery Provider (Places API New)

Integrates with the Google Places API (New) Text Search / Nearby Search
to discover local agricultural shops. Handles timeouts, errors, and rate limits fail-soft.
"""
from typing import List, Optional
import httpx
from src.core.logging import logger
from src.shops.repository import haversine_distance
from src.shops.discovery.base import ShopDiscoveryProvider
from src.shops.discovery.schemas import DiscoveredShopItem


class GooglePlacesDiscoveryProvider(ShopDiscoveryProvider):
    """
    Google Places API (New) implementation.
    Calls places:searchText with a targeted agricultural query and circular location bias.
    """

    SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

    def __init__(self, api_key: str, timeout_seconds: float = 5.0):
        self.api_key = api_key.strip() if api_key else ""
        self.timeout_seconds = timeout_seconds

    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 25000,
        query: Optional[str] = None,
    ) -> List[DiscoveredShopItem]:
        if not self.api_key:
            logger.info("[GOOGLE PLACES] Bypassing discovery: GOOGLE_PLACES_API_KEY is not configured.")
            return []

        search_query = query or "fertilizer shop OR seed store OR agro agency"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,places.location,"
                "places.nationalPhoneNumber,places.internationalPhoneNumber,places.googleMapsUri,"
                "places.rating,places.userRatingCount,places.businessStatus,places.primaryType"
            ),
        }

        payload = {
            "textQuery": search_query,
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "radius": float(radius_meters),
                }
            },
            "maxResultCount": 10,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.SEARCH_TEXT_URL, json=payload, headers=headers)
                if resp.status_code != 200:
                    logger.warning(
                        f"[GOOGLE PLACES] Search request failed with status {resp.status_code}: {resp.text[:200]}"
                    )
                    return []

                data = resp.json()
                places = data.get("places", [])
                results: List[DiscoveredShopItem] = []

                for p in places:
                    loc = p.get("location", {})
                    p_lat = loc.get("latitude")
                    p_lng = loc.get("longitude")
                    if p_lat is None or p_lng is None:
                        continue

                    # Filter out permanently closed businesses
                    status = p.get("businessStatus", "OPERATIONAL")
                    if status == "CLOSED_PERMANENTLY":
                        continue

                    dist = haversine_distance(latitude, longitude, p_lat, p_lng)

                    # Extract display name safely
                    display_name_obj = p.get("displayName", {})
                    name = display_name_obj.get("text", "Agricultural Shop")

                    phone = p.get("nationalPhoneNumber") or p.get("internationalPhoneNumber")

                    results.append(
                        DiscoveredShopItem(
                            provider="google_places",
                            provider_place_id=p.get("id", f"gp_{p_lat}_{p_lng}"),
                            shop_name=name,
                            business_type=p.get("primaryType") or "agricultural_input_store",
                            address=p.get("formattedAddress"),
                            latitude=p_lat,
                            longitude=p_lng,
                            phone_number=phone,
                            maps_url=p.get("googleMapsUri"),
                            rating=p.get("rating"),
                            user_ratings_total=p.get("userRatingCount"),
                            is_operational=(status == "OPERATIONAL"),
                            distance_km=dist,
                            raw_payload=p,
                        )
                    )

                results.sort(key=lambda x: (x.distance_km if x.distance_km is not None else 9999))
                logger.info(f"[GOOGLE PLACES] Discovered {len(results)} shops near ({latitude}, {longitude})")
                return results

        except httpx.TimeoutException:
            logger.warning(f"[GOOGLE PLACES] Request timed out after {self.timeout_seconds}s. Fail-soft returning [].")
            return []
        except Exception as exc:
            logger.warning(f"[GOOGLE PLACES] Unexpected error during places discovery: {exc}. Fail-soft returning [].")
            return []

    async def search_by_text(
        self,
        text_query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_meters: int = 25000,
    ) -> List[DiscoveredShopItem]:
        if not self.api_key:
            return []

        if latitude is not None and longitude is not None:
            return await self.search_nearby(latitude, longitude, radius_meters=radius_meters, query=text_query)

        # Text search without explicit coordinates
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,places.location,"
                "places.nationalPhoneNumber,places.googleMapsUri,places.rating,places.userRatingCount,places.businessStatus"
            ),
        }
        payload = {
            "textQuery": text_query,
            "maxResultCount": 10,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.SEARCH_TEXT_URL, json=payload, headers=headers)
                if resp.status_code != 200:
                    return []
                data = resp.json()
                places = data.get("places", [])
                results: List[DiscoveredShopItem] = []
                for p in places:
                    loc = p.get("location", {})
                    p_lat = loc.get("latitude")
                    p_lng = loc.get("longitude")
                    if p_lat is None or p_lng is None:
                        continue
                    name = p.get("displayName", {}).get("text", "Agricultural Shop")
                    results.append(
                        DiscoveredShopItem(
                            provider="google_places",
                            provider_place_id=p.get("id", f"gp_{p_lat}_{p_lng}"),
                            shop_name=name,
                            business_type="agricultural_input_store",
                            address=p.get("formattedAddress"),
                            latitude=p_lat,
                            longitude=p_lng,
                            phone_number=p.get("nationalPhoneNumber"),
                            maps_url=p.get("googleMapsUri"),
                            rating=p.get("rating"),
                            user_ratings_total=p.get("userRatingCount"),
                            is_operational=True,
                            distance_km=None,
                        )
                    )
                return results
        except Exception as exc:
            logger.warning(f"[GOOGLE PLACES] Text search error: {exc}. Fail-soft returning [].")
            return []
