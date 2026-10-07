"""
BhoomiMitra AI — Shop Discovery Orchestrator

Orchestrates external places discovery, agricultural filtering, deduplication
against BhoomiMitra verified partner shops, database persistence in `discovered_shops`,
and fail-soft error containment.
"""
import re
from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.logging import logger
from src.core.models import Shop, DiscoveredShop
from src.config import get_settings
from src.shops.repository import haversine_distance
from src.shops.discovery.schemas import DiscoveredShopItem
from src.shops.discovery.base import ShopDiscoveryProvider
from src.shops.discovery.factory import get_discovery_provider
from src.shops.discovery.filter import AgriculturalShopFilter


def _normalize_phone(phone: Optional[str]) -> str:
    """Normalize phone number to digits only for deduplication."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    # Strip leading 91 or 0
    if len(digits) > 10 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) > 10 and digits.startswith("0"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else digits


class ShopDiscoveryOrchestrator:
    """
    Coordinates automatic external agricultural shop discovery.
    Ensures:
    1. Verified partner shops take precedence and are never overwritten.
    2. Non-agricultural businesses are filtered out deterministically.
    3. Discovered shops are stored in `discovered_shops` table.
    4. Stock for discovered shops is strictly marked UNVERIFIED / UNKNOWN.
    """

    def __init__(self, provider: Optional[ShopDiscoveryProvider] = None):
        self.settings = get_settings()
        if provider is not None:
            self.provider = provider
        elif not self.settings.shop_discovery_enabled:
            self.provider = None
        else:
            self.provider = get_discovery_provider()

    async def discover_nearby_agricultural_shops(
        self,
        db: AsyncSession,
        latitude: float,
        longitude: float,
        radius_meters: Optional[int] = None,
        product_query: Optional[str] = None,
        verified_shops: Optional[List[Shop]] = None,
    ) -> List[DiscoveredShopItem]:
        """
        Discover nearby agricultural shops around coordinates.
        Never raises exceptions — handles all errors fail-soft.
        """
        if not self.settings.shop_discovery_enabled and self.provider is None:
            logger.info("[DISCOVERY ORCHESTRATOR] Discovery is disabled. Returning empty candidates.")
            return []

        if not self.provider:
            logger.warning("[DISCOVERY ORCHESTRATOR] No discovery provider configured. Returning empty candidates.")
            return []

        radius = radius_meters or self.settings.shop_discovery_radius_meters

        # Search term tailored for agricultural suppliers
        search_kw = "fertilizer shop OR seed store OR agro agencies"
        if product_query:
            search_kw = f"{product_query} agricultural shop OR fertilizer shop"

        try:
            raw_candidates = await self.provider.search_nearby(
                latitude=latitude,
                longitude=longitude,
                radius_meters=radius,
                query=search_kw,
            )
        except Exception as prov_err:
            logger.warning(f"[DISCOVERY ORCHESTRATOR] Provider error: {prov_err}. Returning empty candidates.")
            return []

        if not raw_candidates:
            logger.info(f"[DISCOVERY ORCHESTRATOR] No shops found by provider near ({latitude}, {longitude})")
            return []

        # 1. Deterministic Agricultural Filtering
        agri_candidates = [
            c for c in raw_candidates if AgriculturalShopFilter.is_agricultural_shop(c)
        ]
        logger.info(f"[DISCOVERY ORCHESTRATOR] {len(agri_candidates)} / {len(raw_candidates)} passed agricultural filter.")

        # 2. Deduplication against verified partner shops
        verified_phones = set()
        verified_names = set()
        if verified_shops:
            for vs in verified_shops:
                norm_p = _normalize_phone(vs.phone_number)
                if norm_p:
                    verified_phones.add(norm_p)
                if vs.shop_name:
                    verified_names.add(vs.shop_name.strip().lower())

        deduped: List[DiscoveredShopItem] = []
        for c in agri_candidates:
            # Check phone duplicate
            c_phone_norm = _normalize_phone(c.phone_number)
            if c_phone_norm and c_phone_norm in verified_phones:
                logger.debug(f"[DISCOVERY DEDUP] Dropping discovered shop '{c.shop_name}' — phone matches verified shop.")
                continue

            # Check exact name & close vicinity duplicate
            c_name_lower = (c.shop_name or "").strip().lower()
            if c_name_lower in verified_names:
                # If name matches any verified shop, skip so verified shop is always priority
                logger.debug(f"[DISCOVERY DEDUP] Dropping discovered shop '{c.shop_name}' — matches verified partner name.")
                continue

            # Check distance calculation
            if c.distance_km is None:
                c.distance_km = haversine_distance(latitude, longitude, c.latitude, c.longitude)

            deduped.append(c)

        # 3. Persist / cache discovered shops in `discovered_shops` table (fail-soft)
        await self._persist_discovered_shops(db, deduped)

        # 4. Sort by distance
        deduped.sort(key=lambda x: (x.distance_km if x.distance_km is not None else 9999.0))
        return deduped

    async def _persist_discovered_shops(self, db: AsyncSession, shops: List[DiscoveredShopItem]) -> None:
        """Cache discovered shops into database without crashing transaction."""
        if not shops or not db:
            return

        try:
            for s in shops:
                stmt = select(DiscoveredShop).where(
                    DiscoveredShop.provider == s.provider,
                    DiscoveredShop.provider_place_id == s.provider_place_id,
                )
                res = await db.execute(stmt)
                existing = res.scalar_one_or_none()

                if existing:
                    existing.last_verified_at = datetime.utcnow()
                    existing.latitude = s.latitude
                    existing.longitude = s.longitude
                    existing.phone_number = s.phone_number or existing.phone_number
                    existing.address = s.address or existing.address
                    existing.maps_url = s.maps_url or existing.maps_url
                    existing.rating = s.rating or existing.rating
                    existing.user_ratings_total = s.user_ratings_total or existing.user_ratings_total
                    existing.is_operational = s.is_operational
                    db.add(existing)
                else:
                    new_shop = DiscoveredShop(
                        provider=s.provider,
                        provider_place_id=s.provider_place_id,
                        shop_name=s.shop_name,
                        business_type=s.business_type,
                        address=s.address,
                        latitude=s.latitude,
                        longitude=s.longitude,
                        phone_number=s.phone_number,
                        maps_url=s.maps_url,
                        rating=s.rating,
                        user_ratings_total=s.user_ratings_total,
                        is_operational=s.is_operational,
                    )
                    db.add(new_shop)

            await db.flush()
        except Exception as persist_err:
            logger.warning(f"[DISCOVERY PERSIST] Non-fatal error persisting discovered shops: {persist_err}")

    async def geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Geocode location name using configured discovery provider.
        """
        if not location_name:
            return None
        if self.provider and hasattr(self.provider, "geocode_location"):
            try:
                coords = await self.provider.geocode_location(location_name)
                if coords:
                    return coords
            except Exception as geo_err:
                logger.warning(f"[DISCOVERY ORCHESTRATOR] Geocode failed for '{location_name}': {geo_err}")
        return None

