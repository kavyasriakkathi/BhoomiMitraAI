"""
BhoomiMitra AI — Mock Shop Discovery Provider
Used for local testing, offline workflows, and deterministic unit/regression test suites.
"""
from typing import List, Optional, Tuple, Dict
from src.core.logging import logger
from src.shops.repository import haversine_distance
from src.shops.discovery.base import ShopDiscoveryProvider
from src.shops.discovery.schemas import DiscoveredShopItem


class MockShopDiscoveryProvider(ShopDiscoveryProvider):
    """
    Mock discovery provider returning realistic rural Indian agricultural retail shops.
    Supports injecting specific mock results or simulating provider outages for fail-soft tests.
    """

    def __init__(
        self,
        should_fail: bool = False,
        injected_shops: Optional[List[DiscoveredShopItem]] = None,
        custom_geocodes: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.should_fail = should_fail
        self._injected_shops = injected_shops
        self._custom_geocodes = custom_geocodes or {}

    def set_injected_shops(self, shops: Optional[List[DiscoveredShopItem]]) -> None:
        self._injected_shops = shops

    def set_should_fail(self, fail: bool) -> None:
        self.should_fail = fail

    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 25000,
        query: Optional[str] = None,
    ) -> List[DiscoveredShopItem]:
        if self.should_fail:
            logger.warning("[MOCK DISCOVERY] Simulating discovery provider failure / timeout.")
            return []

        if self._injected_shops is not None:
            # Filter injected shops by radius
            results = []
            max_km = radius_meters / 1000.0
            for s in self._injected_shops:
                dist = s.distance_km if s.distance_km is not None else haversine_distance(latitude, longitude, s.latitude, s.longitude)
                s_copy = s.model_copy()
                s_copy.distance_km = dist
                if dist <= max_km:
                    results.append(s_copy)
            results.sort(key=lambda x: (x.distance_km if x.distance_km is not None else 9999))
            return results

        # Default realistic rural Telangana mock dataset
        default_dataset = [
            DiscoveredShopItem(
                provider="mock",
                provider_place_id="mock_place_sri_lakshmi_fert",
                shop_name="Sri Lakshmi Fertilizers & Pesticides",
                business_type="fertilizer_dealer",
                address="Main Road, Near Bus Stand, Korutla, Jagtial, Telangana",
                latitude=18.8210,
                longitude=78.7125,
                phone_number="+91 9440123456",
                maps_url="https://maps.google.com/?q=18.8210,78.7125",
                rating=4.3,
                user_ratings_total=48,
                is_operational=True,
            ),
            DiscoveredShopItem(
                provider="mock",
                provider_place_id="mock_place_sai_agro_agencies",
                shop_name="Sai Agro Agencies & Seeds",
                business_type="seed_store",
                address="Market Yard Gate, Korutla, Jagtial, Telangana",
                latitude=18.8245,
                longitude=78.7150,
                phone_number="+91 9848123789",
                maps_url="https://maps.google.com/?q=18.8245,78.7150",
                rating=4.5,
                user_ratings_total=72,
                is_operational=True,
            ),
            DiscoveredShopItem(
                provider="mock",
                provider_place_id="mock_place_kisan_seva_kendra",
                shop_name="Kisan Seva Kendra Agro Store",
                business_type="agricultural_input_store",
                address="Station Road, Jagtial, Telangana",
                latitude=18.7950,
                longitude=78.9120,
                phone_number=None,  # Intentionally null to test no fake phone
                maps_url="https://maps.google.com/?q=18.7950,78.9120",
                rating=4.0,
                user_ratings_total=19,
                is_operational=True,
            ),
        ]

        max_km = radius_meters / 1000.0
        results = []
        for s in default_dataset:
            dist = haversine_distance(latitude, longitude, s.latitude, s.longitude)
            s_copy = s.model_copy()
            s_copy.distance_km = dist
            if dist <= max_km:
                results.append(s_copy)

        results.sort(key=lambda x: (x.distance_km if x.distance_km is not None else 9999))
        return results

    async def search_by_text(
        self,
        text_query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_meters: int = 25000,
    ) -> List[DiscoveredShopItem]:
        if self.should_fail:
            return []

        # If coordinates provided, delegate to search_nearby
        if latitude is not None and longitude is not None:
            return await self.search_nearby(latitude, longitude, radius_meters=radius_meters, query=text_query)

        # Fallback coordinate for mock (e.g. Korutla center: 18.82, 78.71)
        return await self.search_nearby(18.8200, 78.7100, radius_meters=radius_meters, query=text_query)

    async def geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        if self.should_fail or not location_name:
            return None
        loc_l = location_name.lower().strip()
        if hasattr(self, "_custom_geocodes") and self._custom_geocodes and loc_l in self._custom_geocodes:
            return self._custom_geocodes[loc_l]
        mock_coords = {
            "narapally": (17.4059, 78.6180),
            "నారపల్లి": (17.4059, 78.6180),
            "korutla": (18.8200, 78.7100),
            "కోరుట్ల": (18.8200, 78.7100),
            "jagtial": (18.7900, 78.9100),
            "జగిత్యాల": (18.7900, 78.9100),
            "warangal": (17.9700, 79.5900),
            "వరంగల్": (17.9700, 79.5900),
            "hyderabad": (17.3850, 78.4867),
            "హైదరాబాద్": (17.3850, 78.4867),
        }
        return mock_coords.get(loc_l)

