"""
BhoomiMitra AI — Shop Discovery Provider Protocol / Interface
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.shops.discovery.schemas import DiscoveredShopItem


class ShopDiscoveryProvider(ABC):
    """
    Abstract interface for external shop / places discovery providers.
    All providers must implement search_nearby and handle failures fail-soft.
    """

    @abstractmethod
    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 25000,
        query: Optional[str] = None,
    ) -> List[DiscoveredShopItem]:
        """
        Search for nearby agricultural shops around a coordinate within radius_meters.
        Never raises exceptions; returns empty list on error.
        """
        pass

    @abstractmethod
    async def search_by_text(
        self,
        text_query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_meters: int = 25000,
    ) -> List[DiscoveredShopItem]:
        """
        Search for agricultural shops by location text (e.g. 'fertilizer shops in Korutla').
        Never raises exceptions; returns empty list on error.
        """
        pass
