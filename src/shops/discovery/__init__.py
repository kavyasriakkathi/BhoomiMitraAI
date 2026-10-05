"""
BhoomiMitra AI — Shop Discovery Module
"""
from src.shops.discovery.schemas import DiscoveredShopItem
from src.shops.discovery.base import ShopDiscoveryProvider
from src.shops.discovery.mock_provider import MockShopDiscoveryProvider
from src.shops.discovery.google_places_provider import GooglePlacesDiscoveryProvider
from src.shops.discovery.filter import AgriculturalShopFilter
from src.shops.discovery.factory import get_discovery_provider, reset_discovery_provider
from src.shops.discovery.service import ShopDiscoveryOrchestrator

__all__ = [
    "DiscoveredShopItem",
    "ShopDiscoveryProvider",
    "MockShopDiscoveryProvider",
    "GooglePlacesDiscoveryProvider",
    "AgriculturalShopFilter",
    "get_discovery_provider",
    "reset_discovery_provider",
    "ShopDiscoveryOrchestrator",
]
