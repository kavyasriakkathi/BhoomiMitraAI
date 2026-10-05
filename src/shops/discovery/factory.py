"""
BhoomiMitra AI — Shop Discovery Provider Factory
"""
from typing import Optional
from src.config import get_settings
from src.shops.discovery.base import ShopDiscoveryProvider
from src.shops.discovery.mock_provider import MockShopDiscoveryProvider
from src.shops.discovery.google_places_provider import GooglePlacesDiscoveryProvider

_provider_instance: Optional[ShopDiscoveryProvider] = None


def get_discovery_provider(override_provider: Optional[str] = None) -> ShopDiscoveryProvider:
    """
    Returns the configured shop discovery provider instance.
    Fails closed if discovery is enabled but provider is missing or invalid.
    Never falls back to mock when discovery is enabled.
    """
    global _provider_instance
    settings = get_settings()

    # If discovery is globally disabled and no override was requested:
    if not settings.shop_discovery_enabled and not override_provider:
        raise RuntimeError(
            "Shop discovery is disabled (SHOP_DISCOVERY_ENABLED=false). "
            "Cannot instantiate a discovery provider when discovery is disabled."
        )

    raw_provider = override_provider if override_provider is not None else settings.shop_discovery_provider
    provider_type = (raw_provider or "").lower().strip()

    if not provider_type:
        raise ValueError(
            "SHOP_DISCOVERY_ENABLED is True, but SHOP_DISCOVERY_PROVIDER is missing or not configured. "
            "Cannot fall back to mock in enabled mode. "
            "Please explicitly configure SHOP_DISCOVERY_PROVIDER='google_places' (production) "
            "or 'mock' (development/testing)."
        )

    if provider_type == "google_places":
        return GooglePlacesDiscoveryProvider(
            api_key=settings.google_places_api_key,
            timeout_seconds=settings.shop_discovery_api_timeout_seconds,
        )
    elif provider_type == "mock":
        if settings.is_production:
            raise ValueError(
                "SHOP_DISCOVERY_PROVIDER='mock' is strictly forbidden in production (APP_ENV=production). "
                "Only 'google_places' is permitted as a production discovery provider."
            )
        return MockShopDiscoveryProvider()
    else:
        raise ValueError(
            f"Invalid SHOP_DISCOVERY_PROVIDER '{provider_type}'. "
            "Supported providers are 'google_places' and 'mock'. "
            "Failing closed to prevent exposing unverified or unexpected shops."
        )


def reset_discovery_provider() -> None:
    """Reset cached provider instance (useful for testing)."""
    global _provider_instance
    _provider_instance = None
