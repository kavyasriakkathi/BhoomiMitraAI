"""
BhoomiMitra AI — Market Price Dependency Injection

Follows the exact pattern of src/schemes/dependencies.py.
"""
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.config import get_settings
from src.market.repository import MarketPriceRepository
from src.market.agmarknet_client import AgmarknetClient
from src.market.service import MarketService


async def get_market_service(db: AsyncSession = Depends(get_db)) -> MarketService:
    settings = get_settings()
    repository = MarketPriceRepository(db)
    client = AgmarknetClient(
        api_key=settings.data_gov_api_key,
        api_url=settings.agmarknet_api_url,
        cache_ttl_seconds=settings.market_price_cache_ttl_seconds,
    )
    return MarketService(repository=repository, client=client)
