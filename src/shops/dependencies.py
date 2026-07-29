from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.shops.repository import ShopRepository
from src.shops.service import ShopService


def get_shop_repository(db: AsyncSession = Depends(get_db)) -> ShopRepository:
    return ShopRepository(db)


def get_shop_service(
    repo: ShopRepository = Depends(get_shop_repository),
) -> ShopService:
    return ShopService(repo)
