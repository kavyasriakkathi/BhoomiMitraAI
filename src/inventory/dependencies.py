from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.inventory.repository import InventoryRepository
from src.inventory.service import InventoryService


def get_inventory_repository(db: AsyncSession = Depends(get_db)) -> InventoryRepository:
    return InventoryRepository(db)


def get_inventory_service(
    repo: InventoryRepository = Depends(get_inventory_repository),
) -> InventoryService:
    return InventoryService(repo)
