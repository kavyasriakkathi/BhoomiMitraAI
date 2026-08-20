from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.memory.repository import FarmerMemoryRepository
from src.memory.service import FarmerMemoryService

async def get_memory_service(
    db: AsyncSession = Depends(get_db)
) -> FarmerMemoryService:
    """Dependency provider for FarmerMemoryService."""
    repository = FarmerMemoryRepository(db)
    return FarmerMemoryService(repository)
