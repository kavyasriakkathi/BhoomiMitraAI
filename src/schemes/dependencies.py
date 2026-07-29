from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.schemes.repository import SchemeRepository
from src.farmers.repository import FarmerRepository
from src.schemes.service import SchemeService


async def get_scheme_service(db: AsyncSession = Depends(get_db)) -> SchemeService:
    repository = SchemeRepository(db)
    farmer_repository = FarmerRepository(db)
    return SchemeService(repository=repository, farmer_repository=farmer_repository)
