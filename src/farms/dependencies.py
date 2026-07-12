from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farms.repository import FarmRepository
from src.farms.service import FarmService
from src.farmers.repository import FarmerRepository
from src.farmers.dependencies import get_farmer_repository


def get_farm_repository(session: AsyncSession = Depends(get_db)) -> FarmRepository:
    return FarmRepository(session)


def get_farm_service(
    repository: FarmRepository = Depends(get_farm_repository),
    farmer_repository: FarmerRepository = Depends(get_farmer_repository),
) -> FarmService:
    return FarmService(repository, farmer_repository)
