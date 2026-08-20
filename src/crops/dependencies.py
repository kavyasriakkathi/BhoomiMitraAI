from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.crops.repository import CropRepository
from src.crops.service import CropService
from src.farms.repository import FarmRepository
from src.farms.dependencies import get_farm_repository

def get_crop_repository(session: AsyncSession = Depends(get_db)) -> CropRepository:
    return CropRepository(session)

def get_crop_service(
    repository: CropRepository = Depends(get_crop_repository),
    farm_repository: FarmRepository = Depends(get_farm_repository),
) -> CropService:
    return CropService(repository, farm_repository)
