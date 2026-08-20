from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.crop_health.repository import CropHealthRepository
from src.crop_health.service import CropHealthService
from src.crops.repository import CropRepository
from src.farmers.repository import FarmerRepository
from src.crops.dependencies import get_crop_repository
from src.farmers.dependencies import get_farmer_repository

def get_crop_health_repository(session: AsyncSession = Depends(get_db)) -> CropHealthRepository:
    return CropHealthRepository(session)

def get_crop_health_service(
    repository: CropHealthRepository = Depends(get_crop_health_repository),
    crop_repository: CropRepository = Depends(get_crop_repository),
    farmer_repository: FarmerRepository = Depends(get_farmer_repository),
) -> CropHealthService:
    return CropHealthService(repository, crop_repository, farmer_repository)
