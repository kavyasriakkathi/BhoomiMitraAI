from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farmer_profiles.repository import FarmerProfileRepository
from src.farmer_profiles.service import FarmerProfileService
from src.farmers.repository import FarmerRepository
from src.farmers.dependencies import get_farmer_repository

def get_farmer_profile_repository(session: AsyncSession = Depends(get_db)) -> FarmerProfileRepository:
    return FarmerProfileRepository(session)

def get_farmer_profile_service(
    repository: FarmerProfileRepository = Depends(get_farmer_profile_repository),
    farmer_repository: FarmerRepository = Depends(get_farmer_repository)
) -> FarmerProfileService:
    return FarmerProfileService(repository, farmer_repository)
