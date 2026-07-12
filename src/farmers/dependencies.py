from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farmers.repository import FarmerRepository
from src.farmers.service import FarmerService

def get_farmer_repository(session: AsyncSession = Depends(get_db)) -> FarmerRepository:
    return FarmerRepository(session)

def get_farmer_service(repository: FarmerRepository = Depends(get_farmer_repository)) -> FarmerService:
    return FarmerService(repository)
