from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.advisory.repository import AdvisoryRepository
from src.advisory.service import AdvisoryService
from src.farmers.repository import FarmerRepository
from src.farmers.dependencies import get_farmer_repository

def get_advisory_repository(session: AsyncSession = Depends(get_db)) -> AdvisoryRepository:
    return AdvisoryRepository(session)

def get_advisory_service(
    repository: AdvisoryRepository = Depends(get_advisory_repository),
    farmer_repository: FarmerRepository = Depends(get_farmer_repository),
) -> AdvisoryService:
    return AdvisoryService(repository, farmer_repository)
