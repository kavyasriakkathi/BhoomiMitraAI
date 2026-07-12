from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import Advisory
from src.advisory.repository import AdvisoryRepository
from src.advisory.schemas import AdvisoryCreate, AdvisoryUpdate
from src.farmers.repository import FarmerRepository

class AdvisoryService:
    def __init__(self, repository: AdvisoryRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def create(self, data: AdvisoryCreate) -> Advisory:
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        advisory = Advisory(**data.model_dump())
        return await self.repository.create(advisory)

    async def get_by_id(self, advisory_id: UUID) -> Advisory:
        advisory = await self.repository.get_by_id(advisory_id)
        if not advisory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Advisory not found.")
        return advisory

    async def list(self, page: int = 1, size: int = 10) -> Tuple[int, List[Advisory]]:
        return await self.repository.list(page=page, size=size)

    async def list_by_farmer(self, farmer_id: UUID, page: int = 1, size: int = 10) -> Tuple[int, List[Advisory]]:
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        return await self.repository.list_by_farmer(farmer_id, page=page, size=size)

    async def update(self, advisory_id: UUID, data: AdvisoryUpdate) -> Advisory:
        advisory = await self.get_by_id(advisory_id)

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(advisory, key, value)

        return await self.repository.update(advisory)

    async def delete(self, advisory_id: UUID) -> None:
        advisory = await self.get_by_id(advisory_id)
        await self.repository.delete(advisory)
