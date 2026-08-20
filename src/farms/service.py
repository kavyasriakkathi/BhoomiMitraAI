from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import Farm
from src.farms.repository import FarmRepository
from src.farms.schemas import FarmCreate, FarmUpdate
from src.farmers.repository import FarmerRepository


class FarmService:
    def __init__(self, repository: FarmRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def create_farm(self, data: FarmCreate) -> Farm:
        # Validate farmer exists
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )

        farm = Farm(**data.model_dump())
        return await self.repository.create(farm)

    async def get_farm(self, farm_id: UUID) -> Farm:
        farm = await self.repository.get_by_id(farm_id)
        if not farm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farm not found."
            )
        return farm

    async def get_farms(self, page: int = 1, size: int = 10) -> Tuple[int, List[Farm]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def get_farmer_farms(
        self, farmer_id: UUID, page: int = 1, size: int = 10
    ) -> Tuple[int, List[Farm]]:
        # Validate farmer exists
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )

        skip = (page - 1) * size
        return await self.repository.get_by_farmer_id(farmer_id, skip=skip, limit=size)

    async def update_farm(self, farm_id: UUID, data: FarmUpdate) -> Farm:
        farm = await self.get_farm(farm_id)

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(farm, key, value)

        await self.repository.session.flush()
        await self.repository.session.refresh(farm)
        return farm

    async def delete_farm(self, farm_id: UUID) -> None:
        farm = await self.get_farm(farm_id)
        await self.repository.delete(farm)
