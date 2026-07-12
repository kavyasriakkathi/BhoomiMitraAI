from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import Farmer
from src.farmers.repository import FarmerRepository
from src.farmers.schemas import FarmerCreate, FarmerUpdate

class FarmerService:
    def __init__(self, repository: FarmerRepository):
        self.repository = repository

    async def create_farmer(self, farmer_data: FarmerCreate) -> Farmer:
        existing_farmer = await self.repository.get_by_phone(farmer_data.phone_number)
        if existing_farmer:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Farmer with this phone number already exists."
            )
        
        farmer = Farmer(**farmer_data.model_dump())
        return await self.repository.create(farmer)

    async def get_farmer(self, farmer_id: UUID) -> Farmer:
        farmer = await self.repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )
        return farmer

    async def get_farmers(self, page: int = 1, size: int = 10) -> Tuple[int, List[Farmer]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def update_farmer(self, farmer_id: UUID, farmer_data: FarmerUpdate) -> Farmer:
        farmer = await self.get_farmer(farmer_id)
        
        update_data = farmer_data.model_dump(exclude_unset=True)
        if "phone_number" in update_data and update_data["phone_number"] != farmer.phone_number:
            existing = await self.repository.get_by_phone(update_data["phone_number"])
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Phone number already registered to another farmer."
                )
                
        for key, value in update_data.items():
            setattr(farmer, key, value)
            
        await self.repository.session.flush()
        await self.repository.session.refresh(farmer)
        return farmer

    async def delete_farmer(self, farmer_id: UUID) -> None:
        farmer = await self.get_farmer(farmer_id)
        await self.repository.delete(farmer)
