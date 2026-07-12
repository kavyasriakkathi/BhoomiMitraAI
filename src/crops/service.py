from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import Crop
from src.crops.repository import CropRepository
from src.crops.schemas import CropCreate, CropUpdate
from src.farms.repository import FarmRepository

class CropService:
    def __init__(self, repository: CropRepository, farm_repository: FarmRepository):
        self.repository = repository
        self.farm_repository = farm_repository

    async def create_crop(self, data: CropCreate) -> Crop:
        farm = await self.farm_repository.get_by_id(data.farm_id)
        if not farm:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found.")

        crop = Crop(**data.model_dump())
        return await self.repository.create(crop)

    async def get_crop(self, crop_id: UUID) -> Crop:
        crop = await self.repository.get_by_id(crop_id)
        if not crop:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop not found.")
        return crop

    async def get_crops(self, page: int = 1, size: int = 10) -> Tuple[int, List[Crop]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def get_farm_crops(self, farm_id: UUID, page: int = 1, size: int = 10) -> Tuple[int, List[Crop]]:
        farm = await self.farm_repository.get_by_id(farm_id)
        if not farm:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found.")

        skip = (page - 1) * size
        return await self.repository.get_by_farm_id(farm_id, skip=skip, limit=size)

    async def update_crop(self, crop_id: UUID, data: CropUpdate) -> Crop:
        crop = await self.get_crop(crop_id)

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(crop, key, value)

        await self.repository.session.flush()
        await self.repository.session.refresh(crop)
        return crop

    async def delete_crop(self, crop_id: UUID) -> None:
        crop = await self.get_crop(crop_id)
        await self.repository.delete(crop)
