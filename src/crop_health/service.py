from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import CropHealth
from src.crop_health.repository import CropHealthRepository
from src.crop_health.schemas import CropHealthCreate, CropHealthUpdate
from src.crops.repository import CropRepository
from src.farmers.repository import FarmerRepository

class CropHealthService:
    def __init__(self, repository: CropHealthRepository, crop_repository: CropRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.crop_repository = crop_repository
        self.farmer_repository = farmer_repository

    async def create_diagnosis(self, data: CropHealthCreate) -> CropHealth:
        crop = await self.crop_repository.get_by_id(data.crop_id)
        if not crop:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop not found.")
        
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        diagnosis = CropHealth(**data.model_dump())
        return await self.repository.create(diagnosis)

    async def get_diagnosis(self, diagnosis_id: UUID) -> CropHealth:
        diagnosis = await self.repository.get_by_id(diagnosis_id)
        if not diagnosis:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop health diagnosis not found.")
        return diagnosis

    async def get_diagnoses(self, page: int = 1, size: int = 10) -> Tuple[int, List[CropHealth]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def get_crop_diagnoses(self, crop_id: UUID, page: int = 1, size: int = 10) -> Tuple[int, List[CropHealth]]:
        crop = await self.crop_repository.get_by_id(crop_id)
        if not crop:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop not found.")

        skip = (page - 1) * size
        return await self.repository.get_by_crop_id(crop_id, skip=skip, limit=size)

    async def get_farmer_diagnoses(self, farmer_id: UUID, page: int = 1, size: int = 10) -> Tuple[int, List[CropHealth]]:
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        skip = (page - 1) * size
        return await self.repository.get_by_farmer_id(farmer_id, skip=skip, limit=size)

    async def update_diagnosis(self, diagnosis_id: UUID, data: CropHealthUpdate) -> CropHealth:
        diagnosis = await self.get_diagnosis(diagnosis_id)

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(diagnosis, key, value)

        await self.repository.session.flush()
        await self.repository.session.refresh(diagnosis)
        return diagnosis

    async def delete_diagnosis(self, diagnosis_id: UUID) -> None:
        diagnosis = await self.get_diagnosis(diagnosis_id)
        await self.repository.delete(diagnosis)
