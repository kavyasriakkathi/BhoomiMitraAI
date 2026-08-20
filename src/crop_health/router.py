from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.crop_health.schemas import CropHealthCreate, CropHealthUpdate, CropHealthResponse, PaginatedCropHealthResponse
from src.crop_health.service import CropHealthService
from src.crop_health.dependencies import get_crop_health_service

router = APIRouter()

@router.post("", response_model=CropHealthResponse, status_code=status.HTTP_201_CREATED,
    summary="Register a new crop health diagnosis", description="Create a new crop health diagnosis record.")
async def create_diagnosis(data: CropHealthCreate, service: CropHealthService = Depends(get_crop_health_service), db: AsyncSession = Depends(get_db)):
    diagnosis = await service.create_diagnosis(data)
    await db.commit()
    return diagnosis

@router.get("", response_model=PaginatedCropHealthResponse, status_code=status.HTTP_200_OK,
    summary="List diagnoses", description="Retrieve a paginated list of all crop health diagnoses.")
async def get_diagnoses(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: CropHealthService = Depends(get_crop_health_service)):
    total, items = await service.get_diagnoses(page=page, size=size)
    return PaginatedCropHealthResponse(total=total, items=items, page=page, size=size)

@router.get("/crop/{crop_id}", response_model=PaginatedCropHealthResponse, status_code=status.HTTP_200_OK,
    summary="List diagnoses for a crop", description="Retrieve diagnoses belonging to a specific crop.")
async def get_crop_diagnoses(crop_id: UUID, page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: CropHealthService = Depends(get_crop_health_service)):
    total, items = await service.get_crop_diagnoses(crop_id, page=page, size=size)
    return PaginatedCropHealthResponse(total=total, items=items, page=page, size=size)

@router.get("/farmer/{farmer_id}", response_model=PaginatedCropHealthResponse, status_code=status.HTTP_200_OK,
    summary="List diagnoses for a farmer", description="Retrieve diagnoses belonging to a specific farmer.")
async def get_farmer_diagnoses(farmer_id: UUID, page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: CropHealthService = Depends(get_crop_health_service)):
    total, items = await service.get_farmer_diagnoses(farmer_id, page=page, size=size)
    return PaginatedCropHealthResponse(total=total, items=items, page=page, size=size)

@router.get("/{diagnosis_id}", response_model=CropHealthResponse, status_code=status.HTTP_200_OK,
    summary="Get a diagnosis", description="Retrieve a single crop health diagnosis by its UUID.")
async def get_diagnosis(diagnosis_id: UUID, service: CropHealthService = Depends(get_crop_health_service)):
    return await service.get_diagnosis(diagnosis_id)

@router.put("/{diagnosis_id}", response_model=CropHealthResponse, status_code=status.HTTP_200_OK,
    summary="Update a diagnosis", description="Update crop health diagnosis details. Only provided fields will be updated.")
async def update_diagnosis(diagnosis_id: UUID, data: CropHealthUpdate, service: CropHealthService = Depends(get_crop_health_service), db: AsyncSession = Depends(get_db)):
    diagnosis = await service.update_diagnosis(diagnosis_id, data)
    await db.commit()
    return diagnosis

@router.delete("/{diagnosis_id}", status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a diagnosis", description="Hard delete a crop health diagnosis record by its UUID.")
async def delete_diagnosis(diagnosis_id: UUID, service: CropHealthService = Depends(get_crop_health_service), db: AsyncSession = Depends(get_db)):
    await service.delete_diagnosis(diagnosis_id)
    await db.commit()
