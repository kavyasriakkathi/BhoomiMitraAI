from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.crops.schemas import CropCreate, CropUpdate, CropResponse, PaginatedCropResponse
from src.crops.service import CropService
from src.crops.dependencies import get_crop_service
from src.auth.dependencies import require_admin
from src.core.models import UserAccount

router = APIRouter()

@router.post("", response_model=CropResponse, status_code=status.HTTP_201_CREATED,
    summary="Register a new crop", description="Create a new crop record for a farm.")
async def create_crop(data: CropCreate, service: CropService = Depends(get_crop_service), db: AsyncSession = Depends(get_db)):
    crop = await service.create_crop(data)
    await db.commit()
    return crop

@router.get("", response_model=PaginatedCropResponse, status_code=status.HTTP_200_OK,
    summary="List crops", description="Retrieve a paginated list of all crops.")
async def get_crops(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: CropService = Depends(get_crop_service)):
    total, items = await service.get_crops(page=page, size=size)
    return PaginatedCropResponse(total=total, items=items, page=page, size=size)

@router.get("/farm/{farm_id}", response_model=PaginatedCropResponse, status_code=status.HTTP_200_OK,
    summary="List crops for a farm", description="Retrieve crops belonging to a specific farm.")
async def get_farm_crops(farm_id: UUID, page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: CropService = Depends(get_crop_service)):
    total, items = await service.get_farm_crops(farm_id, page=page, size=size)
    return PaginatedCropResponse(total=total, items=items, page=page, size=size)

@router.get("/{crop_id}", response_model=CropResponse, status_code=status.HTTP_200_OK,
    summary="Get a crop", description="Retrieve a single crop by its UUID.")
async def get_crop(crop_id: UUID, service: CropService = Depends(get_crop_service)):
    return await service.get_crop(crop_id)

@router.put("/{crop_id}", response_model=CropResponse, status_code=status.HTTP_200_OK,
    summary="Update a crop", description="Update crop details. Only provided fields will be updated.")
async def update_crop(crop_id: UUID, data: CropUpdate, service: CropService = Depends(get_crop_service), db: AsyncSession = Depends(get_db)):
    crop = await service.update_crop(crop_id, data)
    await db.commit()
    return crop

@router.delete("/{crop_id}", status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a crop (Admin only)", description="Hard delete a crop record by its UUID.")
async def delete_crop(crop_id: UUID, current_user: UserAccount = Depends(require_admin), service: CropService = Depends(get_crop_service), db: AsyncSession = Depends(get_db)):
    await service.delete_crop(crop_id)
    await db.commit()
