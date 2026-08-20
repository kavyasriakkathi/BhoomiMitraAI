from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farms.schemas import FarmCreate, FarmUpdate, FarmResponse, PaginatedFarmResponse
from src.farms.service import FarmService
from src.farms.dependencies import get_farm_service

router = APIRouter()

@router.post("", response_model=FarmResponse, status_code=status.HTTP_201_CREATED,
    summary="Register a new farm",
    description="Create a new farm record for a farmer. The farmer must exist.")
async def create_farm(data: FarmCreate, service: FarmService = Depends(get_farm_service), db: AsyncSession = Depends(get_db)):
    farm = await service.create_farm(data)
    await db.commit()
    return farm

@router.get("", response_model=PaginatedFarmResponse, status_code=status.HTTP_200_OK,
    summary="List farms", description="Retrieve a paginated list of all farms.")
async def get_farms(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: FarmService = Depends(get_farm_service)):
    total, items = await service.get_farms(page=page, size=size)
    return PaginatedFarmResponse(total=total, items=items, page=page, size=size)

@router.get("/farmer/{farmer_id}", response_model=PaginatedFarmResponse, status_code=status.HTTP_200_OK,
    summary="List farms for a farmer", description="Retrieve farms belonging to a specific farmer.")
async def get_farmer_farms(farmer_id: UUID, page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: FarmService = Depends(get_farm_service)):
    total, items = await service.get_farmer_farms(farmer_id, page=page, size=size)
    return PaginatedFarmResponse(total=total, items=items, page=page, size=size)

@router.get("/{farm_id}", response_model=FarmResponse, status_code=status.HTTP_200_OK,
    summary="Get a farm", description="Retrieve a single farm by its UUID.")
async def get_farm(farm_id: UUID, service: FarmService = Depends(get_farm_service)):
    return await service.get_farm(farm_id)

@router.put("/{farm_id}", response_model=FarmResponse, status_code=status.HTTP_200_OK,
    summary="Update a farm", description="Update farm details. Only provided fields will be updated.")
async def update_farm(farm_id: UUID, data: FarmUpdate, service: FarmService = Depends(get_farm_service), db: AsyncSession = Depends(get_db)):
    farm = await service.update_farm(farm_id, data)
    await db.commit()
    return farm

@router.delete("/{farm_id}", status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a farm", description="Hard delete a farm record by its UUID.")
async def delete_farm(farm_id: UUID, service: FarmService = Depends(get_farm_service), db: AsyncSession = Depends(get_db)):
    await service.delete_farm(farm_id)
    await db.commit()
