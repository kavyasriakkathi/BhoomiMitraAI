from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farmer_profiles.schemas import (
    FarmerProfileCreate, 
    FarmerProfileUpdate, 
    FarmerProfileResponse, 
    PaginatedFarmerProfileResponse
)
from src.farmer_profiles.service import FarmerProfileService
from src.farmer_profiles.dependencies import get_farmer_profile_service

router = APIRouter()

@router.post(
    "",
    response_model=FarmerProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new farmer profile",
    description="Registers a new profile for a farmer. A farmer can only have one profile."
)
async def create_farmer_profile(
    profile_data: FarmerProfileCreate,
    service: FarmerProfileService = Depends(get_farmer_profile_service),
    db: AsyncSession = Depends(get_db)
):
    profile = await service.create_profile(profile_data)
    await db.commit()
    return profile

@router.get(
    "",
    response_model=PaginatedFarmerProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="List farmer profiles",
    description="Retrieve a paginated list of farmer profiles."
)
async def get_farmer_profiles(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    service: FarmerProfileService = Depends(get_farmer_profile_service)
):
    total, items = await service.get_profiles(page=page, size=size)
    return PaginatedFarmerProfileResponse(
        total=total,
        items=items,
        page=page,
        size=size
    )

@router.get(
    "/farmer/{farmer_id}",
    response_model=FarmerProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a farmer profile by farmer ID",
    description="Retrieve a profile associated with a specific farmer UUID."
)
async def get_farmer_profile_by_farmer(
    farmer_id: UUID,
    service: FarmerProfileService = Depends(get_farmer_profile_service)
):
    return await service.get_profile_by_farmer(farmer_id)

@router.get(
    "/{profile_id}",
    response_model=FarmerProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a farmer profile",
    description="Retrieve a single farmer profile by its UUID."
)
async def get_farmer_profile(
    profile_id: UUID,
    service: FarmerProfileService = Depends(get_farmer_profile_service)
):
    return await service.get_profile(profile_id)

@router.put(
    "/{profile_id}",
    response_model=FarmerProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a farmer profile",
    description="Update a farmer's profile details. Only provided fields will be updated."
)
async def update_farmer_profile(
    profile_id: UUID,
    profile_data: FarmerProfileUpdate,
    service: FarmerProfileService = Depends(get_farmer_profile_service),
    db: AsyncSession = Depends(get_db)
):
    profile = await service.update_profile(profile_id, profile_data)
    await db.commit()
    return profile

@router.delete(
    "/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a farmer profile",
    description="Hard delete a farmer profile by its UUID."
)
async def delete_farmer_profile(
    profile_id: UUID,
    service: FarmerProfileService = Depends(get_farmer_profile_service),
    db: AsyncSession = Depends(get_db)
):
    await service.delete_profile(profile_id)
    await db.commit()
