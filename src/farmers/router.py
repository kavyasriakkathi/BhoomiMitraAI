from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.farmers.schemas import FarmerCreate, FarmerUpdate, FarmerResponse, PaginatedFarmerResponse
from src.farmers.service import FarmerService
from src.farmers.dependencies import get_farmer_service
from src.auth.dependencies import require_admin
from src.core.models import UserAccount

router = APIRouter()

@router.post(
    "",
    response_model=FarmerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new farmer",
    description="Registers a new farmer into the system. Phone number must be unique."
)
async def create_farmer(
    farmer_data: FarmerCreate,
    service: FarmerService = Depends(get_farmer_service),
    db: AsyncSession = Depends(get_db)
):
    farmer = await service.create_farmer(farmer_data)
    await db.commit()
    return farmer

@router.get(
    "",
    response_model=PaginatedFarmerResponse,
    status_code=status.HTTP_200_OK,
    summary="List farmers",
    description="Retrieve a paginated list of farmers."
)
async def get_farmers(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    service: FarmerService = Depends(get_farmer_service)
):
    total, items = await service.get_farmers(page=page, size=size)
    return PaginatedFarmerResponse(
        total=total,
        items=items,
        page=page,
        size=size
    )

@router.get(
    "/{farmer_id}",
    response_model=FarmerResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a farmer",
    description="Retrieve a single farmer by their UUID."
)
async def get_farmer(
    farmer_id: UUID,
    service: FarmerService = Depends(get_farmer_service)
):
    return await service.get_farmer(farmer_id)

@router.put(
    "/{farmer_id}",
    response_model=FarmerResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a farmer",
    description="Update a farmer's details. Only provided fields will be updated."
)
async def update_farmer(
    farmer_id: UUID,
    farmer_data: FarmerUpdate,
    service: FarmerService = Depends(get_farmer_service),
    db: AsyncSession = Depends(get_db)
):
    farmer = await service.update_farmer(farmer_id, farmer_data)
    await db.commit()
    return farmer

@router.delete(
    "/{farmer_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a farmer (Admin only)",
    description="Hard delete a farmer by their UUID."
)
async def delete_farmer(
    farmer_id: UUID,
    current_user: UserAccount = Depends(require_admin),
    service: FarmerService = Depends(get_farmer_service),
    db: AsyncSession = Depends(get_db)
):
    await service.delete_farmer(farmer_id)
    await db.commit()
