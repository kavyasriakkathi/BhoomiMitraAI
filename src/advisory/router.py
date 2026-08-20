from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.advisory.schemas import AdvisoryCreate, AdvisoryUpdate, AdvisoryResponse, AdvisoryListResponse
from src.advisory.service import AdvisoryService
from src.advisory.dependencies import get_advisory_service

router = APIRouter()

@router.post("", response_model=AdvisoryResponse, status_code=status.HTTP_201_CREATED,
    summary="Register a new advisory", description="Create a new advisory record.")
async def create_advisory(data: AdvisoryCreate, service: AdvisoryService = Depends(get_advisory_service), db: AsyncSession = Depends(get_db)):
    advisory = await service.create(data)
    await db.commit()
    return advisory

@router.get("", response_model=AdvisoryListResponse, status_code=status.HTTP_200_OK,
    summary="List advisories", description="Retrieve a paginated list of all advisories.")
async def get_advisories(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: AdvisoryService = Depends(get_advisory_service)):
    total, items = await service.list(page=page, size=size)
    return AdvisoryListResponse(total=total, items=items, page=page, size=size)

@router.get("/farmer/{farmer_id}", response_model=AdvisoryListResponse, status_code=status.HTTP_200_OK,
    summary="List advisories for a farmer", description="Retrieve advisories belonging to a specific farmer.")
async def get_farmer_advisories(farmer_id: UUID, page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100),
    service: AdvisoryService = Depends(get_advisory_service)):
    total, items = await service.list_by_farmer(farmer_id, page=page, size=size)
    return AdvisoryListResponse(total=total, items=items, page=page, size=size)

@router.get("/{advisory_id}", response_model=AdvisoryResponse, status_code=status.HTTP_200_OK,
    summary="Get an advisory", description="Retrieve a single advisory by its UUID.")
async def get_advisory(advisory_id: UUID, service: AdvisoryService = Depends(get_advisory_service)):
    return await service.get_by_id(advisory_id)

@router.put("/{advisory_id}", response_model=AdvisoryResponse, status_code=status.HTTP_200_OK,
    summary="Update an advisory", description="Update advisory details. Only provided fields will be updated.")
async def update_advisory(advisory_id: UUID, data: AdvisoryUpdate, service: AdvisoryService = Depends(get_advisory_service), db: AsyncSession = Depends(get_db)):
    advisory = await service.update(advisory_id, data)
    await db.commit()
    return advisory

@router.delete("/{advisory_id}", status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an advisory", description="Hard delete an advisory record by its UUID.")
async def delete_advisory(advisory_id: UUID, service: AdvisoryService = Depends(get_advisory_service), db: AsyncSession = Depends(get_db)):
    await service.delete(advisory_id)
    await db.commit()
