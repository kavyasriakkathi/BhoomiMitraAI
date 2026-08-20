from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.schemes.schemas import (
    GovernmentSchemeCreate,
    GovernmentSchemeResponse,
    FarmerEligibilityResponse,
    SchemeApplicationCreate,
    SchemeApplicationResponse,
)
from src.schemes.service import SchemeService
from src.schemes.dependencies import get_scheme_service

router = APIRouter()


@router.get(
    "",
    response_model=List[GovernmentSchemeResponse],
    status_code=status.HTTP_200_OK,
    summary="List active government schemes",
    description="Retrieve all active national and state agriculture schemes with optional state and category filtering.",
)
async def list_government_schemes(
    state: Optional[str] = Query(None, description="Filter by state (e.g. Telangana, Andhra Pradesh)"),
    category: Optional[str] = Query(None, description="Filter by category (e.g. Subsidy, Crop Insurance)"),
    service: SchemeService = Depends(get_scheme_service),
):
    return await service.list_schemes(state=state, category=category)


@router.post(
    "",
    response_model=GovernmentSchemeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new government scheme",
    description="Register a new national or state government agriculture scheme.",
)
async def create_government_scheme(
    data: GovernmentSchemeCreate,
    service: SchemeService = Depends(get_scheme_service),
):
    return await service.create_scheme(data)


@router.get(
    "/eligibility/{farmer_id}",
    response_model=FarmerEligibilityResponse,
    status_code=status.HTTP_200_OK,
    summary="AI Government Schemes Eligibility Assessment",
    description="Automatically evaluates a farmer's profile (state, district, land size) against all national and state government schemes.",
)
async def evaluate_farmer_eligibility(
    farmer_id: UUID,
    service: SchemeService = Depends(get_scheme_service),
):
    return await service.evaluate_farmer_eligibility(farmer_id)


@router.post(
    "/apply",
    response_model=SchemeApplicationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Apply / Bookmark a government scheme",
    description="Submit an application or bookmark a government scheme for a farmer.",
)
async def apply_for_scheme(
    data: SchemeApplicationCreate,
    service: SchemeService = Depends(get_scheme_service),
):
    return await service.apply_for_scheme(data)


@router.get(
    "/farmer-applications/{farmer_id}",
    response_model=List[SchemeApplicationResponse],
    status_code=status.HTTP_200_OK,
    summary="Get farmer scheme applications",
    description="Retrieve all government scheme applications submitted by a farmer.",
)
async def get_farmer_applications(
    farmer_id: UUID,
    service: SchemeService = Depends(get_scheme_service),
):
    return await service.get_farmer_applications(farmer_id)
