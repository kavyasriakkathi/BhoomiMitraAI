"""
BhoomiMitra AI — Startup Pilot Analytics Router

Exposes read-only administrative endpoints for real-time pilot monitoring and KPIs.
Protected by require_admin RBAC dependency.
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.auth.dependencies import require_admin
from src.core.models import UserAccount
from src.analytics.schemas import AnalyticsSummaryResponse, AnalyticsActivityResponse
from src.analytics.service import AnalyticsService

router = APIRouter()


def get_analytics_service(db: AsyncSession = Depends(get_db)) -> AnalyticsService:
    return AnalyticsService(db)


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pilot Analytics KPI Summary",
    description="Retrieve high-level pilot KPIs: DAU, WAU, modality, language, escalation, and delivery health (Admin only).",
)
async def get_analytics_summary(
    service: AnalyticsService = Depends(get_analytics_service),
    current_user: UserAccount = Depends(require_admin),
):
    return await service.get_summary()


@router.get(
    "/activity",
    response_model=AnalyticsActivityResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pilot Daily Activity Time-Series",
    description="Retrieve daily active farmers and message modality trends over past N days (Admin only).",
)
async def get_analytics_activity(
    days: int = Query(7, ge=1, le=30, description="Number of days to retrieve"),
    service: AnalyticsService = Depends(get_analytics_service),
    current_user: UserAccount = Depends(require_admin),
):
    return await service.get_activity(days=days)
