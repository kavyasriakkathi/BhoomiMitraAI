"""
BhoomiMitra AI — Startup Pilot Analytics Router

Exposes read-only administrative endpoints for real-time pilot monitoring and KPIs:
- GET /analytics/summary: Core KPI cards, breakdowns (modality, intent, language, delivery, escalation), and top rankings.
- GET /analytics/activity: Time-series trend over day/week/month or custom number of days.
- GET /analytics/conversations: Paginated interaction history with timings, farmer identifier, query, reply, intent, and delivery health.

Protected by require_admin RBAC dependency.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.auth.dependencies import require_admin
from src.core.models import UserAccount
from src.analytics.schemas import (
    AnalyticsSummaryResponse,
    AnalyticsActivityResponse,
    PaginatedConversationsResponse,
)
from src.analytics.service import AnalyticsService

router = APIRouter()


def get_analytics_service(db: AsyncSession = Depends(get_db)) -> AnalyticsService:
    return AnalyticsService(db)


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pilot Analytics KPI Summary",
    description="Retrieve high-level pilot KPIs: DAU, WAU, response times, modality, language, intent, escalation, rankings, and delivery health (Admin only).",
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
    description="Retrieve daily active farmers and message modality trends over day/week/month periods or past N days (Admin only).",
)
async def get_analytics_activity(
    days: int = Query(7, ge=1, le=90, description="Number of days to retrieve"),
    period: Optional[str] = Query(None, description="Period shorthand: 'day', 'week', or 'month'"),
    service: AnalyticsService = Depends(get_analytics_service),
    current_user: UserAccount = Depends(require_admin),
):
    return await service.get_activity(days=days, period=period)


@router.get(
    "/conversations",
    response_model=PaginatedConversationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Paginated Conversation Analytics",
    description="Retrieve paginated conversation history with farmer identifier, timings, intent, modality, and response times (Admin only).",
)
async def get_analytics_conversations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    farmer_id: Optional[UUID] = Query(None, description="Filter by farmer UUID"),
    intent: Optional[str] = Query(None, description="Filter by intent keyword or category"),
    modality: Optional[str] = Query(None, description="Filter by modality (text, audio, image)"),
    delivery_status: Optional[str] = Query(None, description="Filter by delivery status"),
    search: Optional[str] = Query(None, description="Search term across user message or AI response"),
    service: AnalyticsService = Depends(get_analytics_service),
    current_user: UserAccount = Depends(require_admin),
):
    return await service.get_conversations(
        page=page,
        page_size=page_size,
        farmer_id=farmer_id,
        intent=intent,
        modality=modality,
        delivery_status=delivery_status,
        search=search,
    )
