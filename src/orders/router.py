from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.orders.schemas import (
    OrderRequestCreate,
    OrderRequestUpdateStatus,
    OrderRequestResponse,
    PaginatedOrderRequestResponse,
    SalesAnalyticsResponse,
)
from src.orders.service import OrderService
from src.orders.dependencies import get_order_service
from src.auth.dependencies import require_shop_owner, verify_shop_access
from src.core.models import UserAccount

router = APIRouter()


@router.post("", response_model=OrderRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_order_request(
    payload: OrderRequestCreate,
    service: OrderService = Depends(get_order_service),
):
    """Farmer Cart: Submit a purchase request to an Agri Shop."""
    return await service.create_order_request(payload)


@router.get("/farmer/{farmer_id}", response_model=PaginatedOrderRequestResponse)
async def list_farmer_orders(
    farmer_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    service: OrderService = Depends(get_order_service),
):
    """List purchase requests submitted by a farmer."""
    return await service.list_farmer_orders(farmer_id=farmer_id, page=page, size=size)


@router.get("/shop/{shop_id}", response_model=PaginatedOrderRequestResponse)
async def list_shop_orders(
    shop_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter status: Pending, Accepted, Ready, Completed, Cancelled"),
    current_user: UserAccount = Depends(require_shop_owner),
    service: OrderService = Depends(get_order_service),
):
    """Shop Owner Dashboard: List incoming farmer purchase orders."""
    verify_shop_access(current_user, shop_id)
    return await service.list_shop_orders(shop_id=shop_id, page=page, size=size, status_filter=status)


@router.get("/analytics/{shop_id}", response_model=SalesAnalyticsResponse)
async def get_sales_analytics(
    shop_id: UUID,
    current_user: UserAccount = Depends(require_shop_owner),
    service: OrderService = Depends(get_order_service),
):
    """Shop Owner Analytics: Sales trends, revenue, and popular product demand."""
    verify_shop_access(current_user, shop_id)
    return await service.get_sales_analytics(shop_id)


@router.get("/{order_id}", response_model=OrderRequestResponse)
async def get_order(
    order_id: UUID,
    service: OrderService = Depends(get_order_service),
):
    """Get purchase order request details by ID."""
    return await service.get_order_by_id(order_id)


@router.patch("/{order_id}/status", response_model=OrderRequestResponse)
async def update_order_status(
    order_id: UUID,
    payload: OrderRequestUpdateStatus,
    current_user: UserAccount = Depends(require_shop_owner),
    service: OrderService = Depends(get_order_service),
):
    """Update order status (Pending -> Accepted -> Ready -> Completed / Cancelled)."""
    order = await service.get_order_by_id(order_id)
    verify_shop_access(current_user, order.shop_id)
    return await service.update_status(order_id, payload)

