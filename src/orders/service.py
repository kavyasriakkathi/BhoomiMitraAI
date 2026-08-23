import asyncio
from typing import Optional, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.logging import logger
from src.orders.repository import OrderRepository
from src.orders.notifications import notify_farmer_order_update
from src.orders.schemas import (
    OrderRequestCreate,
    OrderRequestUpdateStatus,
    OrderRequestResponse,
    PaginatedOrderRequestResponse,
    SalesAnalyticsResponse,
)


class OrderService:
    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def create_order_request(self, data: OrderRequestCreate) -> OrderRequestResponse:
        try:
            order = await self.repository.create(data)
            logger.info(f"Created order request '{order.id}' for farmer '{order.farmer_id}' at shop '{order.shop_id}'")

            # Non-blocking asynchronous WhatsApp notification to farmer
            try:
                asyncio.create_task(notify_farmer_order_update(order.id, event="Created"))
            except Exception as notif_err:
                logger.warning(f"Could not schedule creation WhatsApp notification: {notif_err}")

            return OrderRequestResponse.model_validate(order)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err)
            )

    async def get_order_by_id(self, order_id: UUID) -> OrderRequestResponse:
        order = await self.repository.get_by_id(order_id)
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order request with ID '{order_id}' not found."
            )
        return OrderRequestResponse.model_validate(order)

    async def update_status(self, order_id: UUID, data: OrderRequestUpdateStatus) -> OrderRequestResponse:
        valid_statuses = ["Pending", "Accepted", "Ready", "Completed", "Cancelled"]
        if data.status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{data.status}'. Allowed: {', '.join(valid_statuses)}"
            )

        existing_order = await self.repository.get_by_id(order_id)
        if not existing_order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order request with ID '{order_id}' not found."
            )

        previous_status = existing_order.status

        try:
            order = await self.repository.update_status(order_id, data)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err)
            )

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order request with ID '{order_id}' not found."
            )

        # Idempotency check: Only notify farmer if the status actually changed
        if previous_status != data.status:
            try:
                asyncio.create_task(
                    notify_farmer_order_update(
                        order.id, event=data.status, reason=data.notes
                    )
                )
            except Exception as notif_err:
                logger.warning(f"Could not schedule status WhatsApp notification: {notif_err}")

        logger.info(f"Updated status of order '{order_id}' from '{previous_status}' to '{order.status}'")
        return OrderRequestResponse.model_validate(order)

    async def list_farmer_orders(
        self, farmer_id: UUID, page: int = 1, size: int = 20
    ) -> PaginatedOrderRequestResponse:
        orders, total = await self.repository.list_farmer_orders(farmer_id=farmer_id, page=page, size=size)
        items = [OrderRequestResponse.model_validate(o) for o in orders]
        return PaginatedOrderRequestResponse(items=items, total=total, page=page, size=size)

    async def list_shop_orders(
        self, shop_id: UUID, page: int = 1, size: int = 20, status_filter: Optional[str] = None
    ) -> PaginatedOrderRequestResponse:
        orders, total = await self.repository.list_shop_orders(
            shop_id=shop_id, page=page, size=size, status=status_filter
        )
        items = [OrderRequestResponse.model_validate(o) for o in orders]
        return PaginatedOrderRequestResponse(items=items, total=total, page=page, size=size)

    async def get_sales_analytics(self, shop_id: UUID) -> SalesAnalyticsResponse:
        data = await self.repository.get_sales_analytics(shop_id)
        return SalesAnalyticsResponse(**data)
