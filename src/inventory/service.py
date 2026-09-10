from typing import Optional, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.logging import logger
from src.inventory.repository import InventoryRepository
from src.inventory.schemas import (
    InventoryCreate,
    InventoryUpdate,
    StockUpdatePayload,
    InventoryResponse,
    PaginatedInventoryResponse,
    ShopDashboardSummaryResponse,
)


class InventoryService:
    def __init__(self, repository: InventoryRepository):
        self.repository = repository

    async def add_product(self, data: InventoryCreate) -> InventoryResponse:
        item = await self.repository.create(data)
        logger.info(f"Added new product '{item.product_name}' ({item.id}) to shop '{item.shop_id}'")
        if item.quantity_in_stock > 0 and item.available:
            try:
                import asyncio
                from src.shops.stock_alerts import trigger_stock_alert_notifications
                asyncio.create_task(
                    trigger_stock_alert_notifications(
                        inventory_item_id=item.id,
                        shop_id=item.shop_id,
                        product_name=item.product_name,
                        new_quantity=item.quantity_in_stock,
                        unit=item.unit,
                        brand=item.brand,
                    )
                )
            except Exception as alert_err:
                logger.warning(f"Stock alert background dispatch error: {alert_err}")
        return InventoryResponse.model_validate(item)

    async def get_product_by_id(self, item_id: UUID) -> InventoryResponse:
        item = await self.repository.get_by_id(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        return InventoryResponse.model_validate(item)

    async def update_product(self, item_id: UUID, data: InventoryUpdate) -> InventoryResponse:
        existing = await self.repository.get_by_id(item_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        prev_qty = existing.quantity_in_stock
        prev_avail = existing.available

        item = await self.repository.update(item_id, data)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        logger.info(f"Updated product '{item.product_name}' ({item_id})")

        # Check restock transition
        if (prev_qty <= 0 or not prev_avail) and item.quantity_in_stock > 0 and item.available:
            try:
                import asyncio
                from src.shops.stock_alerts import trigger_stock_alert_notifications
                asyncio.create_task(
                    trigger_stock_alert_notifications(
                        inventory_item_id=item.id,
                        shop_id=item.shop_id,
                        product_name=item.product_name,
                        new_quantity=item.quantity_in_stock,
                        unit=item.unit,
                        brand=item.brand,
                    )
                )
            except Exception as alert_err:
                logger.warning(f"Stock alert background dispatch error: {alert_err}")

        return InventoryResponse.model_validate(item)

    async def update_stock(self, item_id: UUID, data: StockUpdatePayload) -> InventoryResponse:
        existing = await self.repository.get_by_id(item_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        prev_qty = existing.quantity_in_stock
        prev_avail = existing.available

        item = await self.repository.update_stock(item_id, data)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        logger.info(f"Updated stock for product '{item.product_name}' ({item_id}) to {item.quantity_in_stock}")

        # Transition: previous quantity <= 0 or unavailable -> new quantity > 0 and available
        if (prev_qty <= 0 or not prev_avail) and item.quantity_in_stock > 0 and item.available:
            try:
                import asyncio
                from src.shops.stock_alerts import trigger_stock_alert_notifications
                asyncio.create_task(
                    trigger_stock_alert_notifications(
                        inventory_item_id=item.id,
                        shop_id=item.shop_id,
                        product_name=item.product_name,
                        new_quantity=item.quantity_in_stock,
                        unit=item.unit,
                        brand=item.brand,
                    )
                )
            except Exception as alert_err:
                logger.warning(f"Stock alert background dispatch error: {alert_err}")

        return InventoryResponse.model_validate(item)

    async def delete_product(self, item_id: UUID) -> None:
        deleted = await self.repository.delete(item_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory product with ID '{item_id}' not found."
            )
        logger.info(f"Deleted product '{item_id}'")

    async def list_products_by_shop(
        self, shop_id: UUID, page: int = 1, size: int = 50, category: Optional[str] = None
    ) -> PaginatedInventoryResponse:
        items, total = await self.repository.list_products_by_shop(
            shop_id=shop_id, page=page, size=size, category=category
        )
        responses = [InventoryResponse.model_validate(i) for i in items]
        return PaginatedInventoryResponse(items=responses, total=total, page=page, size=size)

    async def search_products(
        self,
        query: str,
        shop_id: Optional[UUID] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> List[InventoryResponse]:
        items = await self.repository.search_products(
            query_str=query, shop_id=shop_id, category=category, brand=brand
        )
        return [InventoryResponse.model_validate(i) for i in items]

    async def get_low_stock_products(self, shop_id: UUID) -> List[InventoryResponse]:
        items = await self.repository.get_low_stock_products(shop_id)
        return [InventoryResponse.model_validate(i) for i in items]

    async def get_out_of_stock_products(self, shop_id: UUID) -> List[InventoryResponse]:
        items = await self.repository.get_out_of_stock_products(shop_id)
        return [InventoryResponse.model_validate(i) for i in items]

    async def get_dashboard_summary(self, shop_id: UUID) -> ShopDashboardSummaryResponse:
        items, total = await self.repository.list_products_by_shop(shop_id=shop_id, page=1, size=1000)
        low_stock = await self.repository.get_low_stock_products(shop_id)
        out_of_stock = await self.repository.get_out_of_stock_products(shop_id)

        available_count = len([i for i in items if i.available and i.quantity_in_stock > 0])

        return ShopDashboardSummaryResponse(
            shop_id=shop_id,
            total_products=total,
            available_products_count=available_count,
            low_stock_count=len(low_stock),
            out_of_stock_count=len(out_of_stock),
            low_stock_items=[InventoryResponse.model_validate(i) for i in low_stock],
            out_of_stock_items=[InventoryResponse.model_validate(i) for i in out_of_stock],
        )
