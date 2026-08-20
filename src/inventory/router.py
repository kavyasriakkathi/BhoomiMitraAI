from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.inventory.schemas import (
    InventoryCreate,
    InventoryUpdate,
    StockUpdatePayload,
    InventoryResponse,
    PaginatedInventoryResponse,
    ShopDashboardSummaryResponse,
)
from src.inventory.service import InventoryService
from src.inventory.dependencies import get_inventory_service

router = APIRouter()


@router.post("", response_model=InventoryResponse, status_code=status.HTTP_201_CREATED)
async def add_product(
    payload: InventoryCreate,
    service: InventoryService = Depends(get_inventory_service),
):
    """Add a new product to an Agri Shop inventory."""
    return await service.add_product(payload)


@router.get("/search", response_model=List[InventoryResponse])
async def search_products(
    query: str = Query("", description="Search term for product name, brand, or description"),
    shop_id: Optional[UUID] = Query(None, description="Optional shop ID scope"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    brand: Optional[str] = Query(None, description="Optional brand filter"),
    service: InventoryService = Depends(get_inventory_service),
):
    """Search products by name, category, or brand."""
    return await service.search_products(
        query=query, shop_id=shop_id, category=category, brand=brand
    )


@router.get("/shop/{shop_id}", response_model=PaginatedInventoryResponse)
async def list_shop_products(
    shop_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    category: Optional[str] = Query(None, description="Filter by category"),
    service: InventoryService = Depends(get_inventory_service),
):
    """List all inventory products for a specific shop."""
    return await service.list_products_by_shop(
        shop_id=shop_id, page=page, size=size, category=category
    )


@router.get("/dashboard/{shop_id}", response_model=ShopDashboardSummaryResponse)
async def get_shop_dashboard(
    shop_id: UUID,
    service: InventoryService = Depends(get_inventory_service),
):
    """Shop Owner Dashboard: Summary of stock counts, low stock alerts, and out of stock items."""
    return await service.get_dashboard_summary(shop_id)


@router.get("/dashboard/{shop_id}/low-stock", response_model=List[InventoryResponse])
async def get_low_stock_products(
    shop_id: UUID,
    service: InventoryService = Depends(get_inventory_service),
):
    """View low stock items for a shop."""
    return await service.get_low_stock_products(shop_id)


@router.get("/dashboard/{shop_id}/out-of-stock", response_model=List[InventoryResponse])
async def get_out_of_stock_products(
    shop_id: UUID,
    service: InventoryService = Depends(get_inventory_service),
):
    """View out of stock items for a shop."""
    return await service.get_out_of_stock_products(shop_id)


@router.get("/{item_id}", response_model=InventoryResponse)
async def get_product(
    item_id: UUID,
    service: InventoryService = Depends(get_inventory_service),
):
    """Get inventory product by ID."""
    return await service.get_product_by_id(item_id)


@router.put("/{item_id}", response_model=InventoryResponse)
async def update_product(
    item_id: UUID,
    payload: InventoryUpdate,
    service: InventoryService = Depends(get_inventory_service),
):
    """Update inventory product details and price."""
    return await service.update_product(item_id, payload)


@router.patch("/{item_id}/stock", response_model=InventoryResponse)
async def update_stock(
    item_id: UUID,
    payload: StockUpdatePayload,
    service: InventoryService = Depends(get_inventory_service),
):
    """Update product stock quantity and availability."""
    return await service.update_stock(item_id, payload)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    item_id: UUID,
    service: InventoryService = Depends(get_inventory_service),
):
    """Delete an inventory product."""
    await service.delete_product(item_id)
