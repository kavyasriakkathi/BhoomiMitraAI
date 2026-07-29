from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.shops.schemas import (
    ShopCreate,
    ShopUpdate,
    ShopResponse,
    ShopSearchResponse,
    PaginatedShopResponse,
    FarmerShopSearchResponse,
)
from src.shops.service import ShopService
from src.shops.dependencies import get_shop_service

router = APIRouter()


@router.post("", response_model=ShopResponse, status_code=status.HTTP_201_CREATED)
async def create_shop(
    payload: ShopCreate,
    service: ShopService = Depends(get_shop_service),
):
    """Register a new Agri Shop."""
    return await service.create_shop(payload)


@router.get("/farmer-search", response_model=FarmerShopSearchResponse)
async def farmer_shop_search(
    query: str = Query(..., description="Product or category search query, e.g., 'Urea' or 'Pesticide'"),
    latitude: Optional[float] = Query(None, description="Farmer latitude coordinate"),
    longitude: Optional[float] = Query(None, description="Farmer longitude coordinate"),
    district: Optional[str] = Query(None, description="Farmer district for fallback match"),
    service: ShopService = Depends(get_shop_service),
):
    """Farmer search interface: 'I need Urea' returns available nearby shops with stock & price."""
    return await service.farmer_product_search(
        product_query=query,
        farmer_latitude=latitude,
        farmer_longitude=longitude,
        district=district,
    )


@router.get("/nearby", response_model=List[ShopSearchResponse])
async def get_nearby_shops(
    latitude: float = Query(..., description="Farmer latitude coordinate"),
    longitude: float = Query(..., description="Farmer longitude coordinate"),
    max_radius_km: float = Query(50.0, description="Max search radius in kilometers"),
    service: ShopService = Depends(get_shop_service),
):
    """Find nearby active Agri shops ordered by distance (Haversine formula)."""
    return await service.get_nearby_shops(
        latitude=latitude, longitude=longitude, max_radius_km=max_radius_km
    )


@router.get("/search/location", response_model=List[ShopResponse])
async def search_shops_by_location(
    district: Optional[str] = Query(None),
    mandal: Optional[str] = Query(None),
    village: Optional[str] = Query(None),
    pin_code: Optional[str] = Query(None),
    service: ShopService = Depends(get_shop_service),
):
    """Search registered shops by geographic region (District, Mandal, Village, PIN Code)."""
    return await service.search_by_location(
        district=district, mandal=mandal, village=village, pin_code=pin_code
    )


@router.get("/search/product", response_model=FarmerShopSearchResponse)
async def search_shops_by_product(
    product_name: str = Query(..., description="Product name or brand"),
    service: ShopService = Depends(get_shop_service),
):
    """Search active shops stocking a specific product."""
    return await service.farmer_product_search(product_query=product_name)


@router.get("/{shop_id}", response_model=ShopResponse)
async def get_shop(
    shop_id: UUID,
    service: ShopService = Depends(get_shop_service),
):
    """Get shop details by Shop ID."""
    return await service.get_shop_by_id(shop_id)


@router.put("/{shop_id}", response_model=ShopResponse)
async def update_shop(
    shop_id: UUID,
    payload: ShopUpdate,
    service: ShopService = Depends(get_shop_service),
):
    """Update shop details."""
    return await service.update_shop(shop_id, payload)


@router.delete("/{shop_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_shop(
    shop_id: UUID,
    service: ShopService = Depends(get_shop_service),
):
    """Delete a shop profile."""
    await service.delete_shop(shop_id)


@router.get("", response_model=PaginatedShopResponse)
async def list_shops(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by shop status e.g. active/inactive"),
    service: ShopService = Depends(get_shop_service),
):
    """List all registered Agri shops with pagination."""
    return await service.list_shops(page=page, size=size, status_filter=status)
