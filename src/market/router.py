"""
BhoomiMitra AI — Market Prices Router

REST endpoints for mandi price queries and admin seeding.
Follows the same pattern as src/schemes/router.py.

POST /market/prices  is intentionally protected by a simple admin token
check (header X-Admin-Token) since no full auth system exists in this project.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Header, HTTPException, status

from src.market.schemas import (
    MarketPriceCreate,
    MarketPriceResponse,
    MarketPriceQueryResponse,
)
from src.market.service import MarketService
from src.market.dependencies import get_market_service
from src.config import get_settings

router = APIRouter()


# ------------------------------------------------------------------
# GET /market/prices — Query mandi prices
# ------------------------------------------------------------------

@router.get(
    "/prices",
    response_model=MarketPriceQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query mandi prices for a commodity",
    description=(
        "Returns the most recent mandi prices for the given commodity. "
        "Tries the live Agmarknet API first, falls back to local DB. "
        "Prefers district-level data; falls back to state-level if unavailable."
    ),
)
async def get_market_prices(
    commodity: str = Query(..., description="Commodity name, e.g. 'Tomato', 'Paddy'"),
    district: Optional[str] = Query(None, description="Filter by district, e.g. 'Warangal'"),
    state: Optional[str] = Query(None, description="Filter by state, e.g. 'Telangana'"),
    service: MarketService = Depends(get_market_service),
) -> MarketPriceQueryResponse:
    return await service.get_prices_for_query(
        commodity=commodity,
        district=district,
        state=state,
    )


# ------------------------------------------------------------------
# GET /market/prices/commodities — List available commodities
# ------------------------------------------------------------------

@router.get(
    "/prices/commodities",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List all commodities with stored price data",
    description="Returns distinct commodity names currently in the local database.",
)
async def list_commodities(
    service: MarketService = Depends(get_market_service),
) -> List[str]:
    return await service.list_commodities()


# ------------------------------------------------------------------
# POST /market/prices — Admin-only: manually seed a price record
# ------------------------------------------------------------------

@router.post(
    "/prices",
    response_model=MarketPriceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="[Admin] Manually seed a market price record",
    description=(
        "Inserts a single market price record into the local database. "
        "Requires the X-Admin-Token header matching WHATSAPP_VERIFY_TOKEN "
        "(the simplest available secret in this project). "
        "Skips insertion if the exact same commodity + market + date already exists."
    ),
)
async def create_market_price(
    data: MarketPriceCreate,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    service: MarketService = Depends(get_market_service),
) -> MarketPriceResponse:
    # Simple token check using the existing WHATSAPP_VERIFY_TOKEN secret
    # (no dedicated auth system exists in this project)
    settings = get_settings()
    expected = settings.whatsapp_verify_token
    if not expected or x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid X-Admin-Token header required.",
        )

    result = await service.create_price(data)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A price record for this commodity, market, and date already exists.",
        )
    return result
