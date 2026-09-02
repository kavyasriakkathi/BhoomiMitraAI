"""
BhoomiMitra AI — Market Prices Schemas

Pydantic models for market price request/response validation.
Follows the same pattern as src/schemes/schemas.py.
"""
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class MarketPriceCreate(BaseModel):
    """Used for the admin-only POST /market/prices endpoint."""
    commodity: str = Field(..., min_length=1, max_length=100)
    commodity_telugu: Optional[str] = Field(None, max_length=100)
    market_name: str = Field(..., min_length=1, max_length=150)
    district: str = Field(..., min_length=1, max_length=100)
    state: str = Field(..., min_length=1, max_length=100)
    min_price: float = Field(..., ge=0.0)
    max_price: float = Field(..., ge=0.0)
    modal_price: float = Field(..., ge=0.0)
    unit: str = Field(default="Quintal", max_length=20)
    price_date: datetime
    source: str = Field(default="manual_seed", max_length=50)


class MarketPriceResponse(BaseModel):
    """Public-facing market price record response."""
    id: UUID
    commodity: str
    commodity_telugu: Optional[str] = None
    market_name: str
    district: str
    state: str
    min_price: float
    max_price: float
    modal_price: float
    unit: str
    price_date: datetime
    source: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MarketPriceQueryResponse(BaseModel):
    """
    Response returned when a farmer asks about mandi prices.
    Includes data freshness metadata and source notes so the
    farmer always knows whether they are seeing live or cached data.
    """
    commodity: str
    district: Optional[str] = None
    state: Optional[str] = None
    results: List[MarketPriceResponse]
    data_available: bool
    data_freshness_hours: Optional[float] = None   # How old is the newest record?
    source_note: str                                # Human-readable freshness note
    is_live: bool                                   # True = from API, False = from local DB or cache
    is_today_requested: bool = False                # True if user specifically queried for today's price
