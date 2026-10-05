"""
BhoomiMitra AI — Shop Discovery Schemas
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DiscoveredShopItem(BaseModel):
    """Normalized schema for an externally discovered agricultural shop."""
    provider: str
    provider_place_id: str
    shop_name: str
    business_type: Optional[str] = "agricultural_input_store"
    address: Optional[str] = None
    latitude: float
    longitude: float
    phone_number: Optional[str] = None
    maps_url: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    is_operational: bool = True
    distance_km: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None

    model_config = {"extra": "ignore"}
