from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID

class ShopBase(BaseModel):
    shop_name: str = Field(..., max_length=150, description="Name of the Agri Shop")
    owner_name: str = Field(..., max_length=100, description="Owner's full name")
    phone_number: str = Field(..., max_length=20, description="Contact phone number")
    email: Optional[str] = Field(None, max_length=100, description="Optional email address")
    address: str = Field(..., description="Full street address")
    village: Optional[str] = Field(None, max_length=100)
    mandal: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    pin_code: Optional[str] = Field(None, max_length=20)
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    opening_time: str = Field("08:00", description="Opening time e.g. 08:00 AM")
    closing_time: str = Field("20:00", description="Closing time e.g. 08:00 PM")
    delivery_available: bool = Field(False, description="Home delivery service status")
    home_delivery_radius_km: Optional[float] = Field(None, ge=0.0, description="Delivery radius in km")
    google_maps_link: Optional[str] = Field(None, max_length=500)
    gst_number: Optional[str] = Field(None, max_length=50)
    license_number: Optional[str] = Field(None, max_length=50)
    status: str = Field("active", description="Shop status: active / inactive")


class ShopCreate(ShopBase):
    pass


class ShopUpdate(BaseModel):
    shop_name: Optional[str] = Field(None, max_length=150)
    owner_name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = None
    village: Optional[str] = None
    mandal: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pin_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    delivery_available: Optional[bool] = None
    home_delivery_radius_km: Optional[float] = None
    google_maps_link: Optional[str] = None
    gst_number: Optional[str] = None
    license_number: Optional[str] = None
    status: Optional[str] = None


class ShopResponse(ShopBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShopSearchResponse(ShopResponse):
    distance_km: Optional[float] = None


class PaginatedShopResponse(BaseModel):
    items: List[ShopResponse]
    total: int
    page: int
    size: int


class FarmerShopSearchResult(BaseModel):
    shop_id: UUID
    shop_name: str
    owner_name: str
    distance_km: Optional[float] = None
    product_name: str
    brand: str
    price: float
    discount_price: Optional[float] = None
    unit: str
    quantity_in_stock: int
    phone_number: str
    opening_time: str
    closing_time: str
    status: str
    delivery_available: bool
    formatted_display: str


class FarmerShopSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[FarmerShopSearchResult]
