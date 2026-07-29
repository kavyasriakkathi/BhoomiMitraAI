from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
from uuid import UUID


class OrderRequestBase(BaseModel):
    farmer_id: UUID = Field(..., description="Farmer ID placing the request")
    shop_id: UUID = Field(..., description="Shop ID receiving the request")
    inventory_id: UUID = Field(..., description="Product Inventory ID")
    quantity: int = Field(1, ge=1, description="Quantity requested")
    notes: Optional[str] = Field(None, description="Optional delivery notes or instructions")


class OrderRequestCreate(OrderRequestBase):
    pass


class OrderRequestUpdateStatus(BaseModel):
    status: str = Field(..., description="New status: Pending, Accepted, Ready, Completed, Cancelled")
    notes: Optional[str] = Field(None, description="Optional status update note")


class OrderRequestResponse(OrderRequestBase):
    id: UUID
    product_name: str
    brand: Optional[str] = None
    unit: str
    unit_price: float
    total_price: float
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedOrderRequestResponse(BaseModel):
    items: List[OrderRequestResponse]
    total: int
    page: int
    size: int


class SalesAnalyticsResponse(BaseModel):
    shop_id: UUID
    total_orders: int
    pending_orders: int
    accepted_orders: int
    ready_orders: int
    completed_orders: int
    cancelled_orders: int
    total_revenue_inr: float
    popular_products: List[Dict[str, str]]
    category_demand: List[Dict[str, str]]
