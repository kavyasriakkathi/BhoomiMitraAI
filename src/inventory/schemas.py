from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class ProductCategoryEnum(str, Enum):
    SEEDS = "Seeds"
    FERTILIZERS = "Fertilizers"
    PESTICIDES = "Pesticides"
    FUNGICIDES = "Fungicides"
    HERBICIDES = "Herbicides"
    MICRONUTRIENTS = "Micronutrients"
    BIO_FERTILIZERS = "Bio Fertilizers"
    ORGANIC_PRODUCTS = "Organic Products"
    FARM_EQUIPMENT = "Farm Equipment"
    IRRIGATION = "Irrigation"
    ANIMAL_FEED = "Animal Feed"
    VETERINARY_MEDICINES = "Veterinary Medicines"


class InventoryBase(BaseModel):
    shop_id: UUID = Field(..., description="ID of the parent shop")
    product_name: str = Field(..., max_length=150, description="Product Name e.g. Urea")
    category: str = Field(..., max_length=100, description="Product Category e.g. Fertilizers")
    brand: str = Field(..., max_length=100, description="Brand e.g. IFFCO, Bayer, Coromandel")
    product_description: Optional[str] = Field(None, description="Detailed product description")
    unit: str = Field("Unit", max_length=50, description="Packaging unit e.g. Bag, Bottle, Kg, Litre")
    price: float = Field(..., ge=0.0, description="Regular selling price in INR")
    discount_price: Optional[float] = Field(None, ge=0.0, description="Discounted price in INR")
    quantity_in_stock: int = Field(0, ge=0, description="Current stock quantity available")
    minimum_stock_level: int = Field(5, ge=0, description="Threshold for low stock alert")
    available: bool = Field(True, description="Available for sale status")
    expiry_date: Optional[datetime] = Field(None, description="Expiration date if applicable")


class InventoryCreate(InventoryBase):
    pass


class InventoryUpdate(BaseModel):
    product_name: Optional[str] = Field(None, max_length=150)
    category: Optional[str] = Field(None, max_length=100)
    brand: Optional[str] = Field(None, max_length=100)
    product_description: Optional[str] = None
    unit: Optional[str] = Field(None, max_length=50)
    price: Optional[float] = Field(None, ge=0.0)
    discount_price: Optional[float] = Field(None, ge=0.0)
    quantity_in_stock: Optional[int] = Field(None, ge=0)
    minimum_stock_level: Optional[int] = Field(None, ge=0)
    available: Optional[bool] = None
    expiry_date: Optional[datetime] = None


class StockUpdatePayload(BaseModel):
    quantity_in_stock: int = Field(..., ge=0, description="New stock quantity")
    available: Optional[bool] = Field(None, description="Optionally toggle availability status")


class InventoryResponse(InventoryBase):
    id: UUID
    last_updated: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedInventoryResponse(BaseModel):
    items: List[InventoryResponse]
    total: int
    page: int
    size: int


class ShopDashboardSummaryResponse(BaseModel):
    shop_id: UUID
    total_products: int
    available_products_count: int
    low_stock_count: int
    out_of_stock_count: int
    low_stock_items: List[InventoryResponse]
    out_of_stock_items: List[InventoryResponse]
