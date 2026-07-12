from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, constr, ConfigDict

class FarmerBase(BaseModel):
    phone_number: str = Field(..., pattern=r"^\+?[1-9]\d{1,14}$", description="E.164 standard phone number format")
    preferred_language: str = Field(default="te", max_length=10)
    is_active: bool = Field(default=True)

class FarmerCreate(FarmerBase):
    pass

class FarmerUpdate(BaseModel):
    phone_number: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$")
    preferred_language: Optional[str] = Field(None, max_length=10)
    is_active: Optional[bool] = None

class FarmerResponse(FarmerBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PaginatedFarmerResponse(BaseModel):
    total: int
    items: list[FarmerResponse]
    page: int
    size: int
