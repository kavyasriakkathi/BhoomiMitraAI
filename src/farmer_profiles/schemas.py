from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

class FarmerProfileBase(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=50)
    district: Optional[str] = Field(None, max_length=50)
    current_crop: Optional[str] = Field(None, max_length=100)
    land_size_acres: Optional[float] = Field(None, ge=0)

class FarmerProfileCreate(FarmerProfileBase):
    farmer_id: UUID

class FarmerProfileUpdate(FarmerProfileBase):
    pass

class FarmerProfileResponse(FarmerProfileBase):
    id: UUID
    farmer_id: UUID

    model_config = ConfigDict(from_attributes=True)

class PaginatedFarmerProfileResponse(BaseModel):
    total: int
    items: list[FarmerProfileResponse]
    page: int
    size: int
