from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

VALID_SEASONS = {"kharif", "rabi", "zaid"}
VALID_STATUSES = {"planned", "growing", "harvested"}

class CropBase(BaseModel):
    crop_name: str = Field(..., min_length=1, max_length=100)
    variety: Optional[str] = Field(None, max_length=100)
    sowing_date: Optional[datetime] = None
    season: Optional[str] = Field(None, max_length=50)
    status: Optional[str] = Field(None, max_length=50)

    @field_validator("crop_name")
    @classmethod
    def validate_crop_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("crop_name cannot be empty.")
        return v.strip()

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_SEASONS:
            raise ValueError(f"Invalid season. Allowed values: {', '.join(VALID_SEASONS)}")
        return v.capitalize() if v else v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_STATUSES:
            raise ValueError(f"Invalid status. Allowed values: {', '.join(VALID_STATUSES)}")
        return v.lower() if v else v

class CropCreate(CropBase):
    farm_id: UUID

class CropUpdate(BaseModel):
    crop_name: Optional[str] = Field(None, min_length=1, max_length=100)
    variety: Optional[str] = Field(None, max_length=100)
    sowing_date: Optional[datetime] = None
    season: Optional[str] = Field(None, max_length=50)
    status: Optional[str] = Field(None, max_length=50)

    @field_validator("crop_name")
    @classmethod
    def validate_crop_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("crop_name cannot be empty.")
        return v.strip() if v else v

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_SEASONS:
            raise ValueError(f"Invalid season. Allowed values: {', '.join(VALID_SEASONS)}")
        return v.capitalize() if v else v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_STATUSES:
            raise ValueError(f"Invalid status. Allowed values: {', '.join(VALID_STATUSES)}")
        return v.lower() if v else v

class CropResponse(CropBase):
    id: UUID
    farm_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PaginatedCropResponse(BaseModel):
    total: int
    items: list[CropResponse]
    page: int
    size: int
