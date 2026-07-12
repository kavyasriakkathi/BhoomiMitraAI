import re
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# E.164: optional '+', first digit 1-9, then 6-14 more digits (7-15 total)
PHONE_REGEX = r"^\+?[1-9]\d{6,14}$"

class FarmerBase(BaseModel):
    phone_number: str = Field(..., pattern=PHONE_REGEX, description="E.164 standard phone number format (7-15 digits)")
    preferred_language: str = Field(default="te", max_length=10)
    is_active: bool = Field(default=True)

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        if not re.match(PHONE_REGEX, v):
            raise ValueError("Invalid phone number. Must be 7-15 digits in E.164 format (e.g. +919876543210).")
        return v

class FarmerCreate(FarmerBase):
    pass

class FarmerUpdate(BaseModel):
    phone_number: Optional[str] = Field(None, pattern=PHONE_REGEX)
    preferred_language: Optional[str] = Field(None, max_length=10)
    is_active: Optional[bool] = None

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.match(PHONE_REGEX, v):
            raise ValueError("Invalid phone number. Must be 7-15 digits in E.164 format (e.g. +919876543210).")
        return v

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
