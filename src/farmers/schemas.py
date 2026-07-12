import re
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Strict E.164: mandatory '+', country code (1-3 digits), subscriber number
# Total digits after '+': 10-15 (realistic for mobile numbers with country code)
PHONE_REGEX = r"^\+[1-9]\d{9,14}$"

class FarmerBase(BaseModel):
    phone_number: str = Field(..., pattern=PHONE_REGEX, description="Strict E.164 phone number: must start with '+' followed by 10-15 digits")
    preferred_language: str = Field(default="te", max_length=10)
    is_active: bool = Field(default=True)

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        if not re.match(PHONE_REGEX, v):
            raise ValueError(
                "Invalid phone number. Must start with '+' followed by 10-15 digits "
                "including country code (e.g. +919876543210)."
            )
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
            raise ValueError(
                "Invalid phone number. Must start with '+' followed by 10-15 digits "
                "including country code (e.g. +919876543210)."
            )
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
