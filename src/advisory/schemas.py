from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

class AdvisoryBase(BaseModel):
    advisory_type: Optional[str] = Field(None, max_length=50)
    message: str = Field(..., min_length=1)
    source: Optional[str] = Field(None, max_length=100)

    @field_validator("advisory_type", "message")
    @classmethod
    def validate_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip() if v else v

class AdvisoryCreate(AdvisoryBase):
    farmer_id: UUID

class AdvisoryUpdate(BaseModel):
    advisory_type: Optional[str] = Field(None, max_length=50)
    message: Optional[str] = Field(None, min_length=1)
    source: Optional[str] = Field(None, max_length=100)

    @field_validator("advisory_type", "message")
    @classmethod
    def validate_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip() if v else v

class AdvisoryResponse(AdvisoryBase):
    id: UUID
    farmer_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AdvisoryListResponse(BaseModel):
    total: int
    items: list[AdvisoryResponse]
    page: int
    size: int
