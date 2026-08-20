from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

class CropHealthBase(BaseModel):
    image_url: Optional[str] = Field(None, max_length=500)
    symptoms: str = Field(..., min_length=1)
    disease_name: Optional[str] = Field(None, max_length=100)
    diagnosis_result: str = Field(..., min_length=1)
    treatment_recommendation: str = Field(..., min_length=1)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("symptoms", "diagnosis_result", "treatment_recommendation")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()

class CropHealthCreate(CropHealthBase):
    crop_id: UUID
    farmer_id: UUID

class CropHealthUpdate(BaseModel):
    image_url: Optional[str] = Field(None, max_length=500)
    symptoms: Optional[str] = Field(None, min_length=1)
    disease_name: Optional[str] = Field(None, max_length=100)
    diagnosis_result: Optional[str] = Field(None, min_length=1)
    treatment_recommendation: Optional[str] = Field(None, min_length=1)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("symptoms", "diagnosis_result", "treatment_recommendation")
    @classmethod
    def validate_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip() if v else v

class CropHealthResponse(CropHealthBase):
    id: UUID
    crop_id: UUID
    farmer_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PaginatedCropHealthResponse(BaseModel):
    total: int
    items: list[CropHealthResponse]
    page: int
    size: int
