from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class GovernmentSchemeCreate(BaseModel):
    scheme_name: str = Field(..., min_length=2, max_length=200)
    scheme_code: str = Field(..., min_length=2, max_length=50)
    category: str = Field(..., max_length=100)
    state: str = Field(default="All India", max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    crop_type: str = Field(default="All Crops", max_length=100)
    min_land_acres: float = Field(default=0.0, ge=0.0)
    max_land_acres: Optional[float] = Field(None, ge=0.0)
    description: str
    benefits_summary: str
    eligibility_criteria: str
    required_documents: str
    application_deadline: Optional[datetime] = None
    official_portal_url: Optional[str] = None


class GovernmentSchemeResponse(BaseModel):
    id: UUID
    scheme_name: str
    scheme_code: str
    category: str
    state: str
    district: Optional[str] = None
    crop_type: str
    min_land_acres: float
    max_land_acres: Optional[float] = None
    description: str
    benefits_summary: str
    eligibility_criteria: str
    required_documents: str
    application_deadline: Optional[datetime] = None
    official_portal_url: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SchemeEligibilityItem(BaseModel):
    scheme: GovernmentSchemeResponse
    is_eligible: bool
    match_score_percentage: int
    eligibility_reason: str
    recommended_action: str
    voice_explanation: str


class FarmerEligibilityResponse(BaseModel):
    farmer_id: UUID
    farmer_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    land_size_acres: Optional[float] = None
    total_schemes_evaluated: int
    eligible_schemes_count: int
    schemes: List[SchemeEligibilityItem]


class SchemeApplicationCreate(BaseModel):
    farmer_id: UUID
    scheme_id: UUID
    notes: Optional[str] = None


class SchemeApplicationResponse(BaseModel):
    id: UUID
    farmer_id: UUID
    scheme_id: UUID
    status: str
    notes: Optional[str] = None
    scheme_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
