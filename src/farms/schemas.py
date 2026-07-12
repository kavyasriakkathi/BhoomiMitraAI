from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Allowed enum values
VALID_SOIL_TYPES = {"black", "red", "alluvial", "laterite", "sandy", "clay", "loamy", "saline"}
VALID_IRRIGATION_TYPES = {"drip", "sprinkler", "canal", "borewell", "rainfed", "flood"}


class FarmBase(BaseModel):
    """Shared fields for Farm create and update schemas."""
    farm_name: str = Field(..., min_length=1, max_length=100, description="Name or label for the farm")
    land_size_acres: float = Field(..., gt=0, le=10000, description="Farm area in acres (must be > 0)")
    soil_type: Optional[str] = Field(None, max_length=50, description="Soil type, e.g., Black, Red, Alluvial")
    irrigation_type: Optional[str] = Field(None, max_length=50, description="Irrigation method, e.g., Drip, Canal, Rainfed")
    village: Optional[str] = Field(None, max_length=100, description="Village name")
    district: Optional[str] = Field(None, max_length=50, description="District name")
    state: Optional[str] = Field(None, max_length=50, description="State name")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="GPS latitude (-90 to 90)")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="GPS longitude (-180 to 180)")

    @field_validator("soil_type")
    @classmethod
    def validate_soil_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_SOIL_TYPES:
            raise ValueError(
                f"Invalid soil type '{v}'. Must be one of: {', '.join(sorted(VALID_SOIL_TYPES))}"
            )
        return v.capitalize() if v else v

    @field_validator("irrigation_type")
    @classmethod
    def validate_irrigation_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_IRRIGATION_TYPES:
            raise ValueError(
                f"Invalid irrigation type '{v}'. Must be one of: {', '.join(sorted(VALID_IRRIGATION_TYPES))}"
            )
        return v.capitalize() if v else v

    @field_validator("farm_name")
    @classmethod
    def validate_farm_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("farm_name cannot be blank.")
        return v.strip()


class FarmCreate(FarmBase):
    """Schema for creating a new farm."""
    farmer_id: UUID


class FarmUpdate(BaseModel):
    """Schema for updating a farm. All fields optional."""
    farm_name: Optional[str] = Field(None, min_length=1, max_length=100)
    land_size_acres: Optional[float] = Field(None, gt=0, le=10000)
    soil_type: Optional[str] = Field(None, max_length=50)
    irrigation_type: Optional[str] = Field(None, max_length=50)
    village: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=50)
    state: Optional[str] = Field(None, max_length=50)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)

    @field_validator("soil_type")
    @classmethod
    def validate_soil_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_SOIL_TYPES:
            raise ValueError(
                f"Invalid soil type '{v}'. Must be one of: {', '.join(sorted(VALID_SOIL_TYPES))}"
            )
        return v.capitalize() if v else v

    @field_validator("irrigation_type")
    @classmethod
    def validate_irrigation_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in VALID_IRRIGATION_TYPES:
            raise ValueError(
                f"Invalid irrigation type '{v}'. Must be one of: {', '.join(sorted(VALID_IRRIGATION_TYPES))}"
            )
        return v.capitalize() if v else v


class FarmResponse(BaseModel):
    """Schema for farm API responses."""
    id: UUID
    farmer_id: UUID
    farm_name: str
    land_size_acres: float
    soil_type: Optional[str] = None
    irrigation_type: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedFarmResponse(BaseModel):
    """Paginated list of farms."""
    total: int
    items: list[FarmResponse]
    page: int
    size: int
