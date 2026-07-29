from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class FarmerMemoryResponse(BaseModel):
    """API Response Schema for Farmer Memory Engine"""
    id: UUID
    farmer_id: UUID

    # Voice & Language Personalization
    preferred_language: str = "te"
    preferred_voice: str = "Google-te-IN-Standard-A"
    voice_speed: float = 1.0
    voice_gender: str = "FEMALE"

    # Land & Geographic Details
    farm_size: Optional[float] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    gps_coordinates: Dict[str, Any] = Field(default_factory=dict)

    # Soil & Water
    soil_type: Optional[str] = None
    water_source: Optional[str] = None
    irrigation_method: Optional[str] = None

    # Agronomic & History
    primary_crops: List[str] = Field(default_factory=list)
    secondary_crops: List[str] = Field(default_factory=list)
    crop_history: List[Dict[str, Any]] = Field(default_factory=list)
    disease_history: List[Dict[str, Any]] = Field(default_factory=list)
    pesticide_history: List[Dict[str, Any]] = Field(default_factory=list)
    fertilizer_history: List[Dict[str, Any]] = Field(default_factory=list)
    yield_history: List[Dict[str, Any]] = Field(default_factory=list)
    weather_region: Optional[str] = None

    # Commerce & Shop Preferences
    favorite_shops: List[str] = Field(default_factory=list)
    purchase_history: List[Dict[str, Any]] = Field(default_factory=list)
    preferred_brands: List[str] = Field(default_factory=list)
    budget_range: Optional[str] = None

    # Schemes & Expert Consultations
    government_schemes_used: List[str] = Field(default_factory=list)
    expert_consultation_history: List[Dict[str, Any]] = Field(default_factory=list)

    # AI & Summary
    conversation_summary: Optional[str] = None
    frequently_asked_questions: List[Dict[str, Any]] = Field(default_factory=list)
    ai_learned_preferences: Dict[str, Any] = Field(default_factory=dict)
    risk_factors: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)


class FarmerMemoryUpdate(BaseModel):
    """API Request Schema for Expert/Admin/System Memory Updates"""
    preferred_language: Optional[str] = None
    preferred_voice: Optional[str] = None
    voice_speed: Optional[float] = None
    voice_gender: Optional[str] = None

    farm_size: Optional[float] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    gps_coordinates: Optional[Dict[str, Any]] = None

    soil_type: Optional[str] = None
    water_source: Optional[str] = None
    irrigation_method: Optional[str] = None

    primary_crops: Optional[List[str]] = None
    secondary_crops: Optional[List[str]] = None
    crop_history: Optional[List[Dict[str, Any]]] = None
    disease_history: Optional[List[Dict[str, Any]]] = None
    pesticide_history: Optional[List[Dict[str, Any]]] = None
    fertilizer_history: Optional[List[Dict[str, Any]]] = None
    yield_history: Optional[List[Dict[str, Any]]] = None
    weather_region: Optional[str] = None

    favorite_shops: Optional[List[str]] = None
    purchase_history: Optional[List[Dict[str, Any]]] = None
    preferred_brands: Optional[List[str]] = None
    budget_range: Optional[str] = None

    government_schemes_used: Optional[List[str]] = None
    expert_consultation_history: Optional[List[Dict[str, Any]]] = None

    conversation_summary: Optional[str] = None
    frequently_asked_questions: Optional[List[Dict[str, Any]]] = None
    ai_learned_preferences: Optional[Dict[str, Any]] = None
    risk_factors: Optional[List[str]] = None
    confidence_scores: Optional[Dict[str, float]] = None


class FarmerMemoryRefreshRequest(BaseModel):
    """API Request Schema for POST /memory/refresh"""
    farmer_id: UUID


class FarmerMemorySummaryResponse(BaseModel):
    """API Response Schema for GET /memory/summary/{farmer_id}"""
    farmer_id: UUID
    summary: Optional[str]
    last_updated: datetime
    primary_crops: List[str]
    district: Optional[str]
    risk_factors: List[str]


class VoiceSettingsResponse(BaseModel):
    """Voice Personalization Settings for STT/TTS Configuration"""
    farmer_id: UUID
    preferred_language: str
    preferred_voice: str
    voice_speed: float
    voice_gender: str
