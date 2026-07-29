import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base

class FarmerMemory(Base):
    """
    Extensible Production-Grade Farmer Long-Term Memory Profile Model.
    
    Stores preferences, farm parameters, voice options, agronomic history,
    commerce preferences, government schemes used, AI learned preferences,
    confidence scores, and conversation summaries.
    """
    __tablename__ = "farmer_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), unique=True, index=True, nullable=False)

    # Voice & Language Personalization
    preferred_language = Column(String(10), default="te")
    preferred_voice = Column(String(100), default="Google-te-IN-Standard-A")
    voice_speed = Column(Float, default=1.0)
    voice_gender = Column(String(20), default="FEMALE")

    # Land & Geographic Details
    farm_size = Column(Float, nullable=True)
    village = Column(String(100), nullable=True)
    district = Column(String(50), nullable=True, index=True)
    state = Column(String(50), nullable=True, index=True)
    gps_coordinates = Column(JSON, default=dict) # e.g. {"latitude": 17.38, "longitude": 78.48}

    # Soil & Water
    soil_type = Column(String(50), nullable=True)
    water_source = Column(String(100), nullable=True)
    irrigation_method = Column(String(100), nullable=True)

    # Agronomic & History
    primary_crops = Column(JSON, default=list) # e.g. ["Cotton", "Paddy"]
    secondary_crops = Column(JSON, default=list) # e.g. ["Chilli"]
    crop_history = Column(JSON, default=list)
    disease_history = Column(JSON, default=list)
    pesticide_history = Column(JSON, default=list)
    fertilizer_history = Column(JSON, default=list)
    yield_history = Column(JSON, default=list)
    weather_region = Column(String(100), nullable=True)

    # Commerce & Shop Preferences
    favorite_shops = Column(JSON, default=list)
    purchase_history = Column(JSON, default=list)
    preferred_brands = Column(JSON, default=list)
    budget_range = Column(String(50), nullable=True)

    # Schemes & Expert Consultations
    government_schemes_used = Column(JSON, default=list)
    expert_consultation_history = Column(JSON, default=list)

    # AI & Summary
    conversation_summary = Column(Text, nullable=True)
    frequently_asked_questions = Column(JSON, default=list)
    ai_learned_preferences = Column(JSON, default=dict)
    risk_factors = Column(JSON, default=list)
    confidence_scores = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    farmer = relationship("Farmer", back_populates="memory")
