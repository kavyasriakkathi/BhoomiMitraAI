import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base

class Farmer(Base):
    """Core Farmer Identity Model"""
    __tablename__ = "farmers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String(20), unique=True, index=True, nullable=False)
    preferred_language = Column(String(10), default="te") # 'te', 'hi', 'en', etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    profile = relationship("FarmerProfile", back_populates="farmer", uselist=False, cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="farmer", cascade="all, delete-orphan")
    farms = relationship("Farm", back_populates="farmer", cascade="all, delete-orphan")


class FarmerProfile(Base):
    """Detailed Profile / State Machine for the Farmer"""
    __tablename__ = "farmer_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), unique=True, nullable=False)
    
    full_name = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True, index=True)
    district = Column(String(50), nullable=True, index=True)
    current_crop = Column(String(100), nullable=True)
    land_size_acres = Column(Float, nullable=True)
    
    # Relationships
    farmer = relationship("Farmer", back_populates="profile")


class Conversation(Base):
    """Stores interaction history for AI Context Memory"""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)
    message_id = Column(String(100), unique=True, nullable=False) # Meta message ID for idempotency
    
    user_message = Column(Text, nullable=True)
    user_message_type = Column(String(20), default="text") # 'text', 'audio', 'image'
    
    ai_response = Column(Text, nullable=True)
    intent = Column(String(50), nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Outbound tracking
    outbound_message_id = Column(String(100), nullable=True)  # Meta message ID for the sent reply
    delivery_status = Column(String(20), default="pending")   # 'pending', 'sent', 'delivered', 'read', 'failed'
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    farmer = relationship("Farmer", back_populates="conversations")


class Farm(Base):
    """Farm details — a farmer can own multiple farms"""
    __tablename__ = "farms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)

    farm_name = Column(String(100), nullable=False)
    land_size_acres = Column(Float, nullable=False)
    soil_type = Column(String(50), nullable=True)        # e.g., 'Black', 'Red', 'Alluvial'
    irrigation_type = Column(String(50), nullable=True)  # e.g., 'Drip', 'Canal', 'Rainfed'
    village = Column(String(100), nullable=True)
    district = Column(String(50), nullable=True, index=True)
    state = Column(String(50), nullable=True, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("Farmer", back_populates="farms")
    crops = relationship("Crop", back_populates="farm", cascade="all, delete-orphan")


class Expert(Base):
    """Agricultural Expert Model for Escalation"""
    __tablename__ = "experts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    phone_number = Column(String(20), unique=True, index=True, nullable=False)
    specialty = Column(String(100), nullable=True) # e.g., "Pest Control", "Soil Health"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Crop(Base):
    """Crop details — a farm can have multiple crops"""
    __tablename__ = "crops"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id = Column(UUID(as_uuid=True), ForeignKey("farms.id"), index=True, nullable=False)

    crop_name = Column(String(100), nullable=False)
    variety = Column(String(100), nullable=True)
    sowing_date = Column(DateTime, nullable=True)
    season = Column(String(50), nullable=True)
    status = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farm = relationship("Farm", back_populates="crops")
