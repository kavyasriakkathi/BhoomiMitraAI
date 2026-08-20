import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base
from src.memory.models import FarmerMemory
from src.rag.models import KnowledgeDocument, KnowledgeChunk, EmbeddingMetadata


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
    memory = relationship("FarmerMemory", back_populates="farmer", uselist=False, cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="farmer", cascade="all, delete-orphan")
    farms = relationship("Farm", back_populates="farmer", cascade="all, delete-orphan")
    crop_health_diagnoses = relationship("CropHealth", back_populates="farmer", cascade="all, delete-orphan")
    advisories = relationship("Advisory", back_populates="farmer", cascade="all, delete-orphan")
    order_requests = relationship("OrderRequest", back_populates="farmer", cascade="all, delete-orphan")
    scheme_applications = relationship("SchemeApplication", back_populates="farmer", cascade="all, delete-orphan")


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
    crop_health_diagnoses = relationship("CropHealth", back_populates="crop", cascade="all, delete-orphan")

class CropHealth(Base):
    """Crop health diagnosis records"""
    __tablename__ = "crop_health"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crop_id = Column(UUID(as_uuid=True), ForeignKey("crops.id"), index=True, nullable=False)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)

    image_url = Column(String(500), nullable=True)
    symptoms = Column(Text, nullable=False)
    disease_name = Column(String(100), nullable=True)
    diagnosis_result = Column(Text, nullable=False)
    treatment_recommendation = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    crop = relationship("Crop", back_populates="crop_health_diagnoses")
    farmer = relationship("Farmer", back_populates="crop_health_diagnoses")

class Advisory(Base):
    """Agricultural Advisory for the farmer"""
    __tablename__ = "advisories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)

    advisory_type = Column(String(50), nullable=True) # e.g. "Weather", "Pest", "Market"
    message = Column(Text, nullable=False)
    source = Column(String(100), nullable=True) # e.g. "AI", "Expert"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("Farmer", back_populates="advisories")


class Shop(Base):
    """Agri Shop Registry Model"""
    __tablename__ = "shops"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shop_name = Column(String(150), nullable=False, index=True)
    owner_name = Column(String(100), nullable=False)
    phone_number = Column(String(20), nullable=False, index=True)
    email = Column(String(100), nullable=True)
    address = Column(Text, nullable=False)
    village = Column(String(100), nullable=True)
    mandal = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True, index=True)
    state = Column(String(100), nullable=True, index=True)
    pin_code = Column(String(20), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    opening_time = Column(String(20), default="08:00")
    closing_time = Column(String(20), default="20:00")
    delivery_available = Column(Boolean, default=False)
    home_delivery_radius_km = Column(Float, nullable=True)
    google_maps_link = Column(String(500), nullable=True)
    gst_number = Column(String(50), nullable=True)
    license_number = Column(String(50), nullable=True)
    status = Column(String(20), default="active", index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inventory_items = relationship("Inventory", back_populates="shop", cascade="all, delete-orphan")
    order_requests = relationship("OrderRequest", back_populates="shop", cascade="all, delete-orphan")


class Inventory(Base):
    """Agri Shop Inventory Product Model"""
    __tablename__ = "inventory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shop_id = Column(UUID(as_uuid=True), ForeignKey("shops.id"), index=True, nullable=False)

    product_name = Column(String(150), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), nullable=False, index=True)
    product_description = Column(Text, nullable=True)
    unit = Column(String(50), nullable=False, default="Unit")
    price = Column(Float, nullable=False)
    discount_price = Column(Float, nullable=True)
    quantity_in_stock = Column(Integer, default=0, nullable=False)
    minimum_stock_level = Column(Integer, default=5, nullable=False)
    available = Column(Boolean, default=True, index=True)
    expiry_date = Column(DateTime, nullable=True)

    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    shop = relationship("Shop", back_populates="inventory_items")
    order_requests = relationship("OrderRequest", back_populates="inventory_item", cascade="all, delete-orphan")


class OrderRequest(BaseModel if False else Base):
    """Farmer Cart & Shop Purchase Order Request Model"""
    __tablename__ = "order_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)
    shop_id = Column(UUID(as_uuid=True), ForeignKey("shops.id"), index=True, nullable=False)
    inventory_id = Column(UUID(as_uuid=True), ForeignKey("inventory.id"), index=True, nullable=False)

    product_name = Column(String(150), nullable=False)
    brand = Column(String(100), nullable=True)
    unit = Column(String(50), nullable=False, default="Unit")
    unit_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    total_price = Column(Float, nullable=False)

    status = Column(String(20), default="Pending", index=True) # Pending, Accepted, Ready, Completed, Cancelled
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("Farmer", back_populates="order_requests")
    shop = relationship("Shop", back_populates="order_requests")
    inventory_item = relationship("Inventory", back_populates="order_requests")


class GovernmentScheme(Base):
    """National and State Level Government Agriculture Schemes & Subsidies"""
    __tablename__ = "government_schemes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scheme_name = Column(String(200), nullable=False, index=True)
    scheme_code = Column(String(50), nullable=False, unique=True, index=True) # e.g. PM_KISAN, PMFBY, KCC, SOLAR_PUMP
    category = Column(String(100), nullable=False, index=True)               # Financial Assistance, Crop Insurance, Subsidies, etc.
    state = Column(String(100), default="All India", index=True)              # 'All India' or specific state like 'Telangana'
    district = Column(String(100), nullable=True, index=True)
    crop_type = Column(String(100), default="All Crops", index=True)
    min_land_acres = Column(Float, default=0.0)
    max_land_acres = Column(Float, nullable=True)                            # None means no upper limit
    description = Column(Text, nullable=False)
    benefits_summary = Column(Text, nullable=False)
    eligibility_criteria = Column(Text, nullable=False)
    required_documents = Column(Text, nullable=False)
    application_deadline = Column(DateTime, nullable=True)
    official_portal_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    applications = relationship("SchemeApplication", back_populates="scheme", cascade="all, delete-orphan")


class SchemeApplication(Base):
    """Farmer Applications & Tracking for Government Schemes"""
    __tablename__ = "scheme_applications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farmer_id = Column(UUID(as_uuid=True), ForeignKey("farmers.id"), index=True, nullable=False)
    scheme_id = Column(UUID(as_uuid=True), ForeignKey("government_schemes.id"), index=True, nullable=False)

    status = Column(String(50), default="Eligible", index=True) # Eligible, Applied, Under Review, Approved, Disbursed
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer = relationship("Farmer", back_populates="scheme_applications")
    scheme = relationship("GovernmentScheme", back_populates="applications")


class MarketPrice(Base):
    """Daily mandi/market commodity price record from Agmarknet or manual seeding."""
    __tablename__ = "market_prices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Commodity info
    commodity = Column(String(100), nullable=False, index=True)          # e.g. "Tomato"
    commodity_telugu = Column(String(100), nullable=True)                # e.g. "టమాటా"

    # Market/location info
    market_name = Column(String(150), nullable=False, index=True)        # e.g. "Warangal Mandi"
    district = Column(String(100), nullable=False, index=True)           # e.g. "Warangal"
    state = Column(String(100), nullable=False, index=True)              # e.g. "Telangana"

    # Price data (all in Rs/quintal by default)
    min_price = Column(Float, nullable=False)
    max_price = Column(Float, nullable=False)
    modal_price = Column(Float, nullable=False)                          # Most common traded price
    unit = Column(String(20), nullable=False, default="Quintal")

    # Metadata
    price_date = Column(DateTime, nullable=False, index=True)            # The date the price is valid for
    source = Column(String(50), nullable=False, default="agmarknet_api") # "agmarknet_api" or "manual_seed"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
