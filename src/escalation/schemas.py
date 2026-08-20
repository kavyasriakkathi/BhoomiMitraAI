from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class ExpertBase(BaseModel):
    name: str = Field(..., max_length=100, description="Full name of agricultural expert / AEO")
    phone_number: str = Field(..., max_length=20, description="Contact phone number")
    specialty: Optional[str] = Field(None, max_length=100, description="Specialty, e.g. Pest Control, Soil Health")
    is_active: bool = Field(True, description="Whether expert is available for assignment")


class ExpertCreate(ExpertBase):
    pass


class ExpertUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    specialty: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class ExpertResponse(ExpertBase):
    id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedExpertResponse(BaseModel):
    items: List[ExpertResponse]
    total: int


class EscalationTicketResponse(BaseModel):
    ticket_id: str
    farmer_id: Optional[UUID] = None
    status: str = "Assigned"  # Pending, Assigned, Resolved
    topic: str
    expert_id: Optional[UUID] = None
    expert_name: Optional[str] = None
    expert_specialty: Optional[str] = None
    expert_phone: Optional[str] = None
    region: Optional[str] = None
    created_at: str
    callback_window: str = "Within 30–60 minutes"
    helpline: str = "1800-180-1551"


class FarmerEscalationHistoryResponse(BaseModel):
    farmer_id: UUID
    total_tickets: int
    tickets: List[Dict[str, Any]]
