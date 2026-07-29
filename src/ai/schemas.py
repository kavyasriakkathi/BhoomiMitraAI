from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict

class AIGenerateRequest(BaseModel):
    farmer_id: UUID
    conversation_id: Optional[UUID] = None
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty.")
        return v.strip()

class AIGenerateResponse(BaseModel):
    response_text: str
    intent: Optional[str] = Field(None, max_length=50)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    provider_used: str = Field(..., max_length=50)

class AIHealthResponse(BaseModel):
    status: str = Field(..., max_length=50)
    active_provider: str = Field(..., max_length=50)

class MultimodalDiagnosisResponse(BaseModel):
    disease_name: Optional[str] = Field(None, description="The name of the detected disease or pest.")
    confidence_score: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0.")
    severity: Optional[str] = Field(None, description="Severity of the issue, e.g. low, medium, high.")
    symptoms: str = Field(..., description="Description of the visible symptoms.")
    treatment_recommendation: str = Field(..., description="Recommended actions or treatments.")
    friendly_whatsapp_reply: str = Field(..., description="A natural language reply to send to the farmer on WhatsApp.")
