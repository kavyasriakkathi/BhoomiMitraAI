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
