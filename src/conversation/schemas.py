import re
from typing import Optional, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Allowed enum values
VALID_MESSAGE_TYPES = {"text", "audio", "image"}
VALID_DELIVERY_STATUSES = {"pending", "sent", "delivered", "read", "failed"}


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation record."""
    farmer_id: UUID
    message_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique Meta message ID for idempotency"
    )
    user_message: Optional[str] = Field(None, description="The farmer's incoming message text")
    user_message_type: str = Field(
        default="text", max_length=20,
        description="Type of incoming message: text, audio, or image"
    )
    ai_response: Optional[str] = Field(None, description="AI-generated response")
    intent: Optional[str] = Field(None, max_length=50, description="Detected intent category")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="AI confidence score between 0 and 1"
    )

    @field_validator("user_message_type")
    @classmethod
    def validate_message_type(cls, v: str) -> str:
        if v not in VALID_MESSAGE_TYPES:
            raise ValueError(f"Invalid message type '{v}'. Must be one of: {', '.join(sorted(VALID_MESSAGE_TYPES))}")
        return v

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message_id cannot be blank.")
        return v.strip()


class ConversationUpdate(BaseModel):
    """Schema for updating an existing conversation record."""
    ai_response: Optional[str] = None
    intent: Optional[str] = Field(None, max_length=50)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    outbound_message_id: Optional[str] = Field(None, max_length=100)
    delivery_status: Optional[str] = Field(None, max_length=20)

    @field_validator("delivery_status")
    @classmethod
    def validate_delivery_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_DELIVERY_STATUSES:
            raise ValueError(
                f"Invalid delivery status '{v}'. Must be one of: {', '.join(sorted(VALID_DELIVERY_STATUSES))}"
            )
        return v


class ConversationResponse(BaseModel):
    """Schema for conversation API responses."""
    id: UUID
    farmer_id: UUID
    message_id: str
    user_message: Optional[str] = None
    user_message_type: str
    ai_response: Optional[str] = None
    intent: Optional[str] = None
    confidence_score: Optional[float] = None
    outbound_message_id: Optional[str] = None
    delivery_status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedConversationResponse(BaseModel):
    """Paginated list of conversations."""
    total: int
    items: list[ConversationResponse]
    page: int
    size: int
