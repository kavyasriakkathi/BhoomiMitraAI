from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


def mask_token(token: str) -> str:
    """Mask FCM registration token for safe logging/display."""
    if not token:
        return "***"
    token_str = str(token).strip()
    if len(token_str) <= 12:
        return "***"
    return f"{token_str[:6]}...{token_str[-4:]}"


class PushTokenRegisterRequest(BaseModel):
    """Payload to register or refresh an FCM device registration token."""
    token: str = Field(..., min_length=10, max_length=512, description="FCM device registration token")
    platform: str = Field(default="android", description="Platform identifier ('android', 'ios')")
    device_model: Optional[str] = Field(default=None, max_length=100, description="Device model/manufacturer")
    phone_number: Optional[str] = Field(default=None, description="Farmer phone number if unauthenticated")


class PushTokenDeactivateRequest(BaseModel):
    """Payload to unregister or deactivate an FCM device registration token."""
    token: str = Field(..., min_length=10, max_length=512, description="FCM device registration token")


class PushTokenResponse(BaseModel):
    """Response returned upon registering/updating a device token."""
    id: UUID
    farmer_id: UUID
    token_masked: str
    platform: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
