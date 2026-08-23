"""
Pydantic schemas for authentication and user accounts.
"""

import re
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.auth.constants import UserRole

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+$")


class UserRegisterRequest(BaseModel):
    """Payload for creating a new dashboard user account."""
    email: str = Field(..., max_length=255, description="Unique email address for login")
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password (min 8 chars)")
    role: UserRole = Field(..., description="Role: admin, expert, or shop_owner")
    expert_id: Optional[UUID] = Field(None, description="Linked Expert ID if role is expert")
    shop_id: Optional[UUID] = Field(None, description="Linked Shop ID if role is shop_owner")
    admin_creation_key: Optional[str] = Field(
        None,
        description="Secret key required if registering an admin account without an active admin session",
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email address format")
        return v


class LoginRequest(BaseModel):
    """Payload for user login."""
    email: str = Field(..., max_length=255, description="Registered user email")
    password: str = Field(..., min_length=1, description="User password")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email address format")
        return v


class UserResponse(BaseModel):
    """Public user account response representation."""
    id: UUID
    email: str
    role: str
    is_active: bool
    expert_id: Optional[UUID] = None
    shop_id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """JWT login token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    """Simple status/message response."""
    success: bool
    message: str
