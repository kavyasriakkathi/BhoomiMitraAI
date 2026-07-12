"""
BhoomiMitra AI — WhatsApp Webhook Schemas

Pydantic models for validating incoming Meta WhatsApp payloads
and structuring outbound messages.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Inbound: Meta Webhook Payload (Nested Models)
# ──────────────────────────────────────────────

class WhatsAppProfile(BaseModel):
    name: str


class WhatsAppContact(BaseModel):
    profile: WhatsAppProfile
    wa_id: str  # Farmer's phone number


class WhatsAppTextPayload(BaseModel):
    body: str


class WhatsAppAudioPayload(BaseModel):
    id: str
    mime_type: Optional[str] = None


class WhatsAppImagePayload(BaseModel):
    id: str
    mime_type: Optional[str] = None
    caption: Optional[str] = None


class WhatsAppMessage(BaseModel):
    """Represents a single message within the webhook payload."""
    from_: str = Field(..., alias="from")  # Sender phone number
    id: str  # Unique Meta message ID (used for idempotency)
    timestamp: str
    type: str  # 'text', 'audio', 'image', 'interactive', 'button'

    text: Optional[WhatsAppTextPayload] = None
    audio: Optional[WhatsAppAudioPayload] = None
    image: Optional[WhatsAppImagePayload] = None

    model_config = {"populate_by_name": True}


class WhatsAppMetadata(BaseModel):
    display_phone_number: str
    phone_number_id: str


class WhatsAppValue(BaseModel):
    messaging_product: str
    metadata: WhatsAppMetadata
    contacts: Optional[List[WhatsAppContact]] = None
    messages: Optional[List[WhatsAppMessage]] = None


class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str


class WhatsAppEntry(BaseModel):
    id: str
    changes: List[WhatsAppChange]


class WhatsAppWebhookPayload(BaseModel):
    """Top-level schema for the incoming Meta webhook POST body."""
    object: str
    entry: List[WhatsAppEntry]


# ──────────────────────────────────────────────
# Internal: Parsed Message (Clean, Flat Format)
# ──────────────────────────────────────────────

class ParsedIncomingMessage(BaseModel):
    """A clean, flat representation of an incoming farmer message."""
    phone_number: str
    message_id: str
    timestamp: str
    message_type: str  # 'text', 'audio', 'image'
    text_content: Optional[str] = None
    media_id: Optional[str] = None
    media_mime_type: Optional[str] = None
    sender_name: Optional[str] = None
