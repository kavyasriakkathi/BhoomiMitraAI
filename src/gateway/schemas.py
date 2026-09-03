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

from typing import Optional, List
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Inbound: Meta Webhook Payload (Nested Models)
# ──────────────────────────────────────────────

class WhatsAppProfile(BaseModel):
    name: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppContact(BaseModel):
    profile: Optional[WhatsAppProfile] = None
    wa_id: Optional[str] = None  # Farmer's phone number
    model_config = {"extra": "ignore"}


class WhatsAppTextPayload(BaseModel):
    body: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppAudioPayload(BaseModel):
    id: Optional[str] = None
    mime_type: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppImagePayload(BaseModel):
    id: Optional[str] = None
    mime_type: Optional[str] = None
    caption: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppMessage(BaseModel):
    """Represents a single message within the webhook payload."""
    from_: Optional[str] = Field(None, alias="from")  # Sender phone number
    id: Optional[str] = None  # Unique Meta message ID (used for idempotency)
    timestamp: Optional[str] = None
    type: Optional[str] = "text"  # 'text', 'audio', 'image', 'interactive', 'button'

    text: Optional[WhatsAppTextPayload] = None
    audio: Optional[WhatsAppAudioPayload] = None
    image: Optional[WhatsAppImagePayload] = None

    model_config = {"extra": "ignore", "populate_by_name": True}


class WhatsAppMetadata(BaseModel):
    display_phone_number: Optional[str] = None
    phone_number_id: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppValue(BaseModel):
    messaging_product: Optional[str] = "whatsapp"
    metadata: Optional[WhatsAppMetadata] = None
    contacts: Optional[List[WhatsAppContact]] = None
    messages: Optional[List[WhatsAppMessage]] = None
    model_config = {"extra": "ignore"}


class WhatsAppChange(BaseModel):
    value: Optional[WhatsAppValue] = None
    field: Optional[str] = None
    model_config = {"extra": "ignore"}


class WhatsAppEntry(BaseModel):
    id: Optional[str] = None
    changes: List[WhatsAppChange] = Field(default_factory=list)
    model_config = {"extra": "ignore"}


class WhatsAppWebhookPayload(BaseModel):
    """Top-level schema for the incoming Meta webhook POST body."""
    object: Optional[str] = None
    entry: List[WhatsAppEntry] = Field(default_factory=list)
    model_config = {"extra": "ignore"}


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
    model_config = {"extra": "ignore"}


def mask_phone_number(phone: str) -> str:
    """Mask phone number for safe production logging (e.g. 9198****3210)."""
    if not phone or len(phone) < 7:
        return "***"
    return phone[:4] + "****" + phone[-4:]


def sanitize_headers_for_logging(headers: dict) -> dict:
    """Mask sensitive tokens and signatures from header logs."""
    sensitive_keys = {"authorization", "x-hub-signature", "x-hub-signature-256", "cookie", "set-cookie"}
    return {
        k: ("***REDACTED***" if k.lower() in sensitive_keys else v)
        for k, v in headers.items()
    }


