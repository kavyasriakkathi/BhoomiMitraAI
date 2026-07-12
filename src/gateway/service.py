"""
BhoomiMitra AI — WhatsApp Message Service

Business logic layer for processing incoming WhatsApp messages.
Handles farmer upsert, duplicate detection, and conversation storage.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Farmer, FarmerProfile, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.core.logging import logger


async def get_or_create_farmer(
    db: AsyncSession, phone_number: str, sender_name: Optional[str] = None
) -> Farmer:
    """
    Find an existing farmer by phone number, or create a new one.
    This is the implicit registration step — the first WhatsApp message
    from a farmer automatically creates their account.
    """
    result = await db.execute(
        select(Farmer).where(Farmer.phone_number == phone_number)
    )
    farmer = result.scalar_one_or_none()

    if farmer:
        logger.info(f"Returning farmer found: {farmer.id}")
        return farmer

    # New farmer — create Farmer + empty Profile
    farmer = Farmer(phone_number=phone_number)
    db.add(farmer)
    await db.flush()  # Populate farmer.id before creating profile

    profile = FarmerProfile(
        farmer_id=farmer.id,
        full_name=sender_name,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(farmer)

    logger.info(f"New farmer registered: {farmer.id} ({phone_number})")
    return farmer


async def is_duplicate_message(db: AsyncSession, message_id: str) -> bool:
    """
    Check if a WhatsApp message ID has already been processed.
    This prevents duplicate processing when Meta retries a webhook delivery.
    """
    result = await db.execute(
        select(Conversation.id).where(Conversation.message_id == message_id)
    )
    return result.scalar_one_or_none() is not None


async def store_incoming_message(
    db: AsyncSession,
    farmer: Farmer,
    message: ParsedIncomingMessage,
) -> Conversation:
    """
    Persist an incoming farmer message to the conversations table.
    AI response fields are left NULL — they will be populated later
    when the AI processing pipeline runs.
    """
    conversation = Conversation(
        farmer_id=farmer.id,
        message_id=message.message_id,
        user_message=message.text_content,
        user_message_type=message.message_type,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    logger.info(
        f"Stored message {message.message_id} from farmer {farmer.id} "
        f"(type={message.message_type})"
    )
    return conversation
