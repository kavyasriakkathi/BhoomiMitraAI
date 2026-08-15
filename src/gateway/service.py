"""
BhoomiMitra AI — WhatsApp Message Service

Business logic layer for processing incoming WhatsApp messages.
Handles farmer upsert, duplicate detection, conversation storage,
and orchestrating the full pipeline (STT -> AI -> Outbound).
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Farmer, FarmerProfile, Conversation
from src.core.database import AsyncSessionLocal
from src.gateway.schemas import ParsedIncomingMessage
from src.core.logging import logger

from src.gateway.whatsapp_client import download_media_bytes, send_text_message, mark_message_as_read
from src.language.dependencies import get_language_service
from src.ai.service import process_text_message, process_image_message


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


async def process_message_pipeline(
    parsed: ParsedIncomingMessage,
    sender_name: Optional[str] = None
) -> None:
    """
    Orchestrates the entire lifecycle of an incoming WhatsApp message.
    """

    logger.info("=" * 80)
    logger.info("BACKGROUND PIPELINE STARTED")
    logger.info(f"MESSAGE ID: {parsed.message_id}")
    logger.info(f"SENDER PHONE: {parsed.phone_number}")
    logger.info(f"MESSAGE TYPE: {parsed.message_type}")
    if parsed.text_content:
        logger.info(f"MESSAGE TEXT RECEIVED: {parsed.text_content[:100]}")
    logger.info("=" * 80)

    try:
        async with AsyncSessionLocal() as db:

            logger.info("STEP 1: Checking duplicate message")

            if await is_duplicate_message(db, parsed.message_id):
                logger.warning(f"Duplicate message {parsed.message_id}. Skipping.")
                return

            logger.info("STEP 2: Audio transcription (if needed)")

            if parsed.message_type == "audio" and parsed.media_id:
                media_result = await download_media_bytes(parsed.media_id)

                if not media_result:
                    logger.error("Audio download failed.")
                    return

                audio_bytes, mime_type = media_result

                lang_service = get_language_service()

                transcription = await lang_service.transcribe_audio(
                    audio_bytes,
                    mime_type
                )

                parsed.text_content = transcription.transcription_text

            logger.info("FARMER RESOLUTION STARTED")

            farmer = await get_or_create_farmer(
                db,
                parsed.phone_number,
                sender_name
            )

            logger.info(f"FARMER RESOLUTION SUCCESS: Farmer ID = {farmer.id}")

            logger.info("CONVERSATION STORAGE STARTED")

            conversation = await store_incoming_message(
                db,
                farmer,
                parsed
            )

            logger.info(f"CONVERSATION STORED: Conversation ID = {conversation.id}")

            logger.info("AI PROCESSING STARTED")

            ai_response = None

            if parsed.message_type == "image" and parsed.media_id:

                media_result = await download_media_bytes(parsed.media_id)

                if media_result:
                    image_bytes, mime_type = media_result

                    ai_response = await process_image_message(
                        db,
                        farmer,
                        conversation,
                        image_bytes,
                        mime_type,
                    )

            elif parsed.text_content:

                ai_response = await process_text_message(
                    db,
                    farmer,
                    conversation,
                )

            if ai_response:
                logger.info(f"AI RESPONSE GENERATED ({len(ai_response)} chars): {ai_response[:120]}...")
            else:
                logger.warning("AI RESPONSE GENERATED: NONE (No text response produced)")

            logger.info("WHATSAPP SEND STARTED")

            if ai_response:

                outbound_id = await send_text_message(
                    to_phone=parsed.phone_number,
                    message_text=ai_response,
                )

                if outbound_id:
                    logger.info(f"WHATSAPP SEND SUCCESS: OUTBOUND MESSAGE ID = {outbound_id}")
                else:
                    logger.error("WHATSAPP SEND FAILED: Outbound message ID returned None")

                conversation.outbound_message_id = outbound_id
                conversation.delivery_status = (
                    "sent" if outbound_id else "failed"
                )

                db.add(conversation)
                await db.commit()

            logger.info("STEP 7: Mark as Read")

            await mark_message_as_read(parsed.message_id)

            logger.info("PIPELINE FINISHED SUCCESSFULLY")

    except Exception as e:
        logger.exception(f"BACKGROUND PIPELINE FAILED for message {parsed.message_id}: {e}")