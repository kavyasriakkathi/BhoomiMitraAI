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
from src.gateway.schemas import ParsedIncomingMessage
from src.core.logging import logger

from src.gateway.whatsapp_client import download_media_bytes, send_text_message, mark_message_as_read
from src.language.dependencies import get_language_service
from src.ai.service import process_text_message


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
    db: AsyncSession, 
    parsed: ParsedIncomingMessage, 
    sender_name: Optional[str] = None
) -> None:
    """
    Orchestrates the entire lifecycle of an incoming WhatsApp message.
    Designed to be run safely as a FastAPI BackgroundTask.
    
    Pipeline:
    1. Deduplication
    2. Audio Transcription (if message is voice)
    3. Farmer Upsert
    4. Database Storage
    5. AI Generation
    6. Outbound Dispatch
    """
    try:
        # Step 1: Deduplication
        if await is_duplicate_message(db, parsed.message_id):
            logger.warning(f"Duplicate message {parsed.message_id}. Skipping processing.")
            return

        # Step 2: Audio Transcription (Voice Support)
        if parsed.message_type == "audio" and parsed.media_id:
            logger.info(f"Processing audio message {parsed.media_id} from {parsed.phone_number}")
            
            media_result = await download_media_bytes(parsed.media_id)
            if not media_result:
                logger.error(f"Failed to download audio {parsed.media_id}. Aborting pipeline.")
                await send_text_message(
                    to_phone=parsed.phone_number, 
                    message_text="I'm sorry, I couldn't load your voice message due to a network issue. Could you please type your question?"
                )
                return
                
            audio_bytes, mime_type = media_result
            lang_service = get_language_service()
            
            try:
                transcription = await lang_service.transcribe_audio(audio_bytes, mime_type)
                # Mutate the parsed message so the rest of the pipeline treats it exactly like text
                parsed.text_content = transcription.transcription_text
                logger.info(f"Successfully transcribed audio: '{parsed.text_content}'")
            except Exception as e:
                logger.error(f"Failed to transcribe audio: {e}")
                await send_text_message(
                    to_phone=parsed.phone_number, 
                    message_text="I'm sorry, I couldn't understand that voice message clearly. Could you please type your question?"
                )
                return

        # Step 3 & 4: Ensure Farmer exists and Store Conversation
        farmer = await get_or_create_farmer(db, parsed.phone_number, sender_name)
        conversation = await store_incoming_message(db, farmer, parsed)

        # Step 5: AI Text Processing
        # (Only process if we have text, either from a native text message or a transcribed audio message)
        if parsed.text_content:
            ai_response = await process_text_message(db, farmer, conversation)
            
            # Step 6: Dispatch outbound reply to WhatsApp
            outbound_id = await send_text_message(
                to_phone=parsed.phone_number,
                message_text=ai_response,
            )
            
            # Update delivery status
            conversation.outbound_message_id = outbound_id
            conversation.delivery_status = "sent" if outbound_id else "failed"
            db.add(conversation)
            await db.commit()
            
            logger.info(
                f"Pipeline complete. Reply {'sent' if outbound_id else 'FAILED'} to "
                f"{parsed.phone_number} (outbound_id={outbound_id})"
            )

        # Step 7: Best-effort mark incoming message as read
        await mark_message_as_read(parsed.message_id)

    except Exception as e:
        logger.exception(f"Fatal error in background message pipeline for {parsed.message_id}: {e}")
