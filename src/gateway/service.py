"""
BhoomiMitra AI — WhatsApp Message Service

Business logic layer for processing incoming WhatsApp messages.
Handles farmer upsert, duplicate detection, conversation storage,
and orchestrating the full pipeline (STT -> AI -> Outbound).
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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
    Handles concurrent inserts gracefully via IntegrityError rollback.
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
    try:
        await db.commit()
        await db.refresh(farmer)

        profile = FarmerProfile(
            farmer_id=farmer.id,
            full_name=sender_name,
        )
        db.add(profile)
        await db.commit()

        logger.info(f"New farmer registered: {farmer.id} ({phone_number})")
        return farmer
    except IntegrityError:
        await db.rollback()
        # Concurrent insert occurred, re-query the newly created farmer
        res = await db.execute(
            select(Farmer).where(Farmer.phone_number == phone_number)
        )
        existing_farmer = res.scalar_one_or_none()
        if existing_farmer:
            return existing_farmer
        raise


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
) -> Optional[Conversation]:
    """
    Persist an incoming farmer message to the conversations table.
    AI response fields are left NULL — they will be populated later
    when the AI processing pipeline runs.

    If an IntegrityError occurs due to duplicate message_id unique constraint,
    safely rolls back the transaction and returns None.
    """
    farmer_id = getattr(farmer, "id", None)
    conversation = Conversation(
        farmer_id=farmer_id,
        message_id=message.message_id,
        user_message=message.text_content,
        user_message_type=message.message_type,
    )
    db.add(conversation)
    try:
        await db.commit()
        await db.refresh(conversation)

        logger.info(
            f"Stored message {message.message_id} from farmer {farmer_id} "
            f"(type={message.message_type})"
        )
        return conversation
    except IntegrityError as ie:
        await db.rollback()
        logger.warning(
            f"IntegrityError: Duplicate message_id '{message.message_id}' "
            f"already stored in database for farmer {farmer_id}. Rollback executed. Detail: {ie}"
        )
        return None


async def process_message_pipeline(
    parsed: ParsedIncomingMessage,
    sender_name: Optional[str] = None
) -> None:
    """
    Orchestrates the entire lifecycle of an incoming WhatsApp message with isolated,
    stage-by-stage error handling so no failures pass silently.
    """

    logger.info("=" * 80)
    logger.info("BACKGROUND PIPELINE STARTED")
    logger.info(f"  Message ID  : {parsed.message_id}")
    logger.info(f"  Sender Phone: {parsed.phone_number}")
    logger.info(f"  Message Type: {parsed.message_type}")
    if parsed.text_content:
        logger.info(f"  Message Text: {parsed.text_content[:150]}")
    logger.info("=" * 80)

    try:
        async with AsyncSessionLocal() as db:

            # ── STAGE 1: Duplicate Check ──────────────────────────────
            logger.info("STAGE 1: Checking for duplicate message in DB")
            try:
                if await is_duplicate_message(db, parsed.message_id):
                    logger.warning(f"STAGE 1: Duplicate message ID {parsed.message_id} detected. Skipping pipeline.")
                    return
                logger.info("STAGE 1: Message is unique.")
            except Exception as dup_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 1 - Duplicate Check] Message ID: {parsed.message_id}, Error: {dup_err}")

            # ── STAGE 2: Audio STT (if needed) ────────────────────────
            if parsed.message_type == "audio" and parsed.media_id:
                logger.info("STAGE 2: Audio download and STT transcription started")
                try:
                    media_result = await download_media_bytes(parsed.media_id)
                    if not media_result:
                        logger.error(f"[PIPELINE STAGE FAILED: Stage 2 - Audio Download] Failed to download media ID: {parsed.media_id}")
                        return
                    audio_bytes, mime_type = media_result
                    lang_service = get_language_service()
                    transcription = await lang_service.transcribe_audio(audio_bytes, mime_type)
                    parsed.text_content = transcription.transcription_text
                    logger.info(f"STAGE 2: Audio transcribed successfully: '{parsed.text_content[:100]}...'")
                except Exception as stt_err:
                    logger.exception(f"[PIPELINE STAGE FAILED: Stage 2 - Audio STT] Media ID: {parsed.media_id}, Error: {stt_err}")
                    return

            # ── STAGE 3: Farmer Resolution ────────────────────────────
            logger.info("STAGE 3: Resolving farmer profile in DB")
            try:
                farmer = await get_or_create_farmer(db, parsed.phone_number, sender_name)
                logger.info(f"STAGE 3: Farmer resolved successfully. Farmer ID = {farmer.id}")
            except Exception as db_farmer_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 3 - Farmer Resolution] Phone: {parsed.phone_number}, Error: {db_farmer_err}")
                return

            # ── STAGE 4: Conversation Storage ─────────────────────────
            logger.info("STAGE 4: Storing incoming message in DB")
            conversation = None
            try:
                conversation = await store_incoming_message(db, farmer, parsed)
                if conversation is None:
                    logger.warning(
                        f"STAGE 4: Duplicate message ID {parsed.message_id} detected during insert. "
                        f"Skipping pipeline early."
                    )
                    return
                logger.info(f"STAGE 4: Conversation stored successfully. Conversation ID = {conversation.id}")
            except Exception as db_conv_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 4 - Conversation Storage] Farmer ID: {farmer.id}, Error: {db_conv_err}")
                return

            # ── STAGE 5: AI Processing (Gemini) ───────────────────────
            logger.info("STAGE 5: Generating AI response")
            ai_response = None
            try:
                conv_ref = conversation or Conversation(farmer_id=farmer.id, user_message=parsed.text_content, user_message_type=parsed.message_type)
                
                if parsed.message_type == "image" and parsed.media_id:
                    media_result = await download_media_bytes(parsed.media_id)
                    if media_result:
                        image_bytes, mime_type = media_result
                        ai_response = await process_image_message(
                            db, farmer, conv_ref, image_bytes, mime_type
                        )
                    else:
                        logger.error(f"[PIPELINE STAGE FAILED: Stage 5 - Image Download] Media ID {parsed.media_id} failed download")
                elif parsed.text_content:
                    ai_response = await process_text_message(
                        db, farmer, conv_ref
                    )

                if ai_response:
                    logger.info(f"STAGE 5: AI response generated ({len(ai_response)} chars): {ai_response[:120]}...")
                else:
                    logger.warning(f"STAGE 5: AI generated no text response for farmer {farmer.id}")
            except Exception as ai_err:
                logger.exception(
                    f"[PIPELINE STAGE FAILED: Stage 5 - AI Processing] "
                    f"Farmer ID: {farmer.id}, Message Type: {parsed.message_type}, Text: '{parsed.text_content}', Error: {ai_err}"
                )

            # ── STAGE 6: Outbound WhatsApp Send ───────────────────────
            outbound_id = None
            if ai_response:
                logger.info("STAGE 6: Sending outbound WhatsApp message to Meta Cloud API")
                try:
                    outbound_id = await send_text_message(
                        to_phone=parsed.phone_number,
                        message_text=ai_response,
                    )
                    if outbound_id:
                        logger.info(f"STAGE 6: WhatsApp send SUCCESS. Outbound Meta ID = {outbound_id}")
                    else:
                        logger.error(f"[PIPELINE STAGE FAILED: Stage 6 - Outbound Send] Meta API returned None for {parsed.phone_number}")
                except Exception as send_err:
                    logger.exception(
                        f"[PIPELINE STAGE FAILED: Stage 6 - Outbound Send] "
                        f"Phone: {parsed.phone_number}, Error: {send_err}"
                    )
            else:
                logger.warning("STAGE 6 SKIPPED: No AI response was available to send.")

            # ── STAGE 7: Database Delivery Status Update ──────────────
            if conversation:
                logger.info("STAGE 7: Updating delivery status in database")
                try:
                    conversation.outbound_message_id = outbound_id
                    conversation.delivery_status = "sent" if outbound_id else "failed"
                    db.add(conversation)
                    await db.commit()
                    logger.info(f"STAGE 7: Delivery status set to '{conversation.delivery_status}'")
                except Exception as db_status_err:
                    logger.exception(f"[PIPELINE STAGE FAILED: Stage 7 - DB Status Update] Conversation ID: {conversation.id}, Error: {db_status_err}")

            # ── STAGE 8: Mark Message as Read ──────────────────────────
            logger.info("STAGE 8: Marking message as read with Meta API")
            try:
                await mark_message_as_read(parsed.message_id)
                logger.info("STAGE 8: Read receipt sent.")
            except Exception as read_err:
                logger.warning(f"STAGE 8: Read receipt warning for {parsed.message_id}: {read_err}")

            logger.info("=" * 80)
            logger.info("BACKGROUND PIPELINE COMPLETED")
            logger.info("=" * 80)

    except Exception as pipeline_err:
        logger.exception(f"[CRITICAL PIPELINE FAILURE] Unhandled error in background pipeline for message {parsed.message_id}: {pipeline_err}")