"""
BhoomiMitra AI — WhatsApp Message Service

Business logic layer for processing incoming WhatsApp messages.
Handles farmer upsert, duplicate detection, conversation storage,
and orchestrating the full pipeline (STT -> AI -> Outbound).
"""

import time
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.core.models import Farmer, FarmerProfile, Conversation
from src.core.database import AsyncSessionLocal
from src.gateway.schemas import ParsedIncomingMessage, mask_phone_number
from src.core.logging import logger

from src.gateway.whatsapp_client import download_media_bytes, send_text_message, mark_message_as_read
from src.language.dependencies import get_language_service
from src.ai.service import process_text_message, process_image_message, _finalize_whatsapp_response
from src.ai.prompts import (
    get_fallback_response,
    get_voice_fallback_response,
    get_image_fallback_response,
    get_unsupported_media_fallback_response,
)


# In-memory in-flight lock registry to prevent concurrent execution races across background tasks
_IN_FLIGHT_MESSAGE_IDS: set[str] = set()


def acquire_in_flight_lock(message_id: str) -> bool:
    """Atomically acquires in-flight lock for a message ID. Returns False if already in-flight."""
    if not message_id:
        return True
    if message_id in _IN_FLIGHT_MESSAGE_IDS:
        return False
    _IN_FLIGHT_MESSAGE_IDS.add(message_id)
    return True


def release_in_flight_lock(message_id: str) -> None:
    """Releases in-flight lock for a message ID."""
    if message_id:
        _IN_FLIGHT_MESSAGE_IDS.discard(message_id)


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
            preferred_language="te",  # Default to Telugu
        )
        db.add(profile)
        await db.commit()
        await db.refresh(farmer)

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

    pipeline_start = time.time()
    t_db = 0.0
    t_stt = 0.0
    t_ai = 0.0
    t_outbound = 0.0

    logger.info("=" * 80)
    logger.info("BACKGROUND PIPELINE STARTED")
    logger.info(f"  Message ID  : {parsed.message_id}")
    logger.info(f"  Sender Phone: {mask_phone_number(parsed.phone_number)}")
    logger.info(f"  Message Type: {parsed.message_type}")
    if parsed.text_content:
        logger.info(f"  Message Text: {parsed.text_content[:150]}")
    logger.info("=" * 80)

    # Concurrency Lock Check: Prevent in-flight execution races for the same message ID
    if not acquire_in_flight_lock(parsed.message_id):
        logger.warning(
            f"STAGE 0: Message ID '{parsed.message_id}' is already actively IN-FLIGHT in another background task. "
            "Aborting duplicate pipeline run immediately."
        )
        return

    conversation = None
    farmer = None
    outbound_id = None
    ai_response = None

    try:
        async with AsyncSessionLocal() as db:

            # ── STAGE 1: Duplicate Check ──────────────────────────────
            t_db_start = time.time()
            logger.info("STAGE 1: Checking for duplicate message in DB")
            try:
                if await is_duplicate_message(db, parsed.message_id):
                    logger.warning(f"STAGE 1: Duplicate message ID {parsed.message_id} detected in DB. Skipping pipeline.")
                    return
                logger.info("STAGE 1: Message is unique.")
            except Exception as dup_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 1 - Duplicate Check] Message ID: {parsed.message_id}, Error: {dup_err}")

            # ── STAGE 2: Farmer Resolution ────────────────────────────
            logger.info("STAGE 2: Resolving farmer profile in DB")
            try:
                farmer = await get_or_create_farmer(db, parsed.phone_number, sender_name)
                logger.info(f"STAGE 2: Farmer resolved successfully. Farmer ID = {farmer.id}")
            except Exception as db_farmer_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 2 - Farmer Resolution] Phone: {mask_phone_number(parsed.phone_number)}, Error: {db_farmer_err}")
                return

            pref_lang = getattr(farmer, "preferred_language", "te") or "te"

            # ── STAGE 3: Immediate Conversation Storage in DB ─────────
            # Persist incoming record BEFORE executing expensive external calls (STT / Gemini Vision).
            # This ensures cross-worker queries see the record immediately and concurrent retries abort.
            logger.info("STAGE 3: Storing incoming message record in DB before external processing")
            try:
                conversation = await store_incoming_message(db, farmer, parsed)
            except Exception as db_conv_err:
                logger.exception(f"[PIPELINE STAGE FAILED: Stage 3 - Conversation Storage] Farmer ID: {farmer.id}, Error: {db_conv_err}")
                return

            t_db = time.time() - t_db_start

            if conversation is None:
                logger.warning(f"STAGE 3: Duplicate message ID {parsed.message_id} detected during storage. Exiting pipeline immediately.")
                return

            logger.info(f"STAGE 3: Conversation stored successfully. Conversation ID = {conversation.id}")

            # ── STAGE 4: Audio STT (if needed) ────────────────────────
            if parsed.message_type == "audio":
                t_stt_start = time.time()
                if not parsed.media_id:
                    logger.warning(f"STAGE 4: Audio message received with missing media_id for farmer {farmer.id}")
                    ai_response = get_voice_fallback_response(pref_lang)
                else:
                    logger.info("STAGE 4: Audio download and STT transcription started")
                    try:
                        media_result = await download_media_bytes(parsed.media_id)
                        if not media_result:
                            logger.error(f"[PIPELINE STAGE FAILED: Stage 4 - Audio Download] Failed to download media ID: {parsed.media_id}")
                            ai_response = get_voice_fallback_response(pref_lang)
                        else:
                            audio_bytes, mime_type = media_result
                            lang_service = get_language_service()
                            transcription = await lang_service.transcribe_audio(audio_bytes, mime_type)
                            parsed.text_content = transcription.transcription_text
                            conversation.user_message = parsed.text_content
                            db.add(conversation)
                            await db.commit()
                            logger.info(f"STAGE 4: Audio transcribed successfully: '{parsed.text_content[:100]}...'")
                    except Exception as stt_err:
                        logger.exception(f"[PIPELINE STAGE FAILED: Stage 4 - Audio STT] Media ID: {parsed.media_id}, Error: {stt_err}")
                        ai_response = get_voice_fallback_response(pref_lang)
                t_stt = time.time() - t_stt_start

            # ── STAGE 5: AI Processing (Gemini) ───────────────────────
            # Only run if not already set by voice fallback message
            if not ai_response:
                t_ai_start = time.time()
                logger.info("STAGE 5: Generating AI response")
                try:
                    if parsed.message_type == "image":
                        if not parsed.media_id:
                            logger.warning(f"STAGE 5: Image message with missing media_id for farmer {farmer.id}")
                            ai_response = get_image_fallback_response(pref_lang)
                        else:
                            media_result = await download_media_bytes(parsed.media_id)
                            if media_result:
                                image_bytes, mime_type = media_result
                                ai_response = await process_image_message(
                                    db, farmer, conversation, image_bytes, mime_type
                                )
                            else:
                                logger.error(f"[PIPELINE STAGE FAILED: Stage 5 - Image Download] Media ID {parsed.media_id} failed download")
                                ai_response = get_image_fallback_response(pref_lang)
                    elif parsed.text_content and parsed.text_content.strip():
                        ai_response = await process_text_message(
                            db, farmer, conversation
                        )
                    elif parsed.message_type in ["video", "document", "sticker", "contacts", "location", "interactive", "unsupported"] or parsed.message_type not in ["text", "audio", "image"]:
                        logger.info(f"STAGE 5: Handling unsupported media message type '{parsed.message_type}' for farmer {farmer.id}")
                        ai_response = get_unsupported_media_fallback_response(pref_lang)
                    else:
                        logger.warning(f"STAGE 5: Empty message received from farmer {farmer.id}. Using safe fallback.")
                        ai_response = get_fallback_response(pref_lang)

                    if not ai_response or not ai_response.strip():
                        logger.warning(f"STAGE 5: AI generated no text response for farmer {farmer.id}. Using safe fallback.")
                        ai_response = get_fallback_response(pref_lang)
                    else:
                        logger.info(f"STAGE 5: AI response generated ({len(ai_response)} chars): {ai_response[:120]}...")
                except Exception as ai_err:
                    logger.exception(
                        f"[PIPELINE STAGE FAILED: Stage 5 - AI Processing] "
                        f"Farmer ID: {farmer.id}, Message Type: {parsed.message_type}, Text: '{parsed.text_content}', Error: {ai_err}"
                    )
                    ai_response = get_fallback_response(pref_lang)
                t_ai = time.time() - t_ai_start

            # ── STAGE 6: Outbound WhatsApp Send ───────────────────────
            # Ensure safe, non-empty, finalized message within WhatsApp message budget
            if not ai_response or not ai_response.strip():
                ai_response = get_fallback_response(pref_lang)

            # Apply final WhatsApp response length guard
            ai_response = _finalize_whatsapp_response(ai_response)

            logger.info("STAGE 6: Sending outbound WhatsApp message to Meta Cloud API")
            t_outbound_start = time.time()
            try:
                outbound_id = await send_text_message(
                    to_phone=parsed.phone_number,
                    message_text=ai_response,
                )
                if outbound_id:
                    logger.info(f"STAGE 6: WhatsApp send SUCCESS. Outbound Meta ID = {outbound_id}")
                else:
                    logger.error(f"[PIPELINE STAGE FAILED: Stage 6 - Outbound Send] Meta API returned None for {mask_phone_number(parsed.phone_number)}")
            except Exception as send_err:
                logger.exception(
                    f"[PIPELINE STAGE FAILED: Stage 6 - Outbound Send] "
                    f"Phone: {mask_phone_number(parsed.phone_number)}, Error: {send_err}"
                )
            t_outbound = time.time() - t_outbound_start

            # ── STAGE 7: Database Delivery Status Update ──────────────
            if conversation:
                logger.info("STAGE 7: Updating delivery status in database")
                try:
                    conversation.outbound_message_id = outbound_id
                    conversation.delivery_status = "sent" if outbound_id else "failed"
                    conversation.ai_response = ai_response
                    db.add(conversation)
                    await db.commit()
                    logger.info(f"STAGE 7: Delivery status set to '{conversation.delivery_status}'")

                    if not outbound_id:
                        try:
                            from sqlalchemy import select
                            from src.core.models import Conversation
                            recent_statuses = (await db.execute(
                                select(Conversation.delivery_status)
                                .order_by(Conversation.created_at.desc())
                                .limit(20)
                            )).scalars().all()
                            if recent_statuses:
                                total_cnt = len(recent_statuses)
                                failed_cnt = sum(1 for s in recent_statuses if s == "failed")
                                if total_cnt >= 5 and (failed_cnt / total_cnt) > 0.10:
                                    from src.core.alerting import dispatch_founder_alert, AlertCategory, AlertSeverity
                                    import asyncio
                                    asyncio.create_task(dispatch_founder_alert(
                                        category=AlertCategory.HIGH_DELIVERY_FAILURE,
                                        severity=AlertSeverity.WARNING,
                                        component="whatsapp_gateway",
                                        summary=f"WhatsApp outbound failure rate is {int((failed_cnt/total_cnt)*100)}% ({failed_cnt}/{total_cnt} recent messages failed).",
                                        recommended_action="Inspect Meta Business Manager account quality status and verify active billing.",
                                        details={"failed_count": failed_cnt, "total_recent": total_cnt}
                                    ))
                        except Exception as alert_check_err:
                            logger.debug(f"Delivery rate alert check skipped: {alert_check_err}")
                except Exception as db_status_err:
                    logger.exception(f"[PIPELINE STAGE FAILED: Stage 7 - DB Status Update] Conversation ID: {conversation.id}, Error: {db_status_err}")

            # ── STAGE 8: Mark Message as Read ──────────────────────────
            logger.info("STAGE 8: Marking message as read with Meta API")
            try:
                await mark_message_as_read(parsed.message_id)
                logger.info("STAGE 8: Read receipt sent.")
            except Exception as read_err:
                logger.warning(f"STAGE 8: Read receipt warning for {parsed.message_id}: {read_err}")

            total_pipeline_time = time.time() - pipeline_start
            logger.info("=" * 80)
            logger.info(
                f"[PIPELINE TIMING] msg={parsed.message_id} phone={mask_phone_number(parsed.phone_number)} "
                f"total={total_pipeline_time:.2f}s (db={t_db:.2f}s stt={t_stt:.2f}s ai={t_ai:.2f}s outbound={t_outbound:.2f}s)"
            )
            logger.info("BACKGROUND PIPELINE COMPLETED")
            logger.info("=" * 80)

    except Exception as pipeline_err:
        logger.exception(f"[CRITICAL PIPELINE FAILURE] Unhandled error in background pipeline for message {parsed.message_id}: {pipeline_err}")
        if conversation and not outbound_id:
            try:
                pref_lang = getattr(farmer, "preferred_language", "te") if farmer else "te"
                emergency_fallback = get_fallback_response(pref_lang)
                outbound_id = await send_text_message(to_phone=parsed.phone_number, message_text=emergency_fallback)
                conversation.outbound_message_id = outbound_id
                conversation.delivery_status = "sent" if outbound_id else "failed"
                conversation.ai_response = emergency_fallback
                async with AsyncSessionLocal() as db_recovery:
                    db_recovery.add(conversation)
                    await db_recovery.commit()
            except Exception as recovery_err:
                logger.exception(f"Recovery fallback send failed: {recovery_err}")
    finally:
        release_in_flight_lock(parsed.message_id)