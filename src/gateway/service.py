"""
BhoomiMitra AI — WhatsApp Message Service

Business logic layer for processing incoming WhatsApp messages.
Handles farmer upsert, duplicate detection, conversation storage,
and orchestrating the full pipeline (STT -> AI -> Outbound).
"""

import time
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.core.models import Farmer, FarmerProfile, Conversation
from src.core.database import AsyncSessionLocal
from src.gateway.schemas import ParsedIncomingMessage, mask_phone_number
from src.core.logging import logger

from src.gateway.whatsapp_client import (
    download_media_bytes,
    send_text_message,
    mark_message_as_read,
    upload_media_bytes,
    send_audio_message,
)
from src.config import get_settings, Settings
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
    farmer = Farmer(phone_number=phone_number, preferred_language="te")
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
            f"[DUPLICATE DETECTED: DB_UNIQUE_VIOLATION] Duplicate message_id '{message.message_id}' "
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
    logger.info(f"BACKGROUND PIPELINE STARTED [msg_id={parsed.message_id}]")
    logger.info(f"  Message ID  : {parsed.message_id}")
    logger.info(f"  Sender Phone: {mask_phone_number(parsed.phone_number)}")
    logger.info(f"  Message Type: {parsed.message_type}")
    if parsed.text_content:
        logger.info(f"  Message Text: {parsed.text_content[:150]}")
    logger.info("=" * 80)

    # Concurrency Lock Check: Prevent in-flight execution races for the same message ID
    if not acquire_in_flight_lock(parsed.message_id):
        logger.warning(
            f"[DUPLICATE DETECTED: IN_FLIGHT] STAGE 0: Message ID '{parsed.message_id}' is already actively IN-FLIGHT in another background task. "
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
                    logger.warning(f"[DUPLICATE DETECTED: DB_EXISTS] STAGE 1: Duplicate message ID {parsed.message_id} detected in DB. Skipping pipeline.")
                    return
                logger.info(f"STAGE 1: Message {parsed.message_id} is unique in DB.")
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
                logger.warning(f"[DUPLICATE DETECTED: DB_STORAGE_COLLISION] STAGE 3: Duplicate message ID {parsed.message_id} detected during storage. Exiting pipeline immediately.")
                return

            logger.info(f"[UNIQUE MESSAGE STORED] STAGE 3: Conversation stored successfully. Conversation ID = {conversation.id}")

            # ── STAGE 4: Audio STT (if needed) ────────────────────────
            if parsed.message_type == "audio":
                t_stt_start = time.time()
                logger.info(
                    f"[VOICE PIPELINE START] msg_id={parsed.message_id} "
                    f"media_id={parsed.media_id} mime_type={parsed.media_mime_type}"
                )
                if not parsed.media_id:
                    logger.warning(f"STAGE 4: Audio message received with missing media_id for farmer {farmer.id}")
                    ai_response = get_voice_fallback_response(pref_lang)
                else:
                    logger.info(f"[VOICE MEDIA DOWNLOAD START] media_id={parsed.media_id}")
                    try:
                        media_result = await download_media_bytes(parsed.media_id)
                        if not media_result:
                            logger.error(f"[VOICE MEDIA DOWNLOAD FAILED] media_id={parsed.media_id}")
                            logger.error(f"[PIPELINE STAGE FAILED: Stage 4 - Audio Download] Failed to download media ID: {parsed.media_id}")
                            ai_response = get_voice_fallback_response(pref_lang)
                        else:
                            audio_bytes, mime_type = media_result
                            logger.info(
                                f"[VOICE MEDIA DOWNLOAD SUCCESS] media_id={parsed.media_id} "
                                f"bytes={len(audio_bytes)} mime_type={mime_type}"
                            )
                            logger.info("[VOICE STT START]")
                            lang_service = get_language_service()
                            transcription = await lang_service.transcribe_audio(audio_bytes, mime_type)
                            if not transcription or not transcription.transcription_text or not transcription.transcription_text.strip():
                                logger.warning(f"STAGE 4: Empty or unparseable audio transcript for farmer {farmer.id}")
                                ai_response = get_voice_fallback_response(pref_lang)
                            else:
                                parsed.text_content = transcription.transcription_text.strip()
                                conversation.user_message = parsed.text_content
                                db.add(conversation)
                                await db.commit()
                                logger.info(
                                    f"[VOICE STT SUCCESS] language={transcription.detected_language} "
                                    f"confidence={transcription.confidence} "
                                    f"transcript_length={len(parsed.text_content)}"
                                )
                                logger.info(f"STAGE 4: Audio transcribed successfully: '{parsed.text_content[:100]}...'")
                    except Exception as stt_err:
                        underlying_cause = getattr(stt_err, "__cause__", None) or stt_err
                        cause_type = type(underlying_cause).__name__
                        cause_reason = str(underlying_cause).strip()
                        logger.error(
                            f"[VOICE STT FAILED] exception_type={cause_type} "
                            f"status={getattr(stt_err, 'status_code', 'N/A')} reason={cause_reason}"
                        )
                        logger.error(
                            f"[PIPELINE STAGE FAILED: Stage 4 - Audio STT] Media ID: {parsed.media_id} | "
                            f"Cause: {cause_type} | Reason: {cause_reason}"
                        )
                        logger.exception(f"[PIPELINE STAGE FAILED: Stage 4 - Audio STT] Media ID: {parsed.media_id}, Error: {stt_err}")
                        ai_response = get_voice_fallback_response(pref_lang)
                t_stt = time.time() - t_stt_start

            # ── STAGE 5: AI Processing (Gemini) ───────────────────────
            # Only run if not already set by voice fallback message
            from src.language.detector import detect_language
            stt_lang = (
                transcription.detected_language[:2].lower()
                if ("transcription" in locals() and transcription and getattr(transcription, "detected_language", None))
                else None
            )
            effective_fallback = stt_lang or pref_lang
            active_lang = detect_language(parsed.text_content, fallback=effective_fallback) if parsed.text_content else effective_fallback

            if not ai_response:
                t_ai_start = time.time()
                logger.info(f"[VOICE AI START] Language={active_lang}")
                logger.info(f"STAGE 5: Generating AI response (Language: {active_lang})")
                try:
                    if parsed.message_type == "image":
                        if not parsed.media_id:
                            logger.warning(f"STAGE 5: Image message with missing media_id for farmer {farmer.id}")
                            ai_response = get_image_fallback_response(active_lang)
                        else:
                            media_result = await download_media_bytes(parsed.media_id)
                            if media_result:
                                image_bytes, mime_type = media_result
                                ai_response = await process_image_message(
                                    db, farmer, conversation, image_bytes, mime_type
                                )
                            else:
                                logger.error(f"[PIPELINE STAGE FAILED: Stage 5 - Image Download] Media ID {parsed.media_id} failed download")
                                ai_response = get_image_fallback_response(active_lang)
                    elif parsed.text_content and parsed.text_content.strip():
                        ai_response = await process_text_message(
                            db, farmer, conversation
                        )
                    elif parsed.message_type in ["video", "document", "sticker", "contacts", "location", "interactive", "unsupported"] or parsed.message_type not in ["text", "audio", "image"]:
                        logger.info(f"STAGE 5: Handling unsupported media message type '{parsed.message_type}' for farmer {farmer.id}")
                        ai_response = get_unsupported_media_fallback_response(active_lang)
                    else:
                        logger.warning(f"STAGE 5: Empty message received from farmer {farmer.id}. Using safe fallback.")
                        ai_response = get_fallback_response(active_lang)

                    if not ai_response or not ai_response.strip():
                        logger.warning(f"STAGE 5: AI generated no text response for farmer {farmer.id}. Using safe fallback.")
                        ai_response = get_fallback_response(active_lang)
                    else:
                        if parsed.message_type == "audio":
                            logger.info(f"[VOICE AI SUCCESS] Response length={len(ai_response)} chars")
                        logger.info(f"STAGE 5: AI response generated ({len(ai_response)} chars): {ai_response[:120]}...")
                except Exception as ai_err:
                    if parsed.message_type == "audio":
                        logger.error(f"[VOICE AI FAILED] error={ai_err}")
                    logger.exception(
                        f"[PIPELINE STAGE FAILED: Stage 5 - AI Processing] "
                        f"Farmer ID: {farmer.id}, Message Type: {parsed.message_type}, Text: '{parsed.text_content}', Error: {ai_err}"
                    )
                    ai_response = get_fallback_response(active_lang)
                t_ai = time.time() - t_ai_start

            # ── STAGE 6: Outbound WhatsApp Send ───────────────────────
            # Ensure safe, non-empty, finalized message within WhatsApp message budget
            if not ai_response or not ai_response.strip():
                ai_response = get_fallback_response(active_lang)

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

            # ── STAGE 6B: Outbound Voice Note Send (Omni Voice: Voice, Text, Image) ──
            settings = get_settings()

            # Voice responses are dispatched for incoming queries (audio, text, image).
            # Active by default unless explicitly disabled (enable_voice_responses=False).
            voice_flag = getattr(settings, "enable_voice_responses", None)
            is_explicitly_disabled = (
                voice_flag is False
                if not isinstance(settings, Settings)
                else (
                    "enable_voice_responses" in getattr(settings, "model_fields_set", set())
                    and settings.enable_voice_responses is False
                )
            )
            is_supported_inbound = parsed.message_type in ("audio", "text", "image")
            should_send_voice = (
                is_supported_inbound
                and not is_explicitly_disabled
                and bool(ai_response and ai_response.strip())
            )

            if should_send_voice:
                from src.language.languages import get_language
                lang_meta = get_language(active_lang)
                tts_lang_code = lang_meta.tts_code if lang_meta and lang_meta.tts_code else f"{active_lang}-IN"

                logger.info(
                    f"[VOICE TTS START]\n"
                    f"  language={tts_lang_code}\n"
                    f"  text_length={len(ai_response)}"
                )
                try:
                    lang_service = get_language_service()
                    # Synthesize speech using the exact same ai_response already delivered as text
                    audio_chunks = await lang_service.synthesize_speech(ai_response, active_lang)
                    if audio_chunks:
                        primary_audio_chunk = audio_chunks[0]
                        mime_type = "audio/ogg"
                        logger.info(
                            f"[VOICE TTS SUCCESS]\n"
                            f"  audio_bytes={len(primary_audio_chunk)}\n"
                            f"  mime_type={mime_type}"
                        )

                        logger.info("[VOICE AUDIO UPLOAD START]")
                        try:
                            media_id = await upload_media_bytes(primary_audio_chunk, mime_type=mime_type)
                            if media_id:
                                masked_media = media_id[:6] + "..." if len(media_id) > 6 else media_id
                                logger.info(
                                    f"[VOICE AUDIO UPLOAD SUCCESS]\n"
                                    f"  media_id={masked_media}"
                                )

                                logger.info("[VOICE AUDIO SEND START]")
                                audio_msg_id = await send_audio_message(parsed.phone_number, media_id)
                                if audio_msg_id:
                                    logger.info(f"[VOICE AUDIO SEND SUCCESS] audio_msg_id={audio_msg_id}")
                                else:
                                    logger.warning(
                                        "[VOICE AUDIO FAIL-SOFT]\n"
                                        "  reason=send_audio_message returned None; text response already delivered"
                                    )
                            else:
                                logger.warning(
                                    "[VOICE AUDIO FAIL-SOFT]\n"
                                    "  reason=upload_media_bytes returned None; text response already delivered"
                                )
                        except Exception as upload_send_err:
                            logger.warning(
                                f"[VOICE AUDIO FAIL-SOFT]\n"
                                f"  reason={upload_send_err}"
                            )
                    else:
                        logger.warning(
                            f"[VOICE TTS FAIL-SOFT]\n"
                            f"  reason=TTS returned no audio chunks for language '{active_lang}'; text response already delivered"
                        )
                except Exception as tts_err:
                    logger.warning(
                        f"[VOICE TTS FAIL-SOFT]\n"
                        f"  reason={tts_err}"
                    )

            # ── STAGE 7: Database Delivery Status Update ──────────────
            if conversation:
                logger.info("STAGE 7: Updating delivery status in database")
                try:
                    pipeline_elapsed = time.time() - pipeline_start
                    conversation.outbound_message_id = outbound_id
                    conversation.delivery_status = "sent" if outbound_id else "failed"
                    conversation.ai_response = ai_response
                    conversation.replied_at = datetime.utcnow()
                    conversation.response_time_seconds = round(pipeline_elapsed, 2)
                    db.add(conversation)
                    await db.commit()
                    logger.info(
                        f"STAGE 7: Delivery status set to '{conversation.delivery_status}' "
                        f"(response_time={conversation.response_time_seconds}s)"
                    )

                    if not outbound_id:
                        try:
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
            if parsed.message_type == "audio":
                logger.info(
                    f"[VOICE PIPELINE COMPLETE] msg_id={parsed.message_id} "
                    f"total={total_pipeline_time:.2f}s (stt={t_stt:.2f}s ai={t_ai:.2f}s outbound={t_outbound:.2f}s)"
                )
            logger.info(
                f"[PIPELINE TIMING] msg={parsed.message_id} phone={mask_phone_number(parsed.phone_number)} "
                f"total={total_pipeline_time:.2f}s (db={t_db:.2f}s stt={t_stt:.2f}s ai={t_ai:.2f}s outbound={t_outbound:.2f}s)"
            )
            logger.info("BACKGROUND PIPELINE COMPLETED")
            logger.info("=" * 80)

    except Exception as pipeline_err:
        logger.exception(f"[CRITICAL PIPELINE FAILURE] Unhandled error in background pipeline for message {parsed.message_id}: {pipeline_err}")
        if conversation:
            try:
                pipeline_elapsed = time.time() - pipeline_start
                pref_lang = getattr(farmer, "preferred_language", "te") if farmer else "te"
                emergency_fallback = get_fallback_response(pref_lang)

                # Send outbound fallback message ONLY if no outbound message was already dispatched
                if not outbound_id:
                    try:
                        outbound_id = await send_text_message(
                            to_phone=parsed.phone_number,
                            message_text=emergency_fallback,
                        )
                    except Exception as send_fallback_err:
                        logger.warning(f"Recovery outbound send failed: {send_fallback_err}")
                        outbound_id = None

                status_val = "sent" if outbound_id else "failed"
                final_response = (
                    ai_response
                    if (ai_response and ai_response.strip())
                    else emergency_fallback
                )
                replied_timestamp = datetime.utcnow()
                response_secs = round(pipeline_elapsed, 2)
                conv_id = conversation.id

                # Query fresh conversation in a new isolated session to avoid session attachment conflicts
                async with AsyncSessionLocal() as db_recovery:
                    recovery_conv = (
                        await db_recovery.execute(
                            select(Conversation).where(Conversation.id == conv_id)
                        )
                    ).scalar_one_or_none()

                    if recovery_conv:
                        recovery_conv.outbound_message_id = outbound_id
                        recovery_conv.delivery_status = status_val
                        recovery_conv.ai_response = final_response
                        recovery_conv.replied_at = replied_timestamp
                        recovery_conv.response_time_seconds = response_secs
                        await db_recovery.commit()
                        logger.info(
                            f"[EMERGENCY RECOVERY COMMITTED] Conversation {conv_id} updated: "
                            f"status='{status_val}', response_time={response_secs}s"
                        )
            except Exception as recovery_err:
                logger.exception(f"Recovery fallback commit failed: {recovery_err}")
    finally:
        release_in_flight_lock(parsed.message_id)