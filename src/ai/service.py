import time
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Farmer, Conversation, Crop, Farm
from src.core.logging import logger
from src.ai.repository import AIRepository
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse, MultimodalDiagnosisResponse
from src.crop_health.service import CropHealthService
from src.crop_health.repository import CropHealthRepository
from src.crop_health.schemas import CropHealthCreate
from src.crops.repository import CropRepository
from src.farmers.repository import FarmerRepository
from src.decision_engine.engine import DecisionEngine
from src.decision_engine.models import FarmerInput, DecisionType
from src.ai.prompts import (
    BHOOMIMITRA_SYSTEM_PROMPT,
    build_farmer_context,
    get_fallback_response,
)
from src.ai.gemini_client import generate_response

class AIService:
    def __init__(self, repository: AIRepository):
        self.repository = repository
        self.decision_engine = DecisionEngine()

    async def generate_ai_response(self, request: AIGenerateRequest) -> AIGenerateResponse:
        service_start_time = time.time()
        try:
            # 1. Fetch farmer profile and history for context
            profile = await self.repository.get_farmer_profile(request.farmer_id)
            history_records = await self.repository.get_conversation_history(request.farmer_id)

            # 1.2 Resolve contextual crop and problem (including from recent history for follow-ups)
            effective_crop = profile.current_crop if profile else None
            if not effective_crop:
                effective_crop = self.decision_engine.extract_crop(request.message)
                if not effective_crop:
                    for h in history_records:
                        if h.user_message:
                            c = self.decision_engine.extract_crop(h.user_message)
                            if c:
                                effective_crop = c
                                break

            effective_problem = self.decision_engine.extract_problem(request.message)
            if not effective_problem:
                for h in history_records:
                    if h.user_message:
                        p = self.decision_engine.extract_problem(h.user_message)
                        if p:
                            effective_problem = p
                            break

            # 1.5 Evaluate Decision Engine BEFORE invoking LLM / Gemini
            location_str = None
            if profile:
                loc_parts = [str(p) for p in [getattr(profile, "district", None), getattr(profile, "state", None)] if p and isinstance(p, str)]
                if loc_parts:
                    location_str = ", ".join(loc_parts)

            farmer_input = FarmerInput(
                message=request.message,
                crop=effective_crop,
                growth_stage=None,
                problem=effective_problem,
                location=location_str,
            )

            decision = self.decision_engine.evaluate(farmer_input)
            logger.info(
                f"[DECISION ENGINE] Farmer: {request.farmer_id} | Decision: {decision.decision_type.value} "
                f"| Risk: {decision.risk_level.value} | Reasons: {decision.reasons}"
            )

            # If SAFE_FALLBACK or ASK_CLARIFICATION, return immediately without calling Gemini
            if decision.decision_type in (DecisionType.SAFE_FALLBACK, DecisionType.ASK_CLARIFICATION):
                return AIGenerateResponse(
                    response_text=decision.response,
                    intent=decision.decision_type.value,
                    confidence=1.0,
                    provider_used="decision_engine",
                )

            # 2. Build farmer context string
            farmer_context = build_farmer_context(
                crop=profile.current_crop if profile else None,
                district=profile.district if profile else None,
                state=profile.state if profile else None,
                land_size=profile.land_size_acres if profile else None,
            )

            # 3. Fetch farmer long-term memory context
            from src.memory.service import FarmerMemoryService
            from src.memory.repository import FarmerMemoryRepository
            mem_repo = FarmerMemoryRepository(self.repository.session)
            mem_service = FarmerMemoryService(mem_repo)
            memory_context = await mem_service.format_memory_for_system_prompt(request.farmer_id)

            # 3.5 Retrieve trusted agricultural RAG knowledge
            rag_context_text = ""
            try:
                from src.rag.service import RAGService
                from src.rag.repository import RAGRepository
                rag_repo = RAGRepository(self.repository.session)
                rag_service = RAGService(rag_repo)
                rag_results = await rag_service.search_knowledge(
                    query=request.message,
                    top_k=3,
                    state=profile.state if profile else None,
                    crop=profile.current_crop if profile else None,
                )
                if rag_results:
                    rag_snippets = [f"• Document: {r.document_title} ({r.source}): {r.chunk_text}" for r in rag_results]
                    rag_context_text = "=== RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) ===\n" + "\n".join(rag_snippets)
            except Exception as rag_err:
                logger.warning(f"RAG knowledge retrieval warning: {rag_err}")

            # 4. Build system prompt combining profile, memory engine, and RAG ground truth
            full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{farmer_context}\n\n{memory_context}"
            if rag_context_text:
                full_system_prompt += f"\n\n{rag_context_text}"

            # 5. Prepare conversation history for Gemini (oldest first)
            history = []
            for record in reversed(history_records):
                if record.user_message:
                    history.append({"role": "user", "parts": record.user_message})
                if record.ai_response:
                    # Clean shop section from the history to prevent LLM contamination
                    clean_response = record.ai_response.split("Available Nearby Shops:")[0].split("🏬")[0].strip()
                    history.append({"role": "model", "parts": clean_response})

            # 6. Call Gemini API
            logger.info(f"Processing AI request for farmer {request.farmer_id}: '{request.message[:80]}...'")
            
            ai_text = await generate_response(
                system_prompt=full_system_prompt,
                conversation_history=history,
                user_message=request.message,
                timeout_seconds=20,
            )

            if ai_text is None or not ai_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI provider timed out or returned empty response."
                )

            # 6.5 Validate generated AI response for agricultural safety
            from src.decision_engine.validators import validate_generated_ai_response
            safety_result = validate_generated_ai_response(
                response_text=ai_text,
                user_message=request.message,
            )
            if not safety_result.is_safe:
                logger.warning(
                    f"[AI SAFETY VIOLATION DETECTED] Farmer: {request.farmer_id} | "
                    f"Violations: {safety_result.violations} | Reasons: {safety_result.reasons}"
                )
                ai_text = safety_result.safe_response

            total_ai_time = time.time() - service_start_time
            logger.info(
                f"[AI SERVICE GENERATION SUCCESS]\n"
                f"  Farmer ID    : {request.farmer_id}\n"
                f"  Total Time   : {total_ai_time:.2f}s\n"
                f"  Output Chars : {len(ai_text)}\n"
                f"  Preview      : '{ai_text[:120]}...'"
            )

            # 7. Automatic memory extraction from exchange (runs safely & swiftly)
            try:
                await mem_service.extract_and_update_memory(
                    farmer_id=request.farmer_id,
                    user_message=request.message,
                    ai_response=ai_text
                )
            except Exception as mem_err:
                logger.warning(f"Automatic memory extraction warning for farmer {request.farmer_id}: {mem_err}")

            # 8. Return structured response
            return AIGenerateResponse(
                response_text=ai_text,
                intent=None,
                confidence=None,
                provider_used="gemini"
            )

        except HTTPException:
            raise
        except Exception as e:
            elapsed = time.time() - service_start_time
            logger.exception(f"[AI SERVICE ERROR] Failed generating AI response after {elapsed:.2f}s: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while communicating with the AI provider: {str(e)}"
            )

async def process_text_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
) -> str:
    """
    Backward compatibility wrapper for the gateway router.
    Instantiates the new architecture components to fulfill legacy requests.
    """
    repo = AIRepository(db)
    service = AIService(repo)
    
    logger.info(f"[PROCESS MSG START] Farmer: {farmer.id} | Query: '{conversation.user_message}'")
    
    request = AIGenerateRequest(
        farmer_id=farmer.id,
        conversation_id=conversation.id,
        message=conversation.user_message or ""
    )
    
    try:
        response = await service.generate_ai_response(request)
        ai_response_text = response.response_text
        logger.info(f"[PROCESS MSG RAW GEMINI] Length: {len(ai_response_text) if ai_response_text else 0} | Response: '{ai_response_text}'")
    except HTTPException:
        logger.warning(f"AI unavailable for farmer {farmer.id}. Using fallback.")
        ai_response_text = get_fallback_response(farmer.preferred_language)

    # Enrich with nearby shop inventory recommendations if products match
    try:
        from src.shops.service import enrich_response_with_shops
        logger.info("[PROCESS MSG] Invoking enrich_response_with_shops()")
        ai_response_text = await enrich_response_with_shops(
            db, conversation.user_message or "", ai_response_text
        )
    except Exception as err:
        logger.warning(f"Failed to enrich response with shops: {err}")

    # Run final post-generation safety & language validation
    from src.decision_engine.validators import validate_generated_ai_response
    final_safety = validate_generated_ai_response(
        response_text=ai_response_text,
        user_message=conversation.user_message or "",
    )
    if not final_safety.is_safe:
        logger.warning(f"[PROCESS MSG SAFETY INTERCEPT] Violations: {final_safety.violations} | Reasons: {final_safety.reasons}")
        ai_response_text = final_safety.safe_response

    logger.info(f"[PROCESS MSG FINAL] Length: {len(ai_response_text)} | Final response immediately before DB save: '{ai_response_text}'")

    conversation.ai_response = ai_response_text
    db.add(conversation)
    await db.commit()

    return ai_response_text


async def process_image_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """
    Multimodal pipeline: Takes an image and optional caption, queries Gemini Vision,
    and returns agronomic advice.
    """
    repo = AIRepository(db)
    
    # 1. Fetch farmer profile
    profile = await repo.get_farmer_profile(farmer.id)
    farmer_context = build_farmer_context(
        crop=profile.current_crop if profile else None,
        district=profile.district if profile else None,
        state=profile.state if profile else None,
        land_size=profile.land_size_acres if profile else None,
    )
    
    from src.memory.service import FarmerMemoryService
    from src.memory.repository import FarmerMemoryRepository
    mem_repo = FarmerMemoryRepository(db)
    mem_service = FarmerMemoryService(mem_repo)
    memory_context = await mem_service.format_memory_for_system_prompt(farmer.id)

    # Add a vision-specific system prompt instruction enforcing JSON
    full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\nThe user has uploaded an image of their crop. Diagnose any visible diseases, pests, or deficiencies.\n" \
                         "You MUST return a strictly valid JSON object matching this exact schema:\n" \
                         '{"disease_name": "Name", "confidence_score": 0.95, "severity": "low/medium/high", "symptoms": "Visible symptoms", "treatment_recommendation": "Steps to fix", "friendly_whatsapp_reply": "Natural language reply for the farmer"}\n' \
                         "Provide actionable agronomic advice.\n\n" \
                         f"{farmer_context}\n\n{memory_context}"
    
    # 2. History
    history_records = await repo.get_conversation_history(farmer.id)
    history_records.reverse()
    history = []
    for record in history_records:
        if record.user_message:
            history.append({"role": "user", "parts": record.user_message})
        if record.ai_response:
            # Clean shop section from the history to prevent LLM contamination
            clean_response = record.ai_response.split("Available Nearby Shops:")[0].split("🏬")[0].strip()
            history.append({"role": "model", "parts": clean_response})
            
    # 3. Call Gemini Multimodal
    from src.ai.gemini_client import generate_multimodal_response
    import json
    
    user_caption = conversation.user_message or "Please analyze this image."
    
    try:
        ai_response_text = await generate_multimodal_response(
            system_prompt=full_system_prompt,
            conversation_history=history,
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_message=user_caption
        )
        if not ai_response_text:
            raise Exception("Empty response from AI")
            
        # Parse the structured JSON output
        parsed_json = json.loads(ai_response_text)
        diagnosis_data = MultimodalDiagnosisResponse(**parsed_json)
        reply_text = diagnosis_data.friendly_whatsapp_reply

        # Validate AI vision response for safety
        from src.decision_engine.validators import validate_generated_ai_response
        safety_res = validate_generated_ai_response(
            response_text=f"{reply_text} {diagnosis_data.treatment_recommendation or ''}",
            user_message=user_caption,
        )
        if not safety_res.is_safe:
            logger.warning(f"[VISION SAFETY VIOLATION] Farmer {farmer.id}: {safety_res.reasons}")
            reply_text = safety_res.safe_response

        # 4. Save to Crop Health Module
        # Attempt to find the farmer's most recent crop to link the diagnosis
        result = await db.execute(
            select(Crop.id)
            .join(Farm)
            .where(Farm.farmer_id == farmer.id)
            .order_by(Crop.created_at.desc())
            .limit(1)
        )
        crop_id = result.scalar_one_or_none()

        if crop_id:
            crop_health_service = CropHealthService(
                repository=CropHealthRepository(db),
                crop_repository=CropRepository(db),
                farmer_repository=FarmerRepository(db)
            )
            create_data = CropHealthCreate(
                crop_id=crop_id,
                farmer_id=farmer.id,
                image_url=None, # Media ID is handled via Conversation temporarily
                symptoms=diagnosis_data.symptoms,
                disease_name=diagnosis_data.disease_name,
                diagnosis_result=diagnosis_data.friendly_whatsapp_reply,
                treatment_recommendation=diagnosis_data.treatment_recommendation,
                confidence_score=diagnosis_data.confidence_score,
            )
            await crop_health_service.create_diagnosis(create_data)
            logger.info(f"Structured diagnosis saved to CropHealth for farmer {farmer.id}")

        # Update Farmer Memory with diagnosis
        try:
            await mem_service.extract_and_update_memory(
                farmer_id=farmer.id,
                user_message=user_caption,
                ai_response=f"Disease Diagnosed: {diagnosis_data.disease_name}. Treatment: {diagnosis_data.treatment_recommendation}"
            )
        except Exception as mem_err:
            logger.warning(f"Memory update failed for image message: {mem_err}")

    except Exception as e:
        logger.warning(f"AI Vision unavailable or failed to parse for farmer {farmer.id}: {e}")
        reply_text = get_fallback_response(farmer.preferred_language)
        
    conversation.ai_response = reply_text
    db.add(conversation)
    await db.commit()
    
    return reply_text


async def process_voice_message(
    db: AsyncSession,
    farmer: Farmer,
    conversation: Conversation,
    audio_bytes: bytes,
    mime_type: str,
) -> str:
    """
    Voice message pipeline:
    1. Transcribes voice audio using VoiceService.
    2. If transcription fails or returns empty:
       - Returns a localized retry message.
       - Does NOT call Gemini.
    3. If transcription succeeds:
       - Sets conversation.user_message to the transcribed text.
       - Reuses the existing process_text_message pipeline
         (runs Decision Engine -> Gemini (if ANSWER) -> Safety Validator -> Final response).
    """
    from src.voice.service import get_voice_service
    voice_service = get_voice_service()

    preferred_lang = getattr(farmer, "preferred_language", "te-IN") or "te-IN"
    lang_code = "te-IN" if "te" in preferred_lang.lower() else "en-IN"

    transcription = await voice_service.transcribe_audio(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        language_code=lang_code,
    )

    if not transcription.is_success or not transcription.text.strip():
        logger.warning(
            f"[VOICE PIPELINE] STT failed or empty for farmer {farmer.id}: {transcription.error_message}"
        )
        retry_msg = voice_service.get_stt_failure_message(preferred_lang)
        conversation.ai_response = retry_msg
        db.add(conversation)
        await db.commit()
        return retry_msg

    # Successfully transcribed speech into text
    logger.info(
        f"[VOICE PIPELINE] Farmer {farmer.id} audio transcribed successfully: '{transcription.text}'"
    )
    conversation.user_message = transcription.text
    db.add(conversation)
    await db.commit()

    # Re-route directly through existing text pipeline
    return await process_text_message(db, farmer, conversation)

