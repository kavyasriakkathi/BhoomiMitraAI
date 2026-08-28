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
from src.ai.prompts import (
    BHOOMIMITRA_SYSTEM_PROMPT,
    build_farmer_context,
    get_fallback_response,
)
from src.config import get_settings
from src.ai.gemini_client import generate_response

class AIService:
    def __init__(self, repository: AIRepository):
        self.repository = repository

    async def generate_ai_response(self, request: AIGenerateRequest) -> AIGenerateResponse:
        service_start_time = time.time()
        try:
            # 1. Fetch farmer profile for context
            profile = await self.repository.get_farmer_profile(request.farmer_id)

            # 2. Fetch conversation history (oldest first for Gemini)
            history_records = await self.repository.get_conversation_history(request.farmer_id)
            history_records.reverse()
            history = []
            recent_context_crop = None

            from src.rag.service import extract_crop_from_text
            query_crop = extract_crop_from_text(request.message)

            for record in history_records:
                if record.user_message:
                    history.append({"role": "user", "parts": record.user_message})
                    c = extract_crop_from_text(record.user_message)
                    if c:
                        recent_context_crop = c
                if record.ai_response:
                    # Clean all structured enrichment sections from history to prevent LLM prompt contamination
                    clean_response = record.ai_response
                    for marker in ["Available Nearby Shops:", "🏬", "📊", "🌤️", "🌡️", "🏛️", "🎫", "📜", "👨‍🌾", "🚨", "🆘"]:
                        clean_response = clean_response.split(marker)[0]
                    clean_response = clean_response.strip()
                    if clean_response:
                        history.append({"role": "model", "parts": clean_response})

            # Priority for crop:
            # 1. Explicit crop mentioned in current message (e.g. "టమాటా" / Tomato overrides profile's "Cotton")
            # 2. If short follow-up and query_crop is None, crop from recent conversation history
            # 3. Farmer profile current_crop
            effective_crop = query_crop or (recent_context_crop if recent_context_crop else (profile.current_crop if profile else None))

            # Build farmer context string
            farmer_context = build_farmer_context(
                crop=effective_crop or (profile.current_crop if profile else None),
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

            # Build enriched RAG query for short follow-ups (e.g. "ఎకరానికి ఎంత కావాలి?" / "ఈ వ్యాధికి ఎంత మందు వేయాలి?")
            rag_query = request.message
            msg_tokens = request.message.strip().split()
            if not query_crop and len(msg_tokens) <= 7 and history_records:
                recent_user_msgs = [r.user_message for r in history_records[-2:] if r.user_message]
                if recent_user_msgs:
                    rag_query = f"{' '.join(recent_user_msgs)} {request.message}"

            # 3.5 Retrieve trusted agricultural RAG knowledge
            rag_context_text = ""
            try:
                from src.rag.service import RAGService
                from src.rag.repository import RAGRepository
                rag_repo = RAGRepository(self.repository.session)
                rag_service = RAGService(rag_repo)
                rag_results = await rag_service.search_knowledge(
                    query=rag_query,
                    top_k=3,
                    state=profile.state if profile else None,
                    crop=effective_crop,
                )
                if rag_results:
                    rag_snippets = [f"• Document: {r.document_title} (Crop: {r.crop or 'General'}, Source: {r.source}): {r.chunk_text}" for r in rag_results]
                    rag_context_text = (
                        "=== RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) ===\n"
                        "CRITICAL INSTRUCTION: The following knowledge is verified agronomic ground truth. When the farmer asks about disease diagnosis, symptoms, management, or dosage, you MUST prioritize and use these verified treatments/dosages and translate them directly into the response in the farmer's language (Telugu or English):\n"
                        + "\n".join(rag_snippets)
                    )
            except Exception as rag_err:
                logger.warning(f"RAG knowledge retrieval warning: {rag_err}")

            # 4. Build system prompt combining profile, memory engine, and RAG ground truth
            full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{farmer_context}\n\n{memory_context}"
            if rag_context_text:
                full_system_prompt += f"\n\n{rag_context_text}"


            # 6. Call Gemini API
            logger.info(f"Processing AI request for farmer {request.farmer_id}: '{request.message[:80]}...'")
            
            ai_text = await generate_response(
                system_prompt=full_system_prompt,
                conversation_history=history,
                user_message=request.message,
                timeout_seconds=getattr(get_settings(), "gemini_api_timeout_seconds", 5.0),
            )

            if ai_text is None or not ai_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI provider timed out or returned empty response."
                )

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
    except Exception as exc:
        logger.warning(f"AI unavailable for farmer {farmer.id}: {exc}. Deferring fallback until after specialized enrichment.")
        ai_response_text = ""

    # Enrich with nearby shop inventory recommendations if products match
    try:
        from src.shops.service import enrich_response_with_shops
        logger.info("[PROCESS MSG] Invoking enrich_response_with_shops()")
        ai_response_text = await enrich_response_with_shops(
            db, conversation.user_message or "", ai_response_text, farmer
        )
    except Exception as err:
        logger.warning(f"Failed to enrich response with shops: {err}")

    # Enrich with mandi/market price data if farmer asks about prices
    try:
        from src.market.service import enrich_response_with_market_prices
        logger.info("[PROCESS MSG] Invoking enrich_response_with_market_prices()")
        ai_response_text = await enrich_response_with_market_prices(
            db, conversation.user_message or "", ai_response_text, farmer
        )
    except Exception as mkt_err:
        logger.warning(f"Failed to enrich response with market prices: {mkt_err}")

    # Enrich with weather forecast data if farmer asks about weather
    try:
        from src.weather.service import enrich_response_with_weather
        logger.info("[PROCESS MSG] Invoking enrich_response_with_weather()")
        ai_response_text = await enrich_response_with_weather(
            db, conversation.user_message or "", ai_response_text, farmer
        )
    except Exception as weather_err:
        logger.warning(f"Failed to enrich response with weather: {weather_err}")

    # Enrich with government scheme information if farmer asks about schemes/subsidies
    try:
        from src.schemes.service import enrich_response_with_schemes
        logger.info("[PROCESS MSG] Invoking enrich_response_with_schemes()")
        ai_response_text = await enrich_response_with_schemes(
            db, conversation.user_message or "", ai_response_text, farmer
        )
    except Exception as scheme_err:
        logger.warning(f"Failed to enrich response with schemes: {scheme_err}")

    # Enrich with expert escalation ticket if farmer requests human/specialist assistance
    try:
        from src.escalation.service import enrich_response_with_escalation
        logger.info("[PROCESS MSG] Invoking enrich_response_with_escalation()")
        ai_response_text = await enrich_response_with_escalation(
            db, conversation.user_message or "", ai_response_text, farmer
        )
    except Exception as esc_err:
        logger.warning(f"Failed to enrich response with escalation: {esc_err}")

    # Format and optimize multi-intent response if multiple sections were enriched
    try:
        from src.ai.formatting import format_multi_intent_response
        language = getattr(farmer, "preferred_language", "en") or "en"
        if any(ord(c) > 127 for c in (conversation.user_message or "")):
            language = "te"
        ai_response_text = format_multi_intent_response(
            assembled_text=ai_response_text,
            user_message=conversation.user_message or "",
            language=language,
        )
    except Exception as fmt_err:
        logger.warning(f"Multi-intent formatting warning: {fmt_err}")

    ai_response_text = ai_response_text.strip() if ai_response_text else ""
    if not ai_response_text:
        pref_lang = getattr(farmer, "preferred_language", "en") or "en"
        ai_response_text = get_fallback_response(pref_lang)

    logger.info(f"[PROCESS MSG FINAL] Length: {len(ai_response_text)} | Final response immediately before DB save: '{ai_response_text}'")

    conversation.ai_response = ai_response_text
    db.add(conversation)
    await db.commit()

    return ai_response_text


def _finalize_whatsapp_response(response_text: str, max_chars: int = 1600) -> str:
    """
    Cleans, deduplicates, and optimizes the final outgoing WhatsApp message.

    Guarantees:
    - Eliminates redundant blank lines
    - Prevents multi-block bloat while strictly preserving Expert Escalation,
      Market Prices, Weather, and core agronomic advice
    - Never partially truncates safety, escalation, or structured blocks mid-sentence
    """
    import re
    if not response_text:
        return ""

    text = re.sub(r'\n{3,}', '\n\n', response_text).strip()

    # If text is within comfortable limit, return directly
    if len(text) <= max_chars:
        return text

    # If response is overly long due to multiple stacked blocks, prioritize whole blocks:
    # Priority order:
    # 0. Expert Escalation (👨‍🌾 / 🚨 / 🆘) - Safety Critical (Always included)
    # 1. Market Prices (📊)
    # 2. Weather (🌤️)
    # 3. Shops (🏬)
    # 4. Schemes (📜)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    core_blocks = []
    enrichment_blocks = []

    for b in blocks:
        if any(marker in b for marker in ["👨‍🌾", "🚨", "🆘", "📊", "🌤️", "🏬", "📜"]):
            enrichment_blocks.append(b)
        else:
            core_blocks.append(b)

    def _block_priority(block: str) -> int:
        if any(m in block for m in ["👨‍🌾", "🚨", "🆘"]):
            return 0
        if "📊" in block:
            return 1
        if "🌤️" in block:
            return 2
        if "🏬" in block:
            return 3
        if "📜" in block:
            return 4
        return 5

    enrichment_blocks.sort(key=_block_priority)

    selected_blocks = list(core_blocks)
    current_len = sum(len(b) + 2 for b in selected_blocks)

    for eb in enrichment_blocks:
        is_escalation = any(m in eb for m in ["👨‍🌾", "🚨", "🆘"])
        # Always include escalation, or include block if within budget or top 2 enrichments
        if is_escalation or (current_len + len(eb) + 2 <= max_chars) or len(selected_blocks) < 3:
            selected_blocks.append(eb)
            current_len += len(eb) + 2

    return "\n\n".join(selected_blocks).strip()


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

    # Add a vision-specific system prompt instruction enforcing JSON and diagnostic safety
    full_system_prompt = (
        f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n"
        "The user has uploaded an image of their crop. Diagnose any visible diseases, pests, or deficiencies.\n"
        "IMAGE DIAGNOSIS SAFETY RULES:\n"
        "- Never claim that an image proves a disease with absolute certainty. Use cautious wording like 'appears consistent with', 'may indicate', or 'possible symptoms of'.\n"
        "- State that visual symptoms alone cannot be 100% confirmed from a single photo and ask the farmer to check front/back of leaf, close-up, or whole plant if uncertain.\n"
        "- Do not recommend unverified chemical pesticides or dosages unless grounded in trusted knowledge. Mention standard cultural practices and advise checking with a local Agriculture Extension Officer (AEO).\n"
        "You MUST return a strictly valid JSON object matching this exact schema:\n"
        '{"disease_name": "Name", "confidence_score": 0.85, "severity": "low/medium/high", "symptoms": "Visible symptoms", "treatment_recommendation": "Cautious agronomic steps", "friendly_whatsapp_reply": "Natural language reply for the farmer"}\n'
        "Provide actionable agronomic advice.\n\n"
        f"{farmer_context}\n\n{memory_context}"
    )

    
    # 2. History
    history_records = await repo.get_conversation_history(farmer.id)
    history_records.reverse()
    history = []
    for record in history_records:
        if record.user_message:
            history.append({"role": "user", "parts": record.user_message})
        if record.ai_response:
            # Clean structured enrichment sections from history to prevent LLM contamination
            clean_response = (
                record.ai_response
                .split("Available Nearby Shops:")[0]
                .split("🏬")[0]
                .split("📊")[0]
                .split("🌡️")[0]
                .split("🏛️")[0]
                .split("🎫")[0]
                .strip()
            )
            if clean_response:
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

        # Check if escalation is required for unknown disease or explicit farmer request
        try:
            from src.escalation.service import enrich_response_with_escalation, _detect_escalation_intent
            caption_lower = user_caption.lower() if user_caption else ""
            has_caption_intent, _ = _detect_escalation_intent(caption_lower, user_caption or "")
            is_unidentified = not diagnosis_data.disease_name or diagnosis_data.disease_name.lower() in ("unknown", "unidentified", "none")

            if has_caption_intent or is_unidentified:
                reply_text = await enrich_response_with_escalation(
                    db,
                    user_caption or "Image Crop Diagnosis",
                    reply_text,
                    farmer,
                    force_escalation=is_unidentified,
                    force_reason="inspection" if is_unidentified else None,
                )
        except Exception as esc_err:
            logger.warning(f"Failed to enrich image response with escalation: {esc_err}")

    except Exception as e:
        logger.warning(f"AI Vision unavailable or failed to parse for farmer {farmer.id}: {e}")
        reply_text = get_fallback_response(farmer.preferred_language)
        
    conversation.ai_response = reply_text
    db.add(conversation)
    await db.commit()
    
    return reply_text

