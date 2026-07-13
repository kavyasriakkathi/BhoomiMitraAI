from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Farmer, Conversation
from src.core.logging import logger
from src.ai.repository import AIRepository
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse
from src.ai.prompts import (
    BHOOMIMITRA_SYSTEM_PROMPT,
    build_farmer_context,
    get_fallback_response,
)
from src.ai.gemini_client import generate_response

class AIService:
    def __init__(self, repository: AIRepository):
        self.repository = repository

    async def generate_ai_response(self, request: AIGenerateRequest) -> AIGenerateResponse:
        try:
            # 1. Fetch farmer profile for context
            profile = await self.repository.get_farmer_profile(request.farmer_id)

            # 2. Build farmer context string
            farmer_context = build_farmer_context(
                crop=profile.current_crop if profile else None,
                district=profile.district if profile else None,
                state=profile.state if profile else None,
                land_size=profile.land_size_acres if profile else None,
            )

            # 3. Build system prompt
            full_system_prompt = f"{BHOOMIMITRA_SYSTEM_PROMPT}\n\n{farmer_context}"

            # 4. Fetch conversation history
            history_records = await self.repository.get_conversation_history(request.farmer_id)
            
            # Gemini expects oldest first
            history_records.reverse()
            history = []
            for record in history_records:
                if record.user_message:
                    history.append({"role": "user", "parts": record.user_message})
                if record.ai_response:
                    history.append({"role": "model", "parts": record.ai_response})

            # 5. Call Gemini API
            logger.info(f"Processing AI request for farmer {request.farmer_id}: '{request.message[:80]}...'")
            
            ai_text = await generate_response(
                system_prompt=full_system_prompt,
                conversation_history=history,
                user_message=request.message,
            )

            if ai_text is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI provider timed out or returned no response."
                )

            # 6. Return structured response
            return AIGenerateResponse(
                response_text=ai_text,
                intent=None,
                confidence=None,
                provider_used="gemini"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error generating AI response: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while communicating with the AI provider."
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
    
    request = AIGenerateRequest(
        farmer_id=farmer.id,
        conversation_id=conversation.id,
        message=conversation.user_message or ""
    )
    
    try:
        response = await service.generate_ai_response(request)
        ai_response_text = response.response_text
    except HTTPException:
        logger.warning(f"AI unavailable for farmer {farmer.id}. Using fallback.")
        ai_response_text = get_fallback_response(farmer.preferred_language)
        
    conversation.ai_response = ai_response_text
    db.add(conversation)
    await db.commit()
    
    return ai_response_text
