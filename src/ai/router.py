from fastapi import APIRouter, Depends, status, HTTPException
from src.ai.schemas import AIGenerateRequest, AIGenerateResponse, AIHealthResponse
from src.ai.service import AIService
from src.ai.dependencies import get_ai_service
from src.config import get_settings

router = APIRouter(prefix="/ai", tags=["AI"])

@router.post("/generate", response_model=AIGenerateResponse, status_code=status.HTTP_200_OK,
    summary="Generate AI response", description="Generate an AI response for a farmer based on their context and message.")
async def generate_ai_response(
    request: AIGenerateRequest, 
    service: AIService = Depends(get_ai_service)
):
    return await service.generate_ai_response(request)

@router.get("/health", response_model=AIHealthResponse, status_code=status.HTTP_200_OK,
    summary="AI Provider Health Check", description="Check the availability of the configured LLM provider.")
async def check_ai_health():
    settings = get_settings()
    # If the API key is not configured, the service is unavailable
    if not settings.google_gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI Provider is not configured."
        )
    
    return AIHealthResponse(
        status="healthy",
        active_provider="gemini"
    )
