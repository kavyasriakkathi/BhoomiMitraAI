from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from src.memory.schemas import (
    FarmerMemoryResponse,
    FarmerMemoryUpdate,
    FarmerMemoryRefreshRequest,
    FarmerMemorySummaryResponse,
    VoiceSettingsResponse,
)
from src.memory.service import FarmerMemoryService
from src.memory.dependencies import get_memory_service
from src.auth.dependencies import require_expert
from src.core.models import UserAccount
from src.core.logging import logger

router = APIRouter()

@router.get("/{farmer_id}", response_model=FarmerMemoryResponse, summary="Get Farmer Long-Term Memory Profile")
async def get_farmer_memory(
    farmer_id: UUID,
    service: FarmerMemoryService = Depends(get_memory_service)
):
    """
    Retrieve long-term memory profile for a given farmer.
    Initializes default record if non-existent.
    """
    return await service.get_memory_response(farmer_id)


@router.put("/{farmer_id}", response_model=FarmerMemoryResponse, summary="Update Farmer Memory Profile (Admin / Expert only)")
async def update_farmer_memory(
    farmer_id: UUID,
    data: FarmerMemoryUpdate,
    current_user: UserAccount = Depends(require_expert),
    service: FarmerMemoryService = Depends(get_memory_service)
):
    """
    Update farmer long-term memory profile (Admin / Expert).
    """
    return await service.update_memory(farmer_id, data)


@router.post("/refresh", response_model=FarmerMemoryResponse, summary="Refresh & Sync Farmer Memory Profile")
async def refresh_farmer_memory(
    request: FarmerMemoryRefreshRequest,
    service: FarmerMemoryService = Depends(get_memory_service)
):
    """
    Synchronize farmer memory profile with existing entities (FarmerProfile, Farm, CropHealth, OrderRequest, SchemeApplication).
    """
    return await service.refresh_farmer_memory(request.farmer_id)


@router.get("/summary/{farmer_id}", response_model=FarmerMemorySummaryResponse, summary="Get Farmer Conversation Memory Summary")
async def get_farmer_memory_summary(
    farmer_id: UUID,
    service: FarmerMemoryService = Depends(get_memory_service)
):
    """
    Generate or retrieve conversation history summary for a farmer.
    """
    memory = await service.get_memory(farmer_id)
    summary_text = memory.conversation_summary
    if not summary_text:
        summary_text = await service.summarize_conversations(farmer_id)
        memory = await service.get_memory(farmer_id)

    return FarmerMemorySummaryResponse(
        farmer_id=farmer_id,
        summary=summary_text,
        last_updated=memory.last_updated,
        primary_crops=memory.primary_crops or [],
        district=memory.district,
        risk_factors=memory.risk_factors or []
    )


@router.get("/voice/{farmer_id}", response_model=VoiceSettingsResponse, summary="Get Farmer Voice Personalization Settings")
async def get_farmer_voice_settings(
    farmer_id: UUID,
    service: FarmerMemoryService = Depends(get_memory_service)
):
    """
    Retrieve preferred language, voice, speed, and gender for STT/TTS audio interactions.
    """
    return await service.get_voice_settings(farmer_id)
