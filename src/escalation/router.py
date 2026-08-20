from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.escalation.schemas import (
    ExpertCreate,
    ExpertUpdate,
    ExpertResponse,
    FarmerEscalationHistoryResponse,
)
from src.escalation.service import EscalationService
from src.escalation.dependencies import get_escalation_service

router = APIRouter()


@router.get("/experts", response_model=List[ExpertResponse])
async def list_experts(
    specialty: Optional[str] = Query(None, description="Filter experts by specialty keyword"),
    service: EscalationService = Depends(get_escalation_service),
):
    """List all registered and active agricultural experts."""
    return await service.list_experts(specialty=specialty)


@router.get("/experts/{expert_id}", response_model=ExpertResponse)
async def get_expert(
    expert_id: UUID,
    service: EscalationService = Depends(get_escalation_service),
):
    """Get expert details by ID."""
    return await service.get_expert(expert_id)


@router.post("/experts", response_model=ExpertResponse, status_code=status.HTTP_201_CREATED)
async def create_expert(
    payload: ExpertCreate,
    service: EscalationService = Depends(get_escalation_service),
):
    """Register a new agricultural expert."""
    return await service.create_expert(payload)


@router.put("/experts/{expert_id}", response_model=ExpertResponse)
async def update_expert(
    expert_id: UUID,
    payload: ExpertUpdate,
    service: EscalationService = Depends(get_escalation_service),
):
    """Update expert details."""
    return await service.update_expert(expert_id, payload)


@router.get("/tickets/farmer/{farmer_id}", response_model=FarmerEscalationHistoryResponse)
async def get_farmer_escalation_tickets(
    farmer_id: UUID,
    service: EscalationService = Depends(get_escalation_service),
):
    """Fetch past escalation tickets and consultation history for a farmer."""
    return await service.get_farmer_escalations(farmer_id)
