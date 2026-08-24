from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from src.escalation.schemas import (
    ExpertCreate,
    ExpertUpdate,
    ExpertResponse,
    FarmerEscalationHistoryResponse,
    TicketStatusUpdate,
    TicketQueueItem,
    TicketQueueResponse,
)
from src.escalation.service import EscalationService
from src.escalation.dependencies import get_escalation_service
from src.auth.constants import UserRole
from src.auth.dependencies import require_admin, require_expert, require_roles, verify_expert_access
from src.core.models import UserAccount

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
    current_user: UserAccount = Depends(require_admin),
    service: EscalationService = Depends(get_escalation_service),
):
    """Register a new agricultural expert (Admin only)."""
    return await service.create_expert(payload)


@router.put("/experts/{expert_id}", response_model=ExpertResponse)
async def update_expert(
    expert_id: UUID,
    payload: ExpertUpdate,
    current_user: UserAccount = Depends(require_expert),
    service: EscalationService = Depends(get_escalation_service),
):
    """Update expert details (Expert owner or Admin)."""
    verify_expert_access(current_user, expert_id)
    return await service.update_expert(expert_id, payload)


@router.get("/tickets", response_model=TicketQueueResponse)
async def list_escalation_tickets(
    status: Optional[str] = Query(None, description="Filter tickets by status: Pending, Assigned, In Progress, Resolved"),
    current_user: UserAccount = Depends(require_expert),
    service: EscalationService = Depends(get_escalation_service),
):
    """Fetch all active escalation tickets in queue (Admin / Expert only)."""
    return await service.list_tickets(status_filter=status, current_user=current_user)


@router.patch("/tickets/{ticket_id}/status", response_model=TicketQueueItem)
async def update_ticket_status(
    ticket_id: str,
    payload: TicketStatusUpdate,
    current_user: UserAccount = Depends(require_expert),
    service: EscalationService = Depends(get_escalation_service),
):
    """Update status and resolution notes of an escalation ticket (Admin / Assigned Expert)."""
    return await service.update_ticket_status(ticket_id=ticket_id, payload=payload, current_user=current_user)


@router.get("/tickets/farmer/{farmer_id}", response_model=FarmerEscalationHistoryResponse)
async def get_farmer_escalation_tickets(
    farmer_id: UUID,
    current_user: UserAccount = Depends(require_expert),
    service: EscalationService = Depends(get_escalation_service),
):
    """Fetch past escalation tickets and consultation history for a farmer (Admin / Expert only)."""
    return await service.get_farmer_escalations(farmer_id)


