from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import random
from fastapi import HTTPException, status
from src.core.logging import logger
from src.core.models import Expert
from src.escalation.repository import EscalationRepository
from src.escalation.schemas import (
    ExpertCreate,
    ExpertUpdate,
    ExpertResponse,
    PaginatedExpertResponse,
    EscalationTicketResponse,
    FarmerEscalationHistoryResponse,
)


# ---------------------------------------------------------------------------
# Intent Detection Keywords
# ---------------------------------------------------------------------------

_ESCALATION_KEYWORDS_EN = {
    "expert", "specialist", "agronomist", "officer", "aeo",
    "human agent", "agent", "call me back", "callback", "talk to",
    "connect me", "escalate", "escalation", "human support",
    "contact expert", "agriculture officer", "krishi officer",
    "speak to", "speak with", "talk with",
}

_ESCALATION_KEYWORDS_TE = {
    "వ్యవసాయ అధికారి", "నిపుణుడు", "నిపుణులు", "అధికారి", "సంప్రదించాలి",
    "మాట్లాడాలి", "కాల్ చేయండి", "హెల్ప్‌లైన్", "సహాయం కావాలి", "ఏఈవో",
    "ఆఫీసర్", "కాల్ బ్యాక్", "అధికారితో", "నిపుణుడితో",
}

# ---------------------------------------------------------------------------
# Multilingual Formatting Labels
# ---------------------------------------------------------------------------

_EN_LABELS = {
    "header":          "👨‍🌾 Krishi Officer Escalation Ticket: #{ticket_id}",
    "status":          "✅ Status",
    "status_assigned": "Assigned to District Agriculture Officer",
    "status_pending":  "Queued for Next Available Officer",
    "specialist":      "👤 Specialist",
    "region":          "📍 Region",
    "contact":         "📞 Officer Contact",
    "callback":        "⏱️ Expected Callback Window",
    "callback_time":   "Within 30–60 minutes",
    "helpline_title":  "🚨 Government Kisan Call Centre (Toll-Free)",
    "helpline_number": "📞 1800-180-1551 (6:00 AM - 10:00 PM, Daily)",
    "existing_ticket": "ℹ️ You already have an active escalation ticket (#{ticket_id}). Our officer is reviewing your case.",
    "no_local_expert": "ℹ️ No local officer is currently on duty. Your ticket has been logged and our team will connect with you.",
}

_TE_LABELS = {
    "header":          "👨‍🌾 వ్యవసాయ అధికారి సంప్రదింపు టికెట్: #{ticket_id}",
    "status":          "✅ స్థితి",
    "status_assigned": "జిల్లా వ్యవసాయ అధికారికి కేటాయించబడింది",
    "status_pending":  "అధికారి కేటాయింపు కోసం వేచి ఉంది",
    "specialist":      "👤 నిపుణుడు",
    "region":          "📍 ప్రాంతం",
    "contact":         "📞 అధికారి ఫోన్",
    "callback":        "⏱️ కాల్ బ్యాక్ సమయం",
    "callback_time":   "30–60 నిమిషాలలోపు",
    "helpline_title":  "🚨 జాతీయ కిసాన్ కాల్ సెంటర్ (ఉచిత నంబర్)",
    "helpline_number": "📞 1800-180-1551 (ఉదయం 6:00 - రాత్రి 10:00)",
    "existing_ticket": "ℹ️ మీకు ఇప్పటికే ఒక యాక్టివ్ సంప్రదింపు టికెట్ (#{ticket_id}) ఉంది. మా అధికారి మీ సమస్యను పరిశీలిస్తున్నారు.",
    "no_local_expert": "ℹ️ ప్రస్తుతం స్థానిక అధికారి అందుబాటులో లేరు. మీ టికెట్ నమోదు చేయబడింది, మా బృందం మిమ్మల్ని సంప్రదిస్తుంది.",
}


def _detect_escalation_intent(query_lower: str, query_text: str) -> bool:
    """Detect if query has human / expert escalation intent in English or Telugu."""
    if any(kw in query_lower for kw in _ESCALATION_KEYWORDS_EN):
        return True
    if any(kw in query_text for kw in _ESCALATION_KEYWORDS_TE):
        return True
    return False


def _detect_specialty_hint(query_text: str) -> Optional[str]:
    """Detect agricultural specialty keyword from farmer inquiry."""
    q = query_text.lower()
    if any(w in q for w in ["pest", "insect", "disease", "fungus", "పురుగు", "తెగులు", "మందు"]):
        return "Pest"
    if any(w in q for w in ["soil", "fertilizer", "urea", "dap", "భూమి", "నేల", "ఎరువు"]):
        return "Soil"
    if any(w in q for w in ["cotton", "పత్తి"]):
        return "Cotton"
    if any(w in q for w in ["water", "irrigation", "drip", "నీరు", "సాగునీరు"]):
        return "Irrigation"
    return None


def _generate_ticket_id() -> str:
    """Generate a clean, professional escalation ticket ID."""
    date_str = datetime.utcnow().strftime("%Y%m%d")
    rand_suffix = random.randint(1000, 9999)
    return f"ESC-{date_str}-{rand_suffix}"


def _find_recent_pending_ticket(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Check if farmer already has an unresolved ticket raised today."""
    today_prefix = datetime.utcnow().strftime("%Y%m%d")
    for t in reversed(history):
        if t.get("status") in ("Pending", "Assigned"):
            ticket_id = t.get("ticket_id", "")
            if today_prefix in ticket_id:
                return t
    return None


class EscalationService:
    def __init__(self, repository: EscalationRepository):
        self.repository = repository

    async def list_experts(self, specialty: Optional[str] = None) -> List[ExpertResponse]:
        experts = await self.repository.get_active_experts(specialty)
        return [ExpertResponse.model_validate(e) for e in experts]

    async def get_expert(self, expert_id: UUID) -> ExpertResponse:
        expert = await self.repository.get_by_id(expert_id)
        if not expert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expert with ID '{expert_id}' not found.",
            )
        return ExpertResponse.model_validate(expert)

    async def create_expert(self, data: ExpertCreate) -> ExpertResponse:
        expert = await self.repository.create(data)
        logger.info(f"Registered new expert '{expert.name}' ({expert.id})")
        return ExpertResponse.model_validate(expert)

    async def update_expert(self, expert_id: UUID, data: ExpertUpdate) -> ExpertResponse:
        expert = await self.repository.update(expert_id, data)
        if not expert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expert with ID '{expert_id}' not found.",
            )
        logger.info(f"Updated expert '{expert.name}' ({expert_id})")
        return ExpertResponse.model_validate(expert)

    async def get_farmer_escalations(self, farmer_id: UUID) -> FarmerEscalationHistoryResponse:
        history = await self.repository.get_farmer_consultation_history(farmer_id)
        return FarmerEscalationHistoryResponse(
            farmer_id=farmer_id,
            total_tickets=len(history),
            tickets=history,
        )


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

async def enrich_response_with_escalation(
    db,
    query_text: str,
    ai_response: str,
    farmer=None,
) -> str:
    """
    Detect explicit human/expert escalation intent in the farmer's query.
    If detected, creates or retrieves an escalation ticket, assigns an active specialist,
    and appends a formatted ticket card with official helpline numbers.

    Always returns original ai_response unchanged if:
    - No escalation intent is detected
    - Any exception occurs
    """
    from src.escalation.repository import EscalationRepository

    query_lower = query_text.lower()
    logger.info(f"[ENRICH ESCALATION] Checking query: '{query_text}' | ai_response length: {len(ai_response)}")

    # Step 1: Detect intent (English or Telugu)
    has_intent = _detect_escalation_intent(query_lower, query_text)
    if not has_intent:
        return ai_response

    language = getattr(farmer, "preferred_language", "en") or "en"
    labels = _TE_LABELS if language == "te" else _EN_LABELS
    farmer_id = getattr(farmer, "id", None)

    try:
        repo = EscalationRepository(db)
        await repo.seed_default_experts_if_empty()

        # Step 2: Check for duplicate pending ticket raised today
        history = []
        if farmer_id:
            history = await repo.get_farmer_consultation_history(farmer_id)
        
        existing_ticket = _find_recent_pending_ticket(history)
        if existing_ticket:
            ticket_id = existing_ticket.get("ticket_id", "ESC-ACTIVE")
            header = labels["header"].format(ticket_id=ticket_id)
            existing_note = labels["existing_ticket"].format(ticket_id=ticket_id)
            full_block = "\n".join([
                header,
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"{labels['status']}: {existing_ticket.get('status', 'Assigned')}",
                f"{labels['specialist']}: {existing_ticket.get('expert_name', 'District Agriculture Officer')}",
                f"{labels['contact']}: {existing_ticket.get('expert_phone', 'Official Helpline')}",
                "",
                existing_note,
                "",
                labels["helpline_title"],
                labels["helpline_number"],
            ])
            logger.info(f"[ENRICH ESCALATION] Returning existing active ticket {ticket_id}")
            return ai_response + "\n\n" + full_block

        # Step 3: Match best active expert
        specialty_hint = _detect_specialty_hint(query_text)
        experts = await repo.get_active_experts(specialty=specialty_hint)
        if not experts:
            # Fallback to any active expert
            experts = await repo.get_active_experts()

        assigned_expert: Optional[Expert] = experts[0] if experts else None

        # Step 4: Resolve Region Context from FarmerProfile/Memory
        region_str = ""
        try:
            from sqlalchemy import select
            from src.core.models import FarmerProfile
            if farmer_id:
                prof_res = await db.execute(select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id))
                prof = prof_res.scalar_one_or_none()
                if prof and prof.district:
                    region_str = f"{prof.district}" + (f", {prof.state}" if prof.state else "")
        except Exception as reg_err:
            logger.warning(f"[ENRICH ESCALATION] Could not resolve region: {reg_err}")

        # Step 5: Build Ticket Record
        new_ticket_id = _generate_ticket_id()
        ticket_data = {
            "ticket_id": new_ticket_id,
            "status": "Assigned" if assigned_expert else "Pending",
            "topic": query_text[:120],
            "expert_id": str(assigned_expert.id) if assigned_expert else None,
            "expert_name": assigned_expert.name if assigned_expert else None,
            "expert_specialty": assigned_expert.specialty if assigned_expert else None,
            "expert_phone": assigned_expert.phone_number if assigned_expert else None,
            "region": region_str or "District Agriculture Office",
            "created_at": datetime.utcnow().isoformat(),
        }

        if farmer_id:
            await repo.record_escalation_ticket(farmer_id, ticket_data)

        # Step 6: Format Outbound Confirmation Block
        header = labels["header"].format(ticket_id=new_ticket_id)
        status_text = labels["status_assigned"] if assigned_expert else labels["status_pending"]

        lines = [
            header,
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"{labels['status']}: {status_text}",
        ]

        if assigned_expert:
            lines.append(f"{labels['specialist']}: {assigned_expert.name} ({assigned_expert.specialty})")
            if region_str:
                lines.append(f"{labels['region']}: {region_str}")
            lines.append(f"{labels['contact']}: {assigned_expert.phone_number}")
            lines.append(f"{labels['callback']}: {labels['callback_time']}")
        else:
            lines.append(labels["no_local_expert"])

        lines.append("")
        lines.append(labels["helpline_title"])
        lines.append(labels["helpline_number"])

        full_block = "\n".join(lines)
        logger.info(f"[ENRICH ESCALATION] Appended escalation ticket {new_ticket_id} (Expert: {assigned_expert.name if assigned_expert else 'None'}).")
        return ai_response + "\n\n" + full_block

    except Exception as esc_err:
        logger.warning(f"[ENRICH ESCALATION] Escalation enrichment failed: {esc_err}")
        return ai_response
