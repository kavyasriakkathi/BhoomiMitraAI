from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime
import random
from fastapi import HTTPException, status
from src.core.logging import logger
from src.core.models import Expert, UserAccount
from src.escalation.repository import EscalationRepository
from src.escalation.schemas import (
    ExpertCreate,
    ExpertUpdate,
    ExpertResponse,
    PaginatedExpertResponse,
    EscalationTicketResponse,
    FarmerEscalationHistoryResponse,
    TicketStatusUpdate,
    TicketQueueItem,
    TicketQueueResponse,
)


# ---------------------------------------------------------------------------
# Conservative Intent & Hazard Detection Keywords
# ---------------------------------------------------------------------------

# 1. Explicit request for human agricultural officer / extension officer
_EXPLICIT_ESCALATION_EN = {
    "expert", "specialist", "agronomist", "officer", "aeo",
    "human agent", "agent", "call me back", "callback", "talk to",
    "connect me", "escalate", "escalation", "human support",
    "contact expert", "agriculture officer", "krishi officer",
    "speak to", "speak with", "talk with",
}

_EXPLICIT_ESCALATION_TE = {
    "వ్యవసాయ అధికారి", "నిపుణుడు", "నిపుణులు", "అధికారి", "సంప్రదించాలి",
    "మాట్లాడాలి", "కాల్ చేయండి", "హెల్ప్‌లైన్", "సహాయం కావాలి", "ఏఈవో",
    "ఆఫీసర్", "కాల్ బ్యాక్", "అధికారితో", "నిపుణుడితో",
}

# 2. Banned, highly hazardous chemicals & acute poisoning safety triggers
_HAZARDOUS_CHEMICALS_EN = {
    "paraquat", "endosulfan", "monocrotophos", "phorate", "methyl parathion",
    "pesticide poison", "swallowed pesticide", "ingested pesticide",
    "pesticide poisoning", "chemical poisoning", "drank pesticide",
}

_HAZARDOUS_CHEMICALS_TE = {
    "విషం", "పురుగుల మందు తాగడం", "విషపూరితం", "ఎండోసల్ఫాన్", "మోనోక్రోటోఫాస్", "పారాక్వాట్",
}

# 3. Physical on-field inspection and sudden catastrophic crop mortality
_PHYSICAL_INSPECTION_EN = {
    "field inspection", "visit my field", "farm visit", "inspect my farm",
    "crop dying completely", "mysterious rot", "total crop failure",
    "all plants dying suddenly", "on-site inspection",
}

_PHYSICAL_INSPECTION_TE = {
    "పొలం పరిశీలన", "తోట పరిశీలన", "పొలం చూడండి", "మొక్కలు అన్నీ ఎండిపోతున్నాయి",
    "పంట పూర్తిగా చనిపోతుంది", "క్షేత్ర పరిశీలన",
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
    "hazard_warning":  "⚠️ **URGENT SAFETY CAUTION**: The query involves hazardous/banned chemicals or immediate toxicity. Please do not handle unsafe substances without protective gear. Contact medical or agriculture authorities immediately.",
    "inspection_note": "🌾 **Field Inspection Request Logged**: An agricultural officer has been notified for regional on-field assessment.",
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
    "hazard_warning":  "⚠️ **ముఖ్యమైన భద్రతా హెచ్చరిక**: ఇది ప్రమాదకరమైన/నిషేధిత రసాయనాలకు సంబంధించినది. సురక్షితమైన జాగ్రత్తలు పాటించండి మరియు వెంటనే అధికారులను సంప్రదించండి.",
    "inspection_note": "🌾 **క్షేత్ర పరిశీలన అభ్యర్థన నమోదు చేయబడింది**: మీ పంట పరిశీలన కోసం వ్యవసాయ అధికారికి సమాచారం అందించబడింది.",
}


def _detect_escalation_intent(query_lower: str, query_text: str) -> Tuple[bool, Optional[str]]:
    """
    Conservatively detect if query warrants escalation.
    Returns (should_escalate, trigger_reason).
    Trigger reasons: 'explicit', 'hazard', 'inspection'
    """
    # 1. Hazardous chemicals / poisoning (Highest priority safety trigger)
    if any(kw in query_lower for kw in _HAZARDOUS_CHEMICALS_EN) or any(kw in query_text for kw in _HAZARDOUS_CHEMICALS_TE):
        return True, "hazard"

    # 2. Physical inspection / catastrophic crop death
    if any(kw in query_lower for kw in _PHYSICAL_INSPECTION_EN) or any(kw in query_text for kw in _PHYSICAL_INSPECTION_TE):
        return True, "inspection"

    # 3. Explicit officer / expert request
    if any(kw in query_lower for kw in _EXPLICIT_ESCALATION_EN) or any(kw in query_text for kw in _EXPLICIT_ESCALATION_TE):
        return True, "explicit"

    return False, None


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
        if t.get("status") in ("Pending", "Assigned", "In Progress"):
            ticket_id = t.get("ticket_id", "")
            if today_prefix in ticket_id:
                return t
_DEMO_EXPERT_PHONES = {
    "+91 9848012345", "+91 9876543211", "+91 9701234568", "+91 9988776656",
    "+91-98490-11223", "+91 9849011223", "9848012345", "9876543211", "9701234568", "9988776656"
}
_DEMO_EXPERT_NAMES = {
    "Dr. K. Srinivas Rao", "Dr. Ananya Sharma", "Sri V. Mallikarjun", "Dr. P. Venkateswarlu"
}


def is_verified_expert(expert: Optional[Any], app_env: str = "development") -> bool:
    """
    Determines if an expert is a verified active contact safe to display to farmers.
    In production mode, demo seed expert phone numbers are suppressed to prevent
    exposing dummy contact data to real farmers.
    """
    if not expert or not getattr(expert, "phone_number", None):
        return False

    import re
    phone_clean = re.sub(r'[\s\-+]', '', expert.phone_number)
    name_clean = (getattr(expert, "name", "") or "").strip()

    if name_clean in _DEMO_EXPERT_NAMES and app_env == "production":
        return False

    for dp in _DEMO_EXPERT_PHONES:
        dp_clean = re.sub(r'[\s\-+]', '', dp)
        if dp_clean and dp_clean in phone_clean and app_env == "production":
            return False

    return True


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

    async def list_tickets(
        self,
        status_filter: Optional[str] = None,
        current_user: Optional[UserAccount] = None,
    ) -> TicketQueueResponse:
        """List tickets across all farmers for the expert/admin dashboard."""
        expert_id = None
        if current_user and current_user.role == "expert" and current_user.expert_id:
            expert_id = current_user.expert_id

        raw_tickets = await self.repository.get_all_tickets(
            status_filter=status_filter,
            expert_id=expert_id,
        )

        items = [TicketQueueItem(**t) for t in raw_tickets]
        total = len(items)
        pending = sum(1 for t in items if t.status.lower() == "pending")
        assigned = sum(1 for t in items if t.status.lower() in ("assigned", "in progress"))
        resolved = sum(1 for t in items if t.status.lower() == "resolved")

        return TicketQueueResponse(
            total=total,
            pending=pending,
            assigned=assigned,
            resolved=resolved,
            items=items,
        )

    async def update_ticket_status(
        self,
        ticket_id: str,
        payload: TicketStatusUpdate,
        current_user: Optional[UserAccount] = None,
    ) -> TicketQueueItem:
        """Update status and resolution notes of an escalation ticket."""
        found = await self.repository.get_ticket_by_id(ticket_id)
        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Escalation ticket '{ticket_id}' not found.",
            )

        _, target_ticket = found

        # Enforce RBAC: If expert user, verify assigned or unassigned
        if current_user and current_user.role == "expert":
            assigned_exp = target_ticket.get("expert_id")
            if assigned_exp and str(current_user.expert_id) != str(assigned_exp):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access forbidden: You cannot modify tickets assigned to another expert.",
                )

        updated = await self.repository.update_ticket_status(
            ticket_id=ticket_id,
            new_status=payload.status,
            notes=payload.notes,
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update ticket status.",
            )

        return TicketQueueItem(**updated)


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

async def enrich_response_with_escalation(
    db,
    query_text: str,
    ai_response: str,
    farmer=None,
    force_escalation: bool = False,
    force_reason: Optional[str] = None,
) -> str:
    """
    Conservatively detect if query warrants escalation.
    If detected (or forced by vision diagnosis), creates or retrieves an escalation ticket,
    assigns an active specialist from the database, and appends a formatted ticket card
    with the verified official Kisan Call Centre 1800-180-1551 fallback.
    """
    from src.escalation.repository import EscalationRepository

    query_lower = query_text.lower()
    logger.info(f"[ENRICH ESCALATION] Checking query: '{query_text}' | ai_response length: {len(ai_response)}")

    # Step 1: Detect conservative intent (English or Telugu)
    has_intent, trigger_reason = _detect_escalation_intent(query_lower, query_text)
    if not has_intent and not force_escalation:
        return ai_response

    reason = force_reason or trigger_reason or "explicit"
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
        topic_suffix = f" [{reason.upper()}]" if reason != "explicit" else ""
        ticket_data = {
            "ticket_id": new_ticket_id,
            "status": "Assigned" if assigned_expert else "Pending",
            "topic": (query_text[:100] + topic_suffix) if query_text else f"Crop Diagnosis{topic_suffix}",
            "expert_id": str(assigned_expert.id) if assigned_expert else None,
            "expert_name": assigned_expert.name if assigned_expert else None,
            "expert_specialty": assigned_expert.specialty if assigned_expert else None,
            "expert_phone": assigned_expert.phone_number if assigned_expert else None,
            "region": region_str or "District Agriculture Office",
            "created_at": datetime.utcnow().isoformat(),
        }

        if farmer_id:
            await repo.record_escalation_ticket(farmer_id, ticket_data)

        # Step 6: Format Outbound Confirmation Block with Demo Isolation Guard
        from src.config import get_settings
        settings = get_settings()
        is_verified = is_verified_expert(assigned_expert, app_env=settings.app_env)

        header = labels["header"].format(ticket_id=new_ticket_id)
        status_text = labels["status_assigned"] if (assigned_expert and is_verified) else labels["status_pending"]

        lines = []
        if reason == "hazard":
            lines.append(labels["hazard_warning"])
            lines.append("")
        elif reason == "inspection":
            lines.append(labels["inspection_note"])
            lines.append("")

        lines.extend([
            header,
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"{labels['status']}: {status_text}",
        ])

        if assigned_expert and is_verified:
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

