from typing import List, Optional, Tuple
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import GovernmentScheme, SchemeApplication, Farmer, FarmerProfile
from src.schemes.repository import SchemeRepository
from src.farmers.repository import FarmerRepository
from src.schemes.schemas import (
    GovernmentSchemeCreate,
    GovernmentSchemeResponse,
    FarmerEligibilityResponse,
    SchemeEligibilityItem,
    SchemeApplicationCreate,
    SchemeApplicationResponse,
)


class SchemeService:
    def __init__(self, repository: SchemeRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def seed_defaults_if_empty(self) -> List[GovernmentSchemeResponse]:
        schemes = await self.repository.seed_default_schemes_if_empty()
        return [GovernmentSchemeResponse.model_validate(s) for s in schemes]

    async def list_schemes(self, state: Optional[str] = None, category: Optional[str] = None) -> List[GovernmentSchemeResponse]:
        # Ensure default schemes exist
        await self.repository.seed_default_schemes_if_empty()
        schemes = await self.repository.get_all_active(state=state, category=category)
        return [GovernmentSchemeResponse.model_validate(s) for s in schemes]

    async def create_scheme(self, data: GovernmentSchemeCreate) -> GovernmentSchemeResponse:
        existing = await self.repository.get_by_code(data.scheme_code)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Scheme with code '{data.scheme_code}' already exists."
            )

        scheme = GovernmentScheme(**data.model_dump())
        created = await self.repository.create_scheme(scheme)
        return GovernmentSchemeResponse.model_validate(created)

    async def evaluate_farmer_eligibility(self, farmer_id: UUID) -> FarmerEligibilityResponse:
        """
        AI Government Schemes Eligibility Engine
        Analyzes farmer's state, district, land size, and crop to determine match percentage,
        eligibility reasons, and recommended action steps with voice explanation.
        """
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farmer with ID '{farmer_id}' not found."
            )

        # Seed defaults
        schemes = await self.repository.seed_default_schemes_if_empty()

        profile: Optional[FarmerProfile] = farmer.profile
        farmer_state = (profile.state if profile and profile.state else "Telangana").strip()
        farmer_district = (profile.district if profile and profile.district else "Jagtial").strip()
        farmer_land = profile.land_size_acres if profile and profile.land_size_acres is not None else 5.0
        farmer_name = profile.full_name if profile and profile.full_name else "Farmer"

        evaluated_items: List[SchemeEligibilityItem] = []
        eligible_count = 0

        for scheme in schemes:
            is_eligible = True
            reasons = []
            score = 100

            # 1. State matching check
            if scheme.state.lower() != "all india" and farmer_state.lower() not in scheme.state.lower():
                is_eligible = False
                score -= 40
                reasons.append(f"Scheme is specific to {scheme.state} state.")
            else:
                reasons.append(f"State '{farmer_state}' matches target region ({scheme.state}).")

            # 2. Land Size Check
            if farmer_land < scheme.min_land_acres:
                is_eligible = False
                score -= 30
                reasons.append(f"Requires minimum land of {scheme.min_land_acres} acres (You have {farmer_land} acres).")
            elif scheme.max_land_acres is not None and farmer_land > scheme.max_land_acres:
                is_eligible = False
                score -= 30
                reasons.append(f"Requires land size up to {scheme.max_land_acres} acres (You have {farmer_land} acres).")
            else:
                reasons.append(f"Land size ({farmer_land} acres) fits scheme land boundaries.")

            if is_eligible:
                eligible_count += 1
                recommended_action = f"Gather required documents ({scheme.required_documents}) and apply online or at Meeseva/CSC center."
                voice_exp = f"Namaste {farmer_name}! You are 100% eligible for {scheme.scheme_name}. Benefits include {scheme.benefits_summary}. Recommended action: {recommended_action}"
            else:
                recommended_action = f"Check alternative schemes or update land records."
                voice_exp = f"{scheme.scheme_name} is currently not matching your profile: {' '.join(reasons)}"

            evaluated_items.append(
                SchemeEligibilityItem(
                    scheme=GovernmentSchemeResponse.model_validate(scheme),
                    is_eligible=is_eligible,
                    match_score_percentage=max(score, 20),
                    eligibility_reason=" | ".join(reasons),
                    recommended_action=recommended_action,
                    voice_explanation=voice_exp
                )
            )

        # Sort eligible schemes first, then by match score
        evaluated_items.sort(key=lambda x: (x.is_eligible, x.match_score_percentage), reverse=True)

        return FarmerEligibilityResponse(
            farmer_id=farmer_id,
            farmer_name=farmer_name,
            state=farmer_state,
            district=farmer_district,
            land_size_acres=farmer_land,
            total_schemes_evaluated=len(schemes),
            eligible_schemes_count=eligible_count,
            schemes=evaluated_items
        )

    async def apply_for_scheme(self, data: SchemeApplicationCreate) -> SchemeApplicationResponse:
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        scheme = await self.repository.get_by_id(data.scheme_id)
        if not scheme:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Government scheme not found.")

        app = SchemeApplication(
            farmer_id=data.farmer_id,
            scheme_id=data.scheme_id,
            status="Applied",
            notes=data.notes or f"Application submitted via BhoomiMitra AI for {scheme.scheme_name}"
        )
        created = await self.repository.create_application(app)
        
        resp = SchemeApplicationResponse.model_validate(created)
        resp.scheme_name = scheme.scheme_name
        return resp

    async def get_farmer_applications(self, farmer_id: UUID) -> List[SchemeApplicationResponse]:
        apps = await self.repository.get_farmer_applications(farmer_id)
        results = []
        for a in apps:
            resp = SchemeApplicationResponse.model_validate(a)
            if a.scheme:
                resp.scheme_name = a.scheme.scheme_name
            results.append(resp)
        return results


# ---------------------------------------------------------------------------
# Intent Detection Keywords
# ---------------------------------------------------------------------------

_SCHEME_KEYWORDS_EN = {
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "kisan",
    "fasal bima", "insurance", "credit", "kcc", "solar pump", "government",
    "apply", "eligible", "eligibility", "benefits", "pm kisan",
    "rythu bandhu", "kusum", "pmfby",
}
_SCHEME_KEYWORDS_TE = {
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "పంట బీమా", "యోజన",
    "కిసాన్", "ప్రభుత్వ", "రైతు బంధు", "అర్హత", "ప్రయోజనాలు",
    "క్రెడిట్ కార్డ్", "సౌర పంప్", "ఆర్థిక సహాయం", "గ్రాంట్",
}

# Known crop nouns for future crop-specific prioritisation (not used for exclusion).
_CROP_KEYWORDS: dict = {
    "cotton": "Cotton",   "పత్తి": "Cotton",
    "rice": "Rice",       "వరి": "Rice",       "ధాన్యం": "Rice",
    "maize": "Maize",    "మొక్కజొన్న": "Maize",
    "wheat": "Wheat",    "గోధుమ": "Wheat",
    "chilli": "Chilli",  "మిర్చి": "Chilli",
    "tomato": "Tomato",  "టమాటా": "Tomato",
    "groundnut": "Groundnut", "వేరుశనగ": "Groundnut",
    "soybean": "Soybean", "సోయాబీన్": "Soybean",
    "turmeric": "Turmeric", "పసుపు": "Turmeric",
}

# ---------------------------------------------------------------------------
# Static WhatsApp Response Labels
# ---------------------------------------------------------------------------

_TE_LABELS = {
    "title":       "🏛️ మీకు వర్తించే ప్రభుత్వ పథకాలు ({count} పథకాలు)",
    "benefits":    "💰 ప్రయోజనాలు",
    "eligibility": "✅ అర్హత",
    "documents":   "📄 అవసరమైన పత్రాలు",
    "deadline":    "📅 దరఖాస్తు గడువు",
    "portal":      "🔗 అధికారిక పోర్టల్",
    "no_portal":   "సమీప మీసేవా / CSC కేంద్రాన్ని సంప్రదించండి",
    "disclaimer":  (
        "⚠️ గమనిక: దరఖాస్తు చేసే ముందు అధికారిక పోర్టల్‌లో "
        "తాజా వివరాలు మరియు అర్హత నిబంధనలు నిర్ధారించుకోండి. "
        "ప్రభుత్వ నిబంధనలు ఎప్పుడైనా మారవచ్చు."
    ),
    "more":        "మరిన్ని పథకాల కోసం: /schemes",
    "crop_note":   "(మీరు పేర్కొన్న పంట: {crop})",
}
_EN_LABELS = {
    "title":       "🏛️ Government Schemes Available For You ({count} schemes)",
    "benefits":    "💰 Benefits",
    "eligibility": "✅ Eligibility",
    "documents":   "📄 Required Documents",
    "deadline":    "📅 Application Deadline",
    "portal":      "🔗 Official Portal",
    "no_portal":   "Contact your nearest Meeseva / CSC centre",
    "disclaimer":  (
        "⚠️ Note: Please verify scheme details, amounts and deadlines at the "
        "official portal before applying. Government rules may change at any time."
    ),
    "more":        "See all schemes at: /schemes",
    "crop_note":   "(Crop you mentioned: {crop})",
}


def _detect_scheme_intent(query_lower: str) -> bool:
    """Return True if the query contains a government-scheme keyword."""
    if any(kw in query_lower for kw in _SCHEME_KEYWORDS_EN):
        return True
    # Telugu keywords need exact substring match (not lowercased)
    return False


def _detect_crop_from_query(query_text: str) -> Optional[str]:
    """
    Return a normalised English crop name if the farmer mentions a known crop.
    Used for future prioritisation — never for scheme exclusion.
    """
    query_lower = query_text.lower()
    for kw, normalised in _CROP_KEYWORDS.items():
        if kw in query_lower or kw in query_text:
            return normalised
    return None


def _sort_schemes_by_crop_priority(
    schemes: list,
    mentioned_crop: Optional[str],
) -> list:
    """
    Sort schemes so that crop-specific matches appear first.
    "All Crops" schemes are always included — never excluded.
    Schemes matching the mentioned crop get a higher sort priority.
    """
    def priority(scheme) -> int:
        crop_type = (scheme.crop_type or "All Crops").lower()
        if crop_type == "all crops":
            return 1  # Second priority — always applicable
        if mentioned_crop and mentioned_crop.lower() in crop_type:
            return 0  # Highest priority — exact crop match
        return 2     # Lower priority — different specific crop

    return sorted(schemes, key=priority)


def _format_scheme_block(scheme, labels: dict, language: str) -> str:
    """Format a single scheme as a WhatsApp-friendly text block."""
    deadline_str = ""
    if scheme.application_deadline:
        try:
            deadline_str = scheme.application_deadline.strftime("%d %b %Y")
        except Exception:
            deadline_str = str(scheme.application_deadline)

    portal_str = scheme.official_portal_url or labels["no_portal"]

    lines = [
        f"\n*{scheme.scheme_name}*",
        f"{labels['benefits']}: {scheme.benefits_summary}",
        f"{labels['eligibility']}: {scheme.eligibility_criteria}",
        f"{labels['documents']}: {scheme.required_documents}",
    ]
    if deadline_str:
        lines.append(f"{labels['deadline']}: {deadline_str}")
    lines.append(f"{labels['portal']}: {portal_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

async def enrich_response_with_schemes(
    db,
    query_text: str,
    ai_response: str,
    farmer,
) -> str:
    """
    Detect government-scheme intent in the farmer's query.
    If detected, append a formatted, safety-disclaimed scheme block to the AI response.

    Always returns the original ai_response unchanged if:
    - No scheme intent is detected
    - No schemes are found in the database
    - Any exception occurs
    """
    from src.core.logging import logger

    query_lower = query_text.lower()

    # Step 1: Detect intent (English keywords or Telugu keywords)
    has_intent = _detect_scheme_intent(query_lower)
    if not has_intent:
        # Check Telugu keywords against original (non-lowercased) text
        has_intent = any(kw in query_text for kw in _SCHEME_KEYWORDS_TE)
    if not has_intent:
        return ai_response

    # Step 2: Identify crop mentioned (for prioritisation only — not exclusion)
    mentioned_crop: Optional[str] = _detect_crop_from_query(query_text)

    # Step 3: Resolve farmer location for state filtering
    farmer_state: Optional[str] = None
    language = getattr(farmer, "preferred_language", "en") or "en"

    try:
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        profile_result = await db.execute(
            select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
        )
        profile = profile_result.scalar_one_or_none()
        if profile and profile.state:
            farmer_state = profile.state.strip()
    except Exception as loc_err:
        logger.warning(f"[SCHEMES ENRICH] Could not resolve farmer state: {loc_err}")

    # Step 4: Fetch matching schemes from DB
    try:
        repo = SchemeRepository(db)
        # Seed defaults if the table is empty (idempotent)
        await repo.seed_default_schemes_if_empty()
        schemes = await repo.get_all_active(state=farmer_state)
    except Exception as db_err:
        logger.warning(f"[SCHEMES ENRICH] DB query failed: {db_err}")
        return ai_response

    if not schemes:
        logger.info("[SCHEMES ENRICH] No active schemes found — returning original response.")
        return ai_response

    # Step 5: Sort — crop-specific matches first, then "All Crops", then others
    ranked = _sort_schemes_by_crop_priority(schemes, mentioned_crop)

    # Step 6: Cap at top 3
    top = ranked[:3]

    # Step 7: Format WhatsApp reply block
    labels = _TE_LABELS if language == "te" else _EN_LABELS
    header = labels["title"].format(count=len(top))
    if mentioned_crop:
        header += "\n" + labels["crop_note"].format(crop=mentioned_crop)

    scheme_blocks = [_format_scheme_block(s, labels, language) for s in top]
    footer_parts = [labels["disclaimer"]]
    if len(schemes) > 3:
        footer_parts.append(labels["more"])

    full_block = "\n".join([
        header,
        *scheme_blocks,
        "",
        "\n".join(footer_parts),
    ])

    logger.info(
        f"[SCHEMES ENRICH] Appending {len(top)} schemes "
        f"(crop hint: {mentioned_crop}, state: {farmer_state})."
    )
    return ai_response + "\n\n" + full_block
