from typing import List, Optional
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

    async def seed_defaults_if_empty() -> List[GovernmentSchemeResponse]:
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
