from typing import List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import GovernmentScheme, SchemeApplication, Farmer, FarmerProfile


DEFAULT_INDIAN_SCHEMES = [
    {
        "scheme_name": "PM-KISAN Samman Nidhi",
        "scheme_code": "PM_KISAN",
        "category": "Direct Income Support",
        "state": "All India",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.0,
        "max_land_acres": 5.0,
        "description": "Pradhan Mantri Kisan Samman Nidhi is an initiative by the Government of India in which all small and marginal farmers get up to ₹6,000 per year as minimum income support.",
        "benefits_summary": "₹6,000 per year paid in 3 equal installments of ₹2,000 directly to bank account.",
        "eligibility_criteria": "Small & marginal landholding farmer families with cultivable land holding up to 2 hectares (5 acres).",
        "required_documents": "Aadhaar Card, Land Ownership Papers (Pahani/1B), Bank Account Passbook.",
        "application_deadline": datetime.utcnow() + timedelta(days=180),
        "official_portal_url": "https://pmkisan.gov.in"
    },
    {
        "scheme_name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "scheme_code": "PMFBY",
        "category": "Crop Insurance",
        "state": "All India",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.0,
        "max_land_acres": None,
        "description": "Comprehensive yield-based crop insurance scheme offering financial support to farmers suffering crop loss/damage arising out of unforeseen natural calamities.",
        "benefits_summary": "Full financial coverage against drought, flood, unseasonal rain & pest attacks with low premium rate (1.5% - 2%).",
        "eligibility_criteria": "All farmers growing notified crops in notified areas including sharecroppers and tenant farmers.",
        "required_documents": "Land Sowing Certificate, Bank Account Details, Aadhaar Card, Land Revenue Receipt.",
        "application_deadline": datetime.utcnow() + timedelta(days=45),
        "official_portal_url": "https://pmfby.gov.in"
    },
    {
        "scheme_name": "Kisan Credit Card (KCC) Subsidized Loan",
        "scheme_code": "KCC_LOAN",
        "category": "Subsidized Agriculture Credit",
        "state": "All India",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.0,
        "max_land_acres": None,
        "description": "Provides short-term formal credit to farmers at an effective subsidized interest rate of 4% per annum for prompt repayment.",
        "benefits_summary": "Collateral-free agricultural credit up to ₹1.6 Lakhs (total credit up to ₹3 Lakhs) at 4% interest rate.",
        "eligibility_criteria": "Individual farmers, joint borrowers, tenant farmers, self-help groups (SHGs).",
        "required_documents": "Application Form, Land Pattadar Passbook, Aadhaar, Voter ID, Passport Photograph.",
        "application_deadline": None,
        "official_portal_url": "https://www.myscheme.gov.in/schemes/kcc"
    },
    {
        "scheme_name": "PM-KUSUM Solar Agriculture Pump Subsidy",
        "scheme_code": "SOLAR_PUMP",
        "category": "Solar & Irrigation Subsidy",
        "state": "All India",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.5,
        "max_land_acres": 15.0,
        "description": "Promotes solar energy usage among Indian farmers by providing up to 90% subsidy for setting up off-grid solar agriculture pumpsets.",
        "benefits_summary": "60% Central & State Subsidy + 30% Bank Loan support; farmer pays only 10% of total pump cost.",
        "eligibility_criteria": "Farmers owning agricultural land with borewell/openwell requiring irrigation pumpsets.",
        "required_documents": "Land Possession Certificate, Electricity Board No-Objection Certificate, Bank Passbook, Aadhaar.",
        "application_deadline": datetime.utcnow() + timedelta(days=90),
        "official_portal_url": "https://pmkusum.mnre.gov.in"
    },
    {
        "scheme_name": "Subsidized Fertilizer & Nano Urea Scheme",
        "scheme_code": "FERTILIZER_SUBSIDY",
        "category": "Subsidized Inputs",
        "state": "All India",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.0,
        "max_land_acres": None,
        "description": "Government subsidy providing Urea, DAP, and MOP fertilizers at fixed MRPs far below international market import prices.",
        "benefits_summary": "Nano Urea 500ml bottles available at 50% lower cost than traditional bagged Urea with higher NPK efficiency.",
        "eligibility_criteria": "All active farmers purchasing via POS machines using Aadhaar at licensed agri shops.",
        "required_documents": "Aadhaar Card at licensed agri retail shop.",
        "application_deadline": None,
        "official_portal_url": "https://www.fert.nic.in"
    },
    {
        "scheme_name": "Rythu Bandhu Agriculture Investment Support",
        "scheme_code": "RYTHU_BANDHU",
        "category": "State Crop Investment",
        "state": "Telangana",
        "district": None,
        "crop_type": "All Crops",
        "min_land_acres": 0.0,
        "max_land_acres": 10.0,
        "description": "Flagship Telangana state government scheme providing direct financial grant to farmers every season for purchasing seeds, fertilizers & pesticides.",
        "benefits_summary": "₹10,000 per acre per year (₹5,000 per acre for Kharif + ₹5,000 per acre for Rabi) directly deposited to bank.",
        "eligibility_criteria": "Pattadar landowning farmers residing in Telangana state.",
        "required_documents": "Pattadar Passbook, Aadhaar Card, Bank Account Details.",
        "application_deadline": datetime.utcnow() + timedelta(days=60),
        "official_portal_url": "https://rythubandhu.telangana.gov.in"
    }
]


class SchemeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def seed_default_schemes_if_empty(self) -> List[GovernmentScheme]:
        """Auto-seeds national and state government schemes if database is empty."""
        stmt = select(func.count(GovernmentScheme.id))
        res = await self.session.execute(stmt)
        count = res.scalar_one()

        if count == 0:
            schemes = []
            for item in DEFAULT_INDIAN_SCHEMES:
                scheme = GovernmentScheme(**item)
                self.session.add(scheme)
                schemes.append(scheme)
            await self.session.flush()
            await self.session.commit()
            return schemes

        stmt_all = select(GovernmentScheme).where(GovernmentScheme.is_active == True)
        res_all = await self.session.execute(stmt_all)
        return list(res_all.scalars().all())

    async def get_all_active(self, state: Optional[str] = None, category: Optional[str] = None) -> List[GovernmentScheme]:
        stmt = select(GovernmentScheme).where(GovernmentScheme.is_active == True)
        
        if state and state.lower() != "all india":
            stmt = stmt.where(or_(GovernmentScheme.state == "All India", GovernmentScheme.state.ilike(f"%{state}%")))
            
        if category:
            stmt = stmt.where(GovernmentScheme.category.ilike(f"%{category}%"))

        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_by_id(self, scheme_id: UUID) -> Optional[GovernmentScheme]:
        stmt = select(GovernmentScheme).where(GovernmentScheme.id == scheme_id)
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def get_by_code(self, scheme_code: str) -> Optional[GovernmentScheme]:
        stmt = select(GovernmentScheme).where(GovernmentScheme.scheme_code == scheme_code)
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def create_scheme(self, scheme: GovernmentScheme) -> GovernmentScheme:
        self.session.add(scheme)
        await self.session.flush()
        await self.session.commit()
        await self.session.refresh(scheme)
        return scheme

    async def create_application(self, app: SchemeApplication) -> SchemeApplication:
        self.session.add(app)
        await self.session.flush()
        await self.session.commit()
        await self.session.refresh(app)
        return app

    async def get_farmer_applications(self, farmer_id: UUID) -> List[SchemeApplication]:
        stmt = select(SchemeApplication).where(SchemeApplication.farmer_id == farmer_id).order_by(SchemeApplication.created_at.desc())
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
