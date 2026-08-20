from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.core.models import Expert
from src.memory.models import FarmerMemory
from src.escalation.schemas import ExpertCreate, ExpertUpdate


class EscalationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def seed_default_experts_if_empty(self) -> List[Expert]:
        """Idempotently seeds standard agricultural officers and specialists if table is empty."""
        count_res = await self.db.execute(select(func.count(Expert.id)))
        count = count_res.scalar() or 0
        if count > 0:
            result = await self.db.execute(select(Expert).where(Expert.is_active == True))
            return list(result.scalars().all())

        default_experts = [
            {
                "id": uuid4(),
                "name": "Dr. K. Srinivas Rao",
                "phone_number": "+91 9848012345",
                "specialty": "Pest Control & Crop Protection",
                "is_active": True,
            },
            {
                "id": uuid4(),
                "name": "Dr. Ananya Sharma",
                "phone_number": "+91 9876543211",
                "specialty": "Soil Health & Fertilizer Management",
                "is_active": True,
            },
            {
                "id": uuid4(),
                "name": "Sri V. Mallikarjun",
                "phone_number": "+91 9701234568",
                "specialty": "Agronomy & Irrigation",
                "is_active": True,
            },
            {
                "id": uuid4(),
                "name": "Dr. P. Venkateswarlu",
                "phone_number": "+91 9988776656",
                "specialty": "Cotton & Commercial Crop Specialist",
                "is_active": True,
            },
        ]

        created_experts = []
        for exp_data in default_experts:
            exp = Expert(**exp_data)
            self.db.add(exp)
            created_experts.append(exp)

        await self.db.commit()
        for e in created_experts:
            await self.db.refresh(e)
        return created_experts

    async def get_active_experts(self, specialty: Optional[str] = None) -> List[Expert]:
        """Fetch all active agricultural experts, optionally filtered by specialty keyword."""
        query = select(Expert).where(Expert.is_active == True)
        if specialty:
            query = query.where(Expert.specialty.ilike(f"%{specialty}%"))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_id(self, expert_id: UUID) -> Optional[Expert]:
        """Fetch an expert by UUID."""
        result = await self.db.execute(select(Expert).where(Expert.id == expert_id))
        return result.scalar_one_or_none()

    async def create(self, data: ExpertCreate) -> Expert:
        """Register a new agricultural expert."""
        expert = Expert(**data.model_dump())
        self.db.add(expert)
        await self.db.commit()
        await self.db.refresh(expert)
        return expert

    async def update(self, expert_id: UUID, data: ExpertUpdate) -> Optional[Expert]:
        """Update expert details."""
        expert = await self.get_by_id(expert_id)
        if not expert:
            return None

        update_dict = data.model_dump(exclude_unset=True)
        for k, v in update_dict.items():
            setattr(expert, k, v)

        self.db.add(expert)
        await self.db.commit()
        await self.db.refresh(expert)
        return expert

    async def get_farmer_consultation_history(self, farmer_id: UUID) -> List[Dict[str, Any]]:
        """Fetch past expert escalation tickets from FarmerMemory."""
        result = await self.db.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer_id)
        )
        memory = result.scalar_one_or_none()
        if memory and memory.expert_consultation_history:
            return list(memory.expert_consultation_history)
        return []

    async def record_escalation_ticket(
        self,
        farmer_id: UUID,
        ticket_data: Dict[str, Any],
    ) -> bool:
        """Safely append an escalation ticket to FarmerMemory.expert_consultation_history."""
        result = await self.db.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer_id)
        )
        memory = result.scalar_one_or_none()
        if not memory:
            memory = FarmerMemory(
                farmer_id=farmer_id,
                expert_consultation_history=[ticket_data],
            )
            self.db.add(memory)
        else:
            history = list(memory.expert_consultation_history or [])
            history.append(ticket_data)
            memory.expert_consultation_history = history
            self.db.add(memory)

        await self.db.commit()
        return True
