from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.memory.models import FarmerMemory
from src.core.logging import logger

class FarmerMemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_farmer_id(self, farmer_id: UUID) -> Optional[FarmerMemory]:
        """Fetch FarmerMemory record by farmer_id."""
        result = await self.session.execute(
            select(FarmerMemory).where(FarmerMemory.farmer_id == farmer_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, farmer_id: UUID) -> FarmerMemory:
        """Fetch existing FarmerMemory or create default record if absent."""
        memory = await self.get_by_farmer_id(farmer_id)
        if not memory:
            logger.info(f"Creating default FarmerMemory record for farmer {farmer_id}")
            memory = FarmerMemory(
                farmer_id=farmer_id,
                primary_crops=[],
                secondary_crops=[],
                crop_history=[],
                disease_history=[],
                pesticide_history=[],
                fertilizer_history=[],
                yield_history=[],
                favorite_shops=[],
                purchase_history=[],
                preferred_brands=[],
                government_schemes_used=[],
                expert_consultation_history=[],
                frequently_asked_questions=[],
                ai_learned_preferences={},
                risk_factors=[],
                confidence_scores={},
                gps_coordinates={},
            )
            self.session.add(memory)
            await self.session.commit()
            await self.session.refresh(memory)
        return memory

    async def save(self, memory: FarmerMemory) -> FarmerMemory:
        """Persist memory changes."""
        self.session.add(memory)
        await self.session.commit()
        await self.session.refresh(memory)
        return memory
