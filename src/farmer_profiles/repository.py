from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import FarmerProfile

class FarmerProfileRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, profile: FarmerProfile) -> FarmerProfile:
        self.session.add(profile)
        await self.session.flush()
        await self.session.refresh(profile)
        return profile

    async def get_by_id(self, profile_id: UUID) -> Optional[FarmerProfile]:
        stmt = select(FarmerProfile).where(FarmerProfile.id == profile_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_farmer_id(self, farmer_id: UUID) -> Optional[FarmerProfile]:
        stmt = select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[FarmerProfile]]:
        count_stmt = select(func.count(FarmerProfile.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(FarmerProfile).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, profile: FarmerProfile) -> None:
        await self.session.delete(profile)
        await self.session.flush()
