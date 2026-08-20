from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Farmer

class FarmerRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, farmer: Farmer) -> Farmer:
        self.session.add(farmer)
        await self.session.flush()
        await self.session.refresh(farmer)
        return farmer

    async def get_by_id(self, farmer_id: UUID) -> Optional[Farmer]:
        stmt = select(Farmer).where(Farmer.id == farmer_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone_number: str) -> Optional[Farmer]:
        stmt = select(Farmer).where(Farmer.phone_number == phone_number)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[Farmer]]:
        # Count total
        count_stmt = select(func.count(Farmer.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        # Get items
        stmt = select(Farmer).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, farmer: Farmer) -> None:
        await self.session.delete(farmer)
        await self.session.flush()
