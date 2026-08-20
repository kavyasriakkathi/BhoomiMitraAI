from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Farm


class FarmRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, farm: Farm) -> Farm:
        self.session.add(farm)
        await self.session.flush()
        await self.session.refresh(farm)
        return farm

    async def get_by_id(self, farm_id: UUID) -> Optional[Farm]:
        stmt = select(Farm).where(Farm.id == farm_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_farmer_id(
        self, farmer_id: UUID, skip: int = 0, limit: int = 10
    ) -> Tuple[int, List[Farm]]:
        """Get paginated farms for a specific farmer."""
        count_stmt = select(func.count(Farm.id)).where(
            Farm.farmer_id == farmer_id
        )
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = (
            select(Farm)
            .where(Farm.farmer_id == farmer_id)
            .order_by(Farm.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[Farm]]:
        """Get paginated farms across all farmers."""
        count_stmt = select(func.count(Farm.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = (
            select(Farm)
            .order_by(Farm.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, farm: Farm) -> None:
        await self.session.delete(farm)
        await self.session.flush()
