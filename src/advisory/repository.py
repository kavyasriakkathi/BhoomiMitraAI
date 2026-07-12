from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Advisory

class AdvisoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, advisory: Advisory) -> Advisory:
        self.session.add(advisory)
        await self.session.flush()
        await self.session.refresh(advisory)
        return advisory

    async def get_by_id(self, advisory_id: UUID) -> Optional[Advisory]:
        stmt = select(Advisory).where(Advisory.id == advisory_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(self, page: int = 1, size: int = 10) -> Tuple[int, List[Advisory]]:
        skip = (page - 1) * size
        count_stmt = select(func.count(Advisory.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(Advisory).order_by(Advisory.created_at.desc()).offset(skip).limit(size)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def list_by_farmer(self, farmer_id: UUID, page: int = 1, size: int = 10) -> Tuple[int, List[Advisory]]:
        skip = (page - 1) * size
        count_stmt = select(func.count(Advisory.id)).where(Advisory.farmer_id == farmer_id)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(Advisory).where(Advisory.farmer_id == farmer_id).order_by(Advisory.created_at.desc()).offset(skip).limit(size)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def update(self, advisory: Advisory) -> Advisory:
        await self.session.flush()
        await self.session.refresh(advisory)
        return advisory

    async def delete(self, advisory: Advisory) -> None:
        await self.session.delete(advisory)
        await self.session.flush()
