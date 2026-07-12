from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import CropHealth

class CropHealthRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, diagnosis: CropHealth) -> CropHealth:
        self.session.add(diagnosis)
        await self.session.flush()
        await self.session.refresh(diagnosis)
        return diagnosis

    async def get_by_id(self, diagnosis_id: UUID) -> Optional[CropHealth]:
        stmt = select(CropHealth).where(CropHealth.id == diagnosis_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_crop_id(self, crop_id: UUID, skip: int = 0, limit: int = 10) -> Tuple[int, List[CropHealth]]:
        count_stmt = select(func.count(CropHealth.id)).where(CropHealth.crop_id == crop_id)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(CropHealth).where(CropHealth.crop_id == crop_id).order_by(CropHealth.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def get_by_farmer_id(self, farmer_id: UUID, skip: int = 0, limit: int = 10) -> Tuple[int, List[CropHealth]]:
        count_stmt = select(func.count(CropHealth.id)).where(CropHealth.farmer_id == farmer_id)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(CropHealth).where(CropHealth.farmer_id == farmer_id).order_by(CropHealth.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[CropHealth]]:
        count_stmt = select(func.count(CropHealth.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(CropHealth).order_by(CropHealth.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, diagnosis: CropHealth) -> None:
        await self.session.delete(diagnosis)
        await self.session.flush()
