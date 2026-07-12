from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Crop

class CropRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, crop: Crop) -> Crop:
        self.session.add(crop)
        await self.session.flush()
        await self.session.refresh(crop)
        return crop

    async def get_by_id(self, crop_id: UUID) -> Optional[Crop]:
        stmt = select(Crop).where(Crop.id == crop_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_farm_id(self, farm_id: UUID, skip: int = 0, limit: int = 10) -> Tuple[int, List[Crop]]:
        count_stmt = select(func.count(Crop.id)).where(Crop.farm_id == farm_id)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(Crop).where(Crop.farm_id == farm_id).order_by(Crop.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[Crop]]:
        count_stmt = select(func.count(Crop.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = select(Crop).order_by(Crop.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, crop: Crop) -> None:
        await self.session.delete(crop)
        await self.session.flush()
