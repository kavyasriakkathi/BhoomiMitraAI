from typing import Optional, List, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from src.core.models import Inventory
from src.inventory.schemas import InventoryCreate, InventoryUpdate, StockUpdatePayload


class InventoryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: InventoryCreate) -> Inventory:
        item = Inventory(**data.model_dump())
        self.db.add(item)
        await self.db.commit()
        await self.db.refresh(item)
        return item

    async def get_by_id(self, item_id: UUID) -> Optional[Inventory]:
        result = await self.db.execute(select(Inventory).where(Inventory.id == item_id))
        return result.scalar_one_or_none()

    async def update(self, item_id: UUID, data: InventoryUpdate) -> Optional[Inventory]:
        item = await self.get_by_id(item_id)
        if not item:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(item, key, value)

        item.last_updated = datetime.utcnow()
        self.db.add(item)
        await self.db.commit()
        await self.db.refresh(item)
        return item

    async def update_stock(self, item_id: UUID, data: StockUpdatePayload) -> Optional[Inventory]:
        item = await self.get_by_id(item_id)
        if not item:
            return None

        item.quantity_in_stock = data.quantity_in_stock
        if data.available is not None:
            item.available = data.available
        else:
            # Auto update availability based on stock level
            item.available = data.quantity_in_stock > 0

        item.last_updated = datetime.utcnow()
        self.db.add(item)
        await self.db.commit()
        await self.db.refresh(item)
        return item

    async def delete(self, item_id: UUID) -> bool:
        item = await self.get_by_id(item_id)
        if not item:
            return False

        await self.db.delete(item)
        await self.db.commit()
        return True

    async def list_products_by_shop(
        self, shop_id: UUID, page: int = 1, size: int = 50, category: Optional[str] = None
    ) -> Tuple[List[Inventory], int]:
        query = select(Inventory).where(Inventory.shop_id == shop_id)
        count_query = select(func.count(Inventory.id)).where(Inventory.shop_id == shop_id)

        if category:
            query = query.where(Inventory.category.ilike(f"%{category}%"))
            count_query = count_query.where(Inventory.category.ilike(f"%{category}%"))

        total_res = await self.db.execute(count_query)
        total = total_res.scalar() or 0

        query = query.order_by(Inventory.product_name.asc()).offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return items, total

    async def search_products(
        self,
        query_str: str,
        shop_id: Optional[UUID] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> List[Inventory]:
        query = select(Inventory).where(Inventory.available == True)

        filters = []
        if shop_id:
            filters.append(Inventory.shop_id == shop_id)
        if category:
            filters.append(Inventory.category.ilike(f"%{category}%"))
        if brand:
            filters.append(Inventory.brand.ilike(f"%{brand}%"))
        if query_str:
            filters.append(
                or_(
                    Inventory.product_name.ilike(f"%{query_str}%"),
                    Inventory.brand.ilike(f"%{query_str}%"),
                    Inventory.category.ilike(f"%{query_str}%"),
                    Inventory.product_description.ilike(f"%{query_str}%"),
                )
            )

        if filters:
            query = query.where(and_(*filters))

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_low_stock_products(self, shop_id: UUID) -> List[Inventory]:
        """Fetch items where quantity_in_stock <= minimum_stock_level and quantity_in_stock > 0."""
        stmt = (
            select(Inventory)
            .where(
                and_(
                    Inventory.shop_id == shop_id,
                    Inventory.quantity_in_stock <= Inventory.minimum_stock_level,
                    Inventory.quantity_in_stock > 0,
                )
            )
            .order_by(Inventory.quantity_in_stock.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_out_of_stock_products(self, shop_id: UUID) -> List[Inventory]:
        """Fetch items where quantity_in_stock == 0 or available is False."""
        stmt = (
            select(Inventory)
            .where(
                and_(
                    Inventory.shop_id == shop_id,
                    or_(
                        Inventory.quantity_in_stock == 0,
                        Inventory.available == False,
                    ),
                )
            )
            .order_by(Inventory.product_name.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
