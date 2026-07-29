import math
from typing import Optional, List, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from src.core.models import Shop, Inventory
from src.shops.schemas import ShopCreate, ShopUpdate


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Great Circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


class ShopRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: ShopCreate) -> Shop:
        shop = Shop(**data.model_dump())
        self.db.add(shop)
        await self.db.commit()
        await self.db.refresh(shop)
        return shop

    async def get_by_id(self, shop_id: UUID) -> Optional[Shop]:
        result = await self.db.execute(select(Shop).where(Shop.id == shop_id))
        return result.scalar_one_or_none()

    async def update(self, shop_id: UUID, data: ShopUpdate) -> Optional[Shop]:
        shop = await self.get_by_id(shop_id)
        if not shop:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(shop, key, value)

        self.db.add(shop)
        await self.db.commit()
        await self.db.refresh(shop)
        return shop

    async def delete(self, shop_id: UUID) -> bool:
        shop = await self.get_by_id(shop_id)
        if not shop:
            return False

        await self.db.delete(shop)
        await self.db.commit()
        return True

    async def list_shops(
        self, page: int = 1, size: int = 20, status: Optional[str] = None
    ) -> Tuple[List[Shop], int]:
        query = select(Shop)
        count_query = select(func.count(Shop.id))

        if status:
            query = query.where(Shop.status == status)
            count_query = count_query.where(Shop.status == status)

        total_res = await self.db.execute(count_query)
        total = total_res.scalar() or 0

        query = query.order_by(Shop.shop_name.asc()).offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        shops = list(result.scalars().all())

        return shops, total

    async def search_by_location(
        self,
        district: Optional[str] = None,
        mandal: Optional[str] = None,
        village: Optional[str] = None,
        pin_code: Optional[str] = None,
    ) -> List[Shop]:
        query = select(Shop).where(Shop.status == "active")
        filters = []

        if district:
            filters.append(Shop.district.ilike(f"%{district}%"))
        if mandal:
            filters.append(Shop.mandal.ilike(f"%{mandal}%"))
        if village:
            filters.append(Shop.village.ilike(f"%{village}%"))
        if pin_code:
            filters.append(Shop.pin_code == pin_code)

        if filters:
            query = query.where(and_(*filters))

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def search_shops_by_product(self, product_query: str) -> List[Tuple[Shop, Inventory]]:
        """Search shops selling a specific product (case-insensitive substring match)."""
        stmt = (
            select(Shop, Inventory)
            .join(Inventory, Shop.id == Inventory.shop_id)
            .where(
                and_(
                    Shop.status == "active",
                    Inventory.available == True,
                    or_(
                        Inventory.product_name.ilike(f"%{product_query}%"),
                        Inventory.brand.ilike(f"%{product_query}%"),
                        Inventory.category.ilike(f"%{product_query}%"),
                    ),
                )
            )
        )
        result = await self.db.execute(stmt)
        return list(result.all())

    async def get_nearby_shops(
        self, latitude: float, longitude: float, max_radius_km: float = 50.0
    ) -> List[Tuple[Shop, float]]:
        """Fetch active shops and return them sorted by distance in km."""
        query = select(Shop).where(Shop.status == "active")
        result = await self.db.execute(query)
        all_shops = result.scalars().all()

        nearby = []
        for shop in all_shops:
            if shop.latitude is not None and shop.longitude is not None:
                dist = haversine_distance(latitude, longitude, shop.latitude, shop.longitude)
                if dist <= max_radius_km:
                    nearby.append((shop, dist))

        nearby.sort(key=lambda x: x[1])
        return nearby
