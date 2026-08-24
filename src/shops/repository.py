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

    async def search_shops_by_product(
        self, product_query: str, only_available: bool = False
    ) -> List[Tuple[Shop, Inventory]]:
        """Search shops selling a specific product (case-insensitive substring match)."""
        conditions = [
            Shop.status == "active",
            or_(
                Inventory.product_name.ilike(f"%{product_query}%"),
                Inventory.brand.ilike(f"%{product_query}%"),
                Inventory.category.ilike(f"%{product_query}%"),
            ),
        ]
        if only_available:
            conditions.append(Inventory.available == True)

        stmt = (
            select(Shop, Inventory)
            .join(Inventory, Shop.id == Inventory.shop_id)
            .where(and_(*conditions))
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

    async def seed_default_shops_if_empty(self) -> List[Shop]:
        """Idempotently seed default shops and inventory if table is empty."""
        count_res = await self.db.execute(select(func.count(Shop.id)))
        count = count_res.scalar() or 0
        if count > 0:
            result = await self.db.execute(select(Shop).where(Shop.status == "active"))
            return list(result.scalars().all())

        import uuid
        default_shops_data = [
            {
                "shop_name": "Sri Lakshmi Agro Centre",
                "owner_name": "Ramesh Kumar",
                "phone_number": "+91 9876543210",
                "email": "contact@srilakshmiagro.com",
                "address": "Main Road, Guntur",
                "village": "Guntur Rural",
                "mandal": "Guntur",
                "district": "Guntur",
                "state": "Andhra Pradesh",
                "pin_code": "522001",
                "latitude": 16.3067,
                "longitude": 80.4365,
                "opening_time": "08:00",
                "closing_time": "20:00",
                "delivery_available": True,
                "home_delivery_radius_km": 15.0,
                "status": "active",
                "inventory": [
                    {
                        "product_name": "Urea",
                        "category": "Fertilizers",
                        "brand": "IFFCO",
                        "product_description": "Neem Coated Urea 45kg Bag",
                        "unit": "Bag",
                        "price": 295.0,
                        "quantity_in_stock": 80,
                        "minimum_stock_level": 10,
                        "available": True,
                    },
                    {
                        "product_name": "DAP",
                        "category": "Fertilizers",
                        "brand": "Coromandel",
                        "product_description": "Di-Ammonium Phosphate 50kg Bag",
                        "unit": "Bag",
                        "price": 1350.0,
                        "quantity_in_stock": 45,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Neem Oil 1500 PPM",
                        "category": "Pesticides",
                        "brand": "Krishi Bio",
                        "product_description": "Cold-pressed pure organic neem oil 1L",
                        "unit": "Bottle",
                        "price": 350.0,
                        "quantity_in_stock": 20,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Cotton Seeds RCH-659",
                        "category": "Seeds",
                        "brand": "Rasi Seeds",
                        "product_description": "High yield BG-II certified hybrid cotton seeds",
                        "unit": "Packet",
                        "price": 860.0,
                        "quantity_in_stock": 30,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                ],
            },
            {
                "shop_name": "Kisan Seva Kendra",
                "owner_name": "Srinivas Rao",
                "phone_number": "+91 9848012345",
                "email": "kisan.warangal@gmail.com",
                "address": "Opp. Market Yard, Hanamkonda",
                "village": "Hanamkonda",
                "mandal": "Hanamkonda",
                "district": "Warangal",
                "state": "Telangana",
                "pin_code": "506001",
                "latitude": 17.9689,
                "longitude": 79.5941,
                "opening_time": "08:00",
                "closing_time": "20:00",
                "delivery_available": True,
                "home_delivery_radius_km": 20.0,
                "status": "active",
                "inventory": [
                    {
                        "product_name": "Urea",
                        "category": "Fertilizers",
                        "brand": "KRIBHCO",
                        "product_description": "Neem Coated Urea 45kg Bag",
                        "unit": "Bag",
                        "price": 295.0,
                        "quantity_in_stock": 120,
                        "minimum_stock_level": 15,
                        "available": True,
                    },
                    {
                        "product_name": "DAP",
                        "category": "Fertilizers",
                        "brand": "IFFCO",
                        "product_description": "High-grade DAP fertilizer 50kg Bag",
                        "unit": "Bag",
                        "price": 1350.0,
                        "quantity_in_stock": 60,
                        "minimum_stock_level": 10,
                        "available": True,
                    },
                    {
                        "product_name": "Imidacloprid 17.8% SL",
                        "category": "Pesticides",
                        "brand": "Bayer Confidor",
                        "product_description": "Systemic insecticide for sucking pests 250ml",
                        "unit": "Bottle",
                        "price": 420.0,
                        "quantity_in_stock": 25,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Paddy Seeds BPT-5204",
                        "category": "Seeds",
                        "brand": "Telangana Seeds",
                        "product_description": "Samba Mahsuri certified paddy seed 25kg Bag",
                        "unit": "Bag",
                        "price": 950.0,
                        "quantity_in_stock": 40,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                ],
            },
            {
                "shop_name": "Rythu Mithra Agri Inputs",
                "owner_name": "Venkat Reddy",
                "phone_number": "+91 9988776655",
                "email": "rythumithra.khammam@gmail.com",
                "address": "Wyra Road, Khammam",
                "village": "Khammam",
                "mandal": "Khammam Urban",
                "district": "Khammam",
                "state": "Telangana",
                "pin_code": "507001",
                "latitude": 17.2473,
                "longitude": 80.1514,
                "opening_time": "08:30",
                "closing_time": "19:30",
                "delivery_available": False,
                "home_delivery_radius_km": 0.0,
                "status": "active",
                "inventory": [
                    {
                        "product_name": "MOP Potash",
                        "category": "Fertilizers",
                        "brand": "IPL",
                        "product_description": "Muriate of Potash 50kg Bag",
                        "unit": "Bag",
                        "price": 1700.0,
                        "quantity_in_stock": 35,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Chlorpyrifos 20% EC",
                        "category": "Pesticides",
                        "brand": "Tata Rallis",
                        "product_description": "Broad spectrum insecticide 1L",
                        "unit": "Bottle",
                        "price": 380.0,
                        "quantity_in_stock": 15,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Chilli Seeds Teja",
                        "category": "Seeds",
                        "brand": "Mahyco",
                        "product_description": "High pungent hot pepper hybrid seeds 100g",
                        "unit": "Packet",
                        "price": 1200.0,
                        "quantity_in_stock": 15,
                        "minimum_stock_level": 3,
                        "available": True,
                    },
                    {
                        "product_name": "Zinc Sulphate 33%",
                        "category": "Micronutrients",
                        "brand": "Coromandel",
                        "product_description": "Agricultural grade zinc micro-nutrient 5kg",
                        "unit": "Packet",
                        "price": 180.0,
                        "quantity_in_stock": 50,
                        "minimum_stock_level": 10,
                        "available": True,
                    },
                ],
            },
            {
                "shop_name": "Balaji Fertilizers & Pesticides",
                "owner_name": "Mallikarjun Goud",
                "phone_number": "+91 9701234567",
                "email": "balaji.karimnagar@gmail.com",
                "address": "Collectorate Road, Karimnagar",
                "village": "Karimnagar",
                "mandal": "Karimnagar",
                "district": "Karimnagar",
                "state": "Telangana",
                "pin_code": "505001",
                "latitude": 18.4386,
                "longitude": 79.1288,
                "opening_time": "08:00",
                "closing_time": "20:30",
                "delivery_available": True,
                "home_delivery_radius_km": 10.0,
                "status": "active",
                "inventory": [
                    {
                        "product_name": "Urea",
                        "category": "Fertilizers",
                        "brand": "NFCL",
                        "product_description": "Neem Coated Urea 45kg Bag",
                        "unit": "Bag",
                        "price": 295.0,
                        "quantity_in_stock": 50,
                        "minimum_stock_level": 10,
                        "available": True,
                    },
                    {
                        "product_name": "Fungicide Mancozeb 75% WP",
                        "category": "Fungicides",
                        "brand": "UPL Saaf",
                        "product_description": "Contact & systemic fungicide 500g",
                        "unit": "Packet",
                        "price": 320.0,
                        "quantity_in_stock": 30,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                    {
                        "product_name": "Neem Oil 1500 PPM",
                        "category": "Pesticides",
                        "brand": "Godrej Agrovet",
                        "product_description": "Organic neem formulation for whitefly & bollworm 1L",
                        "unit": "Bottle",
                        "price": 340.0,
                        "quantity_in_stock": 18,
                        "minimum_stock_level": 5,
                        "available": True,
                    },
                ],
            },
        ]

        created_shops = []
        for s_data in default_shops_data:
            inv_list = s_data.pop("inventory")
            shop = Shop(id=uuid.uuid4(), **s_data)
            self.db.add(shop)
            await self.db.flush()
            for inv_data in inv_list:
                inv = Inventory(id=uuid.uuid4(), shop_id=shop.id, **inv_data)
                self.db.add(inv)
            created_shops.append(shop)

        await self.db.commit()
        for s in created_shops:
            await self.db.refresh(s)
        return created_shops
