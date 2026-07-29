import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, delete
from src.core.database import AsyncSessionLocal
from src.core.models import Shop, Inventory


async def seed_data():
    """Seed real Agri Shop 'Mallanna Fertilizer Seeds and Pesticides' and Inventory into database."""
    async with AsyncSessionLocal() as db:
        print("[SEED] Cleaning old sample shop data...")

        # Remove old sample shops (e.g. Sri Lakshmi Agro Centre or old dummy numbers)
        old_shops_res = await db.execute(
            select(Shop).where(Shop.shop_name.ilike("%Sri Lakshmi%") | (Shop.phone_number == "+91 9876543210"))
        )
        old_shops = old_shops_res.scalars().all()
        for old in old_shops:
            await db.delete(old)
        if old_shops:
            await db.flush()
            print(f"[CLEANUP] Removed {len(old_shops)} old sample shop record(s).")

        print("[SEED] Seeding real shop 'Mallanna Fertilizer Seeds and Pesticides'...")

        # Check if Mallanna shop already exists
        result = await db.execute(select(Shop).where(Shop.phone_number == "8976547654"))
        existing_shop = result.scalar_one_or_none()

        if not existing_shop:
            shop = Shop(
                shop_name="Mallanna Fertilizer Seeds and Pesticides",
                owner_name="Mallanna",
                phone_number="8976547654",
                email="contact@mallannaagri.com",
                address="Kallur Road, Korutla",
                village="Korutla",
                mandal="Korutla",
                district="Jagtial",
                state="Telangana",
                pin_code="505326",
                latitude=18.8206,
                longitude=78.7119,
                opening_time="08:00 AM",
                closing_time="08:00 PM",
                delivery_available=True,
                home_delivery_radius_km=25.0,
                google_maps_link="https://www.google.com/maps/search/?api=1&query=18.8206,78.7119",
                gst_number="36AAAPM1234F1Z9",
                license_number="TS/JGT/AGRI/2026/102",
                status="active",
            )
            db.add(shop)
            await db.flush()
            print(f"[CREATED] Created Shop: {shop.shop_name} ({shop.id})")
        else:
            shop = existing_shop
            # Update existing shop details to ensure exact match
            shop.shop_name = "Mallanna Fertilizer Seeds and Pesticides"
            shop.owner_name = "Mallanna"
            shop.address = "Kallur Road, Korutla"
            shop.village = "Korutla"
            shop.mandal = "Korutla"
            shop.district = "Jagtial"
            shop.state = "Telangana"
            shop.latitude = 18.8206
            shop.longitude = 78.7119
            shop.opening_time = "08:00 AM"
            shop.closing_time = "08:00 PM"
            shop.delivery_available = True
            shop.google_maps_link = "https://www.google.com/maps/search/?api=1&query=18.8206,78.7119"
            db.add(shop)
            await db.flush()
            print(f"[UPDATED] Updated Shop: {shop.shop_name} ({shop.id})")

        # Inventory Items
        sample_items = [
            {
                "product_name": "Urea",
                "category": "Fertilizers",
                "brand": "IFFCO",
                "product_description": "Neem Coated Urea (46% Nitrogen) 45kg bag",
                "unit": "Bag",
                "price": 295.0,
                "quantity_in_stock": 50,
                "minimum_stock_level": 10,
                "available": True,
            },
            {
                "product_name": "DAP",
                "category": "Fertilizers",
                "brand": "Coromandel",
                "product_description": "Di-Ammonium Phosphate (18:46:0) 50kg bag",
                "unit": "Bag",
                "price": 1350.0,
                "quantity_in_stock": 30,
                "minimum_stock_level": 5,
                "available": True,
            },
            {
                "product_name": "Neem Oil",
                "category": "Organic Products",
                "brand": "Organic",
                "product_description": "100% Cold Pressed Organic Neem Oil Insecticide 1 Litre",
                "unit": "Bottle",
                "price": 420.0,
                "quantity_in_stock": 20,
                "minimum_stock_level": 5,
                "available": True,
            },
            {
                "product_name": "Imidacloprid 17.8 SL",
                "category": "Pesticides",
                "brand": "Bayer",
                "product_description": "Systemic Insecticide for sucking pests like thrips and aphids 500ml",
                "unit": "Bottle",
                "price": 650.0,
                "quantity_in_stock": 15,
                "minimum_stock_level": 3,
                "available": True,
            },
        ]

        for item_data in sample_items:
            res = await db.execute(
                select(Inventory).where(
                    Inventory.shop_id == shop.id,
                    Inventory.product_name == item_data["product_name"]
                )
            )
            existing_item = res.scalar_one_or_none()

            if not existing_item:
                inventory_item = Inventory(
                    shop_id=shop.id,
                    **item_data
                )
                db.add(inventory_item)
                print(f"  -> Added Product: {item_data['product_name']} ({item_data['brand']}) - RS {item_data['price']}")
            else:
                existing_item.brand = item_data["brand"]
                existing_item.price = item_data["price"]
                existing_item.quantity_in_stock = item_data["quantity_in_stock"]
                existing_item.category = item_data["category"]
                existing_item.unit = item_data["unit"]
                existing_item.available = True
                db.add(existing_item)
                print(f"  -> Updated Product: {item_data['product_name']} - RS {item_data['price']}")

        await db.commit()
        print("[SUCCESS] Real shop seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_data())
