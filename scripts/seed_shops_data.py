import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.core.models import Shop, Inventory


async def seed_data():
    """Seed sample Agri Shop and Inventory into the database."""
    async with AsyncSessionLocal() as db:
        print("🌱 Seeding sample Agri Shop data...")

        # Check if shop already exists
        result = await db.execute(select(Shop).where(Shop.phone_number == "+91 9876543210"))
        existing_shop = result.scalar_one_or_none()

        if not existing_shop:
            shop = Shop(
                shop_name="Sri Lakshmi Agro Centre",
                owner_name="Ramesh Kumar",
                phone_number="+91 9876543210",
                email="contact@srilakshmiagro.com",
                address="Main Road, Guntur",
                village="Guntur Rural",
                mandal="Guntur",
                district="Guntur",
                state="Andhra Pradesh",
                pin_code="522001",
                latitude=16.3067,
                longitude=80.4365,
                opening_time="08:00 AM",
                closing_time="08:00 PM",
                delivery_available=True,
                home_delivery_radius_km=15.0,
                google_maps_link="https://maps.google.com/?q=16.3067,80.4365",
                gst_number="37ABCDE1234F1Z5",
                license_number="AP/GNT/AGRI/2026/089",
                status="active",
            )
            db.add(shop)
            await db.flush()
            print(f"✅ Created Shop: {shop.shop_name} ({shop.id})")
        else:
            shop = existing_shop
            print(f"ℹ️ Shop already exists: {shop.shop_name} ({shop.id})")

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
                print(f"  └─ Added Product: {item_data['product_name']} ({item_data['brand']}) - ₹{item_data['price']}")
            else:
                print(f"  └─ Product already exists: {item_data['product_name']}")

        await db.commit()
        print("✨ Seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_data())
