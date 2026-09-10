import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import secrets
from typing import Optional, Tuple

from sqlalchemy import select, delete
from src.auth.constants import UserRole
from src.auth.security import hash_password
from src.core.database import AsyncSessionLocal
from src.core.models import Shop, Inventory, UserAccount

DEMO_SHOP_OWNER_EMAIL = "demo.shopowner@bhoomimitra.ai"


def get_or_generate_temp_password() -> Tuple[str, bool]:
    """
    Returns (password, is_generated).
    Reads DEMO_SHOP_OWNER_PASSWORD from environment if present.
    Otherwise generates a cryptographically secure temporary password.
    No plaintext passwords are ever hardcoded in source code.
    """
    env_pw = os.getenv("DEMO_SHOP_OWNER_PASSWORD", "").strip()
    if env_pw:
        return env_pw, False
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    random_part = "".join(secrets.choice(alphabet) for _ in range(12))
    return f"BmDemo#{random_part}", True


async def seed_demo_shop_owner(db, shop: Shop) -> Tuple[UserAccount, str]:
    """Seed or update clearly identifiable demo Agri Shop Owner account for testing."""
    print(f"[SEED] Ensuring demo Agri Shop Owner account '{DEMO_SHOP_OWNER_EMAIL}'...")
    temp_password, _ = get_or_generate_temp_password()
    hashed_pw = hash_password(temp_password)

    user_res = await db.execute(
        select(UserAccount).where(
            (UserAccount.email == DEMO_SHOP_OWNER_EMAIL) | (UserAccount.shop_id == shop.id)
        )
    )
    existing_user = user_res.scalar_one_or_none()

    if not existing_user:
        demo_user = UserAccount(
            email=DEMO_SHOP_OWNER_EMAIL,
            password_hash=hashed_pw,
            role=UserRole.SHOP_OWNER.value,
            shop_id=shop.id,
            is_active=True,
        )
        db.add(demo_user)
        await db.flush()
        print(f"[CREATED] Demo Shop Owner account created: {DEMO_SHOP_OWNER_EMAIL} ({demo_user.id})")
    else:
        demo_user = existing_user
        demo_user.email = DEMO_SHOP_OWNER_EMAIL
        demo_user.role = UserRole.SHOP_OWNER.value
        demo_user.shop_id = shop.id
        demo_user.is_active = True
        demo_user.password_hash = hashed_pw
        db.add(demo_user)
        await db.flush()
        print(f"[UPDATED] Demo Shop Owner account updated: {DEMO_SHOP_OWNER_EMAIL} ({demo_user.id})")

    return demo_user, temp_password


async def seed_data():
    """Seed real Agri Shop 'Mallanna Fertilizer Seeds and Pesticides', Inventory, and Demo Shop Owner."""
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

        # Seed or update Demo Agri Shop Owner account
        demo_user, temp_password = await seed_demo_shop_owner(db, shop)

        await db.commit()
        print("[SUCCESS] Real shop and demo shop owner seeding completed successfully!")
        print("=" * 65)
        print("BHOOMIMITRA AI — DEMO AGRI SHOP OWNER TEST CREDENTIALS")
        print("=" * 65)
        print(f"  Email:              {demo_user.email}")
        print(f"  Temporary Password: {temp_password}")
        print(f"  Role:               {demo_user.role}")
        print(f"  Linked Shop:        {shop.shop_name} ({shop.id})")
        print("=" * 65)
        return {
            "shop_id": str(shop.id),
            "shop_name": shop.shop_name,
            "user_id": str(demo_user.id),
            "email": demo_user.email,
            "role": demo_user.role,
            "temporary_password": temp_password,
        }


if __name__ == "__main__":
    asyncio.run(seed_data())
