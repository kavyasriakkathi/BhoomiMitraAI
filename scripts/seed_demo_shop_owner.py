"""
Dedicated seed script for BhoomiMitra AI Demo Agri Shop Owner account.
Creates or updates demo.shopowner@bhoomimitra.ai linked to 'Mallanna Fertilizer Seeds and Pesticides'.

Usage:
    python scripts/seed_demo_shop_owner.py

Optional environment variable:
    DEMO_SHOP_OWNER_PASSWORD - If set, uses this password; otherwise generates a cryptographically secure random password.
"""

import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.core.models import Shop
from scripts.seed_shops_data import seed_data, seed_demo_shop_owner, DEMO_SHOP_OWNER_EMAIL


async def main():
    async with AsyncSessionLocal() as db:
        # Find Mallanna shop
        res = await db.execute(
            select(Shop).where(
                (Shop.shop_name.ilike("%Mallanna Fertilizer%")) | (Shop.phone_number == "8976547654")
            )
        )
        shop = res.scalar_one_or_none()

        if not shop:
            print("[INFO] Mallanna shop not found. Running full shop seed first...")
            result = await seed_data()
            return result

        demo_user, temp_password = await seed_demo_shop_owner(db, shop)
        await db.commit()

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
    asyncio.run(main())
