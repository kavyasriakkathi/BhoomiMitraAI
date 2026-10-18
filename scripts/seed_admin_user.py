"""
Seed or reset an Administrator account for BhoomiMitra AI.

Usage:
    python scripts/seed_admin_user.py [email] [password]

Defaults:
    Email:    admin@bhoomimitra.ai
    Password: from ADMIN_DEFAULT_PASSWORD or prompts/defaults
"""

import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.core.models import UserAccount
from src.auth.constants import UserRole
from src.auth.security import hash_password


async def seed_admin(
    email: str = "admin@bhoomimitra.ai",
    password: str = "AdminBhoomi2026!",
):
    normalized_email = email.lower().strip()
    hashed = hash_password(password)

    async with AsyncSessionLocal() as db:
        res = await db.execute(select(UserAccount).where(UserAccount.email == normalized_email))
        user = res.scalar_one_or_none()

        if user:
            user.password_hash = hashed
            user.role = UserRole.ADMIN.value
            user.is_active = True
            action = "UPDATED (password reset)"
        else:
            user = UserAccount(
                email=normalized_email,
                password_hash=hashed,
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            db.add(user)
            action = "CREATED"

        await db.commit()
        await db.refresh(user)

        print("=" * 65)
        print("BHOOMIMITRA AI — ADMINISTRATOR ACCOUNT PROVISIONED")
        print("=" * 65)
        print(f"  Status:   {action}")
        print(f"  Email:    {user.email}")
        print(f"  Password: {password}")
        print(f"  Role:     {user.role}")
        print(f"  User ID:  {user.id}")
        print("=" * 65)
        return user


if __name__ == "__main__":
    target_email = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ADMIN_EMAIL", "admin@bhoomimitra.ai")
    target_password = sys.argv[2] if len(sys.argv) > 2 else os.getenv("ADMIN_PASSWORD", "AdminBhoomi2026!")

    asyncio.run(seed_admin(target_email, target_password))
