"""
Environment-controlled one-time seed mechanism for BhoomiMitra AI Demo Agri Shop Owner account.
Triggered exclusively when DEMO_SHOP_OWNER_SEED=true on startup.
Never logs or exposes the plaintext password in application logs or exceptions.
"""

import secrets
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.constants import UserRole
from src.auth.security import hash_password
from src.config import get_settings
from src.core.logging import logger
from src.core.models import Shop, UserAccount

DEMO_SHOP_OWNER_EMAIL = "demo.shopowner@bhoomimitra.ai"
MALLANNA_SHOP_NAME = "Mallanna Fertilizer Seeds and Pesticides"
MALLANNA_PHONE = "8976547654"


def _generate_secure_password() -> str:
    """Generate a cryptographically secure random password meeting complexity rules."""
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    random_part = "".join(secrets.choice(alphabet) for _ in range(14))
    return f"BmDemo#{random_part}"


async def ensure_demo_shop_owner_seeded(db: AsyncSession) -> Optional[UserAccount]:
    """
    Idempotently seeds or updates the demo Agri Shop Owner account.
    Runs ONLY when DEMO_SHOP_OWNER_SEED is True.
    Never logs or exposes the plaintext password.
    """
    settings = get_settings()
    if not settings.demo_shop_owner_seed:
        return None

    logger.info("DEMO_SHOP_OWNER_SEED is enabled. Checking demo shop owner account...")

    # 1. Ensure Mallanna shop exists (do NOT auto-create fake shop data)
    shop_res = await db.execute(
        select(Shop).where(
            (Shop.shop_name.ilike(f"%{MALLANNA_SHOP_NAME}%")) | (Shop.phone_number == MALLANNA_PHONE)
        )
    )
    shop = shop_res.scalar_one_or_none()

    if not shop:
        logger.error(
            "Expected demo shop '%s' (phone: %s) was not found in the database. Aborting demo shop owner seeding.",
            MALLANNA_SHOP_NAME,
            MALLANNA_PHONE,
        )
        return None

    # 2. Check for existing demo user account strictly by email
    user_res = await db.execute(
        select(UserAccount).where(UserAccount.email == DEMO_SHOP_OWNER_EMAIL)
    )
    existing_user = user_res.scalar_one_or_none()

    custom_password = (settings.demo_shop_owner_password or "").strip()

    if not existing_user:
        # Check if shop is already associated with another account
        shop_user_res = await db.execute(
            select(UserAccount).where(UserAccount.shop_id == shop.id)
        )
        shop_user = shop_user_res.scalar_one_or_none()
        if shop_user:
            logger.error(
                "Demo shop '%s' (%s) is already associated with another user ('%s'). Aborting demo shop owner seeding without modification.",
                shop.shop_name,
                shop.id,
                shop_user.email,
            )
            return None

        raw_password = custom_password or _generate_secure_password()
        hashed_pw = hash_password(raw_password)

        new_user = UserAccount(
            email=DEMO_SHOP_OWNER_EMAIL,
            password_hash=hashed_pw,
            role=UserRole.SHOP_OWNER.value,
            shop_id=shop.id,
            is_active=True,
        )
        db.add(new_user)
        await db.commit()
        logger.info(
            "Demo Agri Shop Owner account successfully created for '%s' linked to shop '%s' (role: %s).",
            DEMO_SHOP_OWNER_EMAIL,
            shop.shop_name,
            new_user.role,
        )
        return new_user
    else:
        # Existing user by email found.
        # Fail safely if it belongs to a different role or shop association.
        if existing_user.role != UserRole.SHOP_OWNER.value or existing_user.shop_id != shop.id:
            logger.error(
                "Account '%s' already exists with conflicting role ('%s') or shop association (%s). Aborting demo shop owner seeding without modification.",
                DEMO_SHOP_OWNER_EMAIL,
                existing_user.role,
                existing_user.shop_id,
            )
            return None

        # Existing intended demo account: preserve role and shop association.
        existing_user.is_active = True

        # Only update password if explicitly supplied via DEMO_SHOP_OWNER_PASSWORD
        if custom_password:
            existing_user.password_hash = hash_password(custom_password)
            logger.info("Demo Agri Shop Owner password updated from environment configuration.")
        else:
            logger.info("Demo Agri Shop Owner existing password hash preserved.")

        db.add(existing_user)
        await db.commit()
        logger.info(
            "Demo Agri Shop Owner account verified/updated for '%s' (role: %s).",
            DEMO_SHOP_OWNER_EMAIL,
            existing_user.role,
        )
        return existing_user
