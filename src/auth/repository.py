"""
Repository for UserAccount database operations.
"""

from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import UserAccount, Expert, Shop


class AuthRepository:
    """Handles database persistence for user accounts and associated entity checks."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: UUID) -> Optional[UserAccount]:
        """Fetch user account by primary key."""
        stmt = select(UserAccount).where(UserAccount.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[UserAccount]:
        """Fetch user account by email (case-insensitive)."""
        stmt = select(UserAccount).where(UserAccount.email == email.lower().strip())
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_expert_id(self, expert_id: UUID) -> Optional[UserAccount]:
        """Fetch user account linked to an Expert ID."""
        stmt = select(UserAccount).where(UserAccount.expert_id == expert_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_shop_id(self, shop_id: UUID) -> Optional[UserAccount]:
        """Fetch user account linked to a Shop ID."""
        stmt = select(UserAccount).where(UserAccount.shop_id == shop_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def check_expert_exists(self, expert_id: UUID) -> bool:
        """Verify whether an Expert with the given ID exists in the database."""
        stmt = select(Expert.id).where(Expert.id == expert_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def check_shop_exists(self, shop_id: UUID) -> bool:
        """Verify whether a Shop with the given ID exists in the database."""
        stmt = select(Shop.id).where(Shop.id == shop_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def create_user(self, user: UserAccount) -> UserAccount:
        """Save a new user account."""
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
