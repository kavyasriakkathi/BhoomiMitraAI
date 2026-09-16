from typing import Optional, List
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from src.core.models import FarmerPushToken, Farmer
from src.core.logging import logger
from src.notifications.schemas import mask_token


class PushTokenRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def register_or_update_token(
        self,
        farmer_id: UUID,
        token: str,
        platform: str = "android",
        device_model: Optional[str] = None,
    ) -> FarmerPushToken:
        """
        Register a new FCM token or update existing token association.
        Handles device transfers (if token previously belonged to another farmer)
        and token reactivations.
        """
        clean_token = token.strip()
        stmt = select(FarmerPushToken).where(FarmerPushToken.token == clean_token)
        res = await self.db.execute(stmt)
        existing = res.scalar_one_or_none()

        now = datetime.utcnow()
        if existing:
            existing.farmer_id = farmer_id
            existing.platform = platform
            existing.device_model = device_model
            existing.is_active = True
            existing.updated_at = now
            self.db.add(existing)
            await self.db.commit()
            await self.db.refresh(existing)
            logger.info(
                f"[PUSH TOKEN] Updated FCM token {mask_token(clean_token)} for farmer {farmer_id} (active=True)."
            )
            return existing

        new_token = FarmerPushToken(
            farmer_id=farmer_id,
            token=clean_token,
            platform=platform,
            device_model=device_model,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self.db.add(new_token)
        await self.db.commit()
        await self.db.refresh(new_token)
        logger.info(
            f"[PUSH TOKEN] Registered new FCM token {mask_token(clean_token)} for farmer {farmer_id}."
        )
        return new_token

    async def deactivate_token(self, token: str) -> bool:
        """Deactivate a single FCM token (e.g. on user logout/permission revoke)."""
        clean_token = token.strip()
        stmt = select(FarmerPushToken).where(FarmerPushToken.token == clean_token)
        res = await self.db.execute(stmt)
        record = res.scalar_one_or_none()
        if record and record.is_active:
            record.is_active = False
            record.updated_at = datetime.utcnow()
            self.db.add(record)
            await self.db.commit()
            logger.info(f"[PUSH TOKEN] Deactivated FCM token {mask_token(clean_token)}.")
            return True
        return False

    async def deactivate_tokens_batch(self, tokens: List[str]) -> int:
        """Batch deactivate invalid or unregistered tokens reported by Firebase."""
        if not tokens:
            return 0
        clean_tokens = [t.strip() for t in tokens if t and t.strip()]
        if not clean_tokens:
            return 0

        stmt = (
            update(FarmerPushToken)
            .where(FarmerPushToken.token.in_(clean_tokens))
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        res = await self.db.execute(stmt)
        await self.db.commit()
        count = res.rowcount
        logger.info(f"[PUSH TOKEN] Batch deactivated {count} invalid/unregistered FCM token(s).")
        return count

    async def get_active_tokens_for_farmer(self, farmer_id: UUID) -> List[FarmerPushToken]:
        """Fetch all currently active push tokens for a given farmer."""
        stmt = (
            select(FarmerPushToken)
            .where(
                FarmerPushToken.farmer_id == farmer_id,
                FarmerPushToken.is_active == True,
            )
            .order_by(FarmerPushToken.updated_at.desc())
        )
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def mark_tokens_used(self, token_ids: List[UUID]) -> None:
        """Update last_used_at timestamp on tokens that received alerts."""
        if not token_ids:
            return
        stmt = (
            update(FarmerPushToken)
            .where(FarmerPushToken.id.in_(token_ids))
            .values(last_used_at=datetime.utcnow())
        )
        await self.db.execute(stmt)
        await self.db.commit()
