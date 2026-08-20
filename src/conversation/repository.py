from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Conversation


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation: Conversation) -> Conversation:
        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)
        return conversation

    async def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_message_id(self, message_id: str) -> Optional[Conversation]:
        """Lookup by Meta message_id for idempotency checks."""
        stmt = select(Conversation).where(Conversation.message_id == message_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_farmer_id(
        self, farmer_id: UUID, skip: int = 0, limit: int = 10
    ) -> Tuple[int, List[Conversation]]:
        """Get paginated conversations for a specific farmer, ordered by newest first."""
        count_stmt = select(func.count(Conversation.id)).where(
            Conversation.farmer_id == farmer_id
        )
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = (
            select(Conversation)
            .where(Conversation.farmer_id == farmer_id)
            .order_by(Conversation.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[Conversation]]:
        """Get paginated conversations across all farmers, ordered by newest first."""
        count_stmt = select(func.count(Conversation.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = (
            select(Conversation)
            .order_by(Conversation.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return total, items

    async def delete(self, conversation: Conversation) -> None:
        await self.session.delete(conversation)
        await self.session.flush()
