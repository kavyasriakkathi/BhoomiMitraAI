from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Conversation, FarmerProfile

class AIRepository:
    """
    Thin orchestration repository for the AI module.
    Retrieves contextual data from existing models without creating a new AI-specific table.
    """
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_conversation_history(
        self, farmer_id: UUID, limit: int = 10
    ) -> List[Conversation]:
        """Fetch recent conversation history for context injection."""
        stmt = (
            select(Conversation)
            .where(Conversation.farmer_id == farmer_id)
            .where(Conversation.ai_response.isnot(None))
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_farmer_profile(self, farmer_id: UUID) -> Optional[FarmerProfile]:
        """Fetch the farmer's profile for context injection."""
        stmt = select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def save_conversation(self, conversation: Conversation) -> Conversation:
        """Save updates to the conversation (like adding the AI response)."""
        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)
        return conversation
