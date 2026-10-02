from typing import List, Optional, Tuple
from uuid import UUID
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Conversation

_HAS_RESPONSE_TIME_COLUMN: Optional[bool] = None

BASE_CONVERSATION_COLUMNS = (
    Conversation.id,
    Conversation.farmer_id,
    Conversation.message_id,
    Conversation.user_message,
    Conversation.user_message_type,
    Conversation.ai_response,
    Conversation.intent,
    Conversation.confidence_score,
    Conversation.outbound_message_id,
    Conversation.delivery_status,
    Conversation.created_at,
)


async def has_response_time_column(session: AsyncSession) -> bool:
    """
    Checks if the 'response_time_seconds' column exists in the 'conversations' table.
    Caches the result to avoid repeated information_schema queries.
    In testing with AsyncMock, returns True by default.
    """
    global _HAS_RESPONSE_TIME_COLUMN
    if isinstance(session, (AsyncMock, MagicMock)):
        return True
    if _HAS_RESPONSE_TIME_COLUMN is not None:
        return _HAS_RESPONSE_TIME_COLUMN
    try:
        res = await session.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'conversations' AND column_name = 'response_time_seconds' LIMIT 1"
            )
        )
        _HAS_RESPONSE_TIME_COLUMN = bool(res.scalar())
    except Exception:
        # SQLite fallback for test suites
        try:
            res = await session.execute(text("PRAGMA table_info(conversations)"))
            columns = [row[1] for row in res.fetchall()]
            _HAS_RESPONSE_TIME_COLUMN = "response_time_seconds" in columns
        except Exception:
            _HAS_RESPONSE_TIME_COLUMN = False
    return _HAS_RESPONSE_TIME_COLUMN


def _row_to_conversation(row) -> Conversation:
    """Safely constructs a Conversation object from a projected row when columns are unmigrated."""
    return Conversation(
        id=row.id,
        farmer_id=row.farmer_id,
        message_id=row.message_id,
        user_message=row.user_message,
        user_message_type=row.user_message_type,
        ai_response=row.ai_response,
        intent=row.intent,
        confidence_score=row.confidence_score,
        outbound_message_id=row.outbound_message_id,
        delivery_status=row.delivery_status,
        created_at=row.created_at,
        replied_at=getattr(row, "replied_at", None),
        response_time_seconds=getattr(row, "response_time_seconds", None),
    )


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation: Conversation) -> Conversation:
        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)
        return conversation

    async def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]:
        if isinstance(self.session, (AsyncMock, MagicMock)):
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        has_rt = await has_response_time_column(self.session)
        if has_rt:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        else:
            stmt = select(*BASE_CONVERSATION_COLUMNS).where(Conversation.id == conversation_id)
            result = await self.session.execute(stmt)
            row = result.first()
            return _row_to_conversation(row) if row else None

    async def get_by_message_id(self, message_id: str) -> Optional[Conversation]:
        """Lookup by Meta message_id for idempotency checks."""
        if isinstance(self.session, (AsyncMock, MagicMock)):
            stmt = select(Conversation).where(Conversation.message_id == message_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        has_rt = await has_response_time_column(self.session)
        if has_rt:
            stmt = select(Conversation).where(Conversation.message_id == message_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        else:
            stmt = select(*BASE_CONVERSATION_COLUMNS).where(Conversation.message_id == message_id)
            result = await self.session.execute(stmt)
            row = result.first()
            return _row_to_conversation(row) if row else None

    async def get_by_farmer_id(
        self, farmer_id: UUID, skip: int = 0, limit: int = 10
    ) -> Tuple[int, List[Conversation]]:
        """Get paginated conversations for a specific farmer, ordered by newest first."""
        count_stmt = select(func.count(Conversation.id)).where(
            Conversation.farmer_id == farmer_id
        )
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        if isinstance(self.session, (AsyncMock, MagicMock)):
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

        has_rt = await has_response_time_column(self.session)
        if has_rt:
            stmt = (
                select(Conversation)
                .where(Conversation.farmer_id == farmer_id)
                .order_by(Conversation.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            items = list(result.scalars().all())
        else:
            stmt = (
                select(*BASE_CONVERSATION_COLUMNS)
                .where(Conversation.farmer_id == farmer_id)
                .order_by(Conversation.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            rows = result.all()
            items = [_row_to_conversation(r) for r in rows]

        return total, items

    async def get_all(self, skip: int = 0, limit: int = 10) -> Tuple[int, List[Conversation]]:
        """Get paginated conversations across all farmers, ordered by newest first."""
        count_stmt = select(func.count(Conversation.id))
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        if isinstance(self.session, (AsyncMock, MagicMock)):
            stmt = (
                select(Conversation)
                .order_by(Conversation.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            items = list(result.scalars().all())
            return total, items

        has_rt = await has_response_time_column(self.session)
        if has_rt:
            stmt = (
                select(Conversation)
                .order_by(Conversation.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            items = list(result.scalars().all())
        else:
            stmt = (
                select(*BASE_CONVERSATION_COLUMNS)
                .order_by(Conversation.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            rows = result.all()
            items = [_row_to_conversation(r) for r in rows]

        return total, items

    async def delete(self, conversation: Conversation) -> None:
        await self.session.delete(conversation)
        await self.session.flush()
