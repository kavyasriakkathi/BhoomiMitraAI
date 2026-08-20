from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.conversation.repository import ConversationRepository
from src.conversation.service import ConversationService
from src.farmers.repository import FarmerRepository


def get_conversation_repository(session: AsyncSession = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(session)


def get_farmer_repository(session: AsyncSession = Depends(get_db)) -> FarmerRepository:
    return FarmerRepository(session)


def get_conversation_service(
    repository: ConversationRepository = Depends(get_conversation_repository),
    farmer_repository: FarmerRepository = Depends(get_farmer_repository),
) -> ConversationService:
    return ConversationService(repository, farmer_repository)
