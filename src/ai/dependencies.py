from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.ai.repository import AIRepository
from src.ai.service import AIService

def get_ai_repository(session: AsyncSession = Depends(get_db)) -> AIRepository:
    return AIRepository(session)

def get_ai_service(repository: AIRepository = Depends(get_ai_repository)) -> AIService:
    return AIService(repository)
