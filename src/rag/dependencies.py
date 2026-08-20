from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.rag.repository import RAGRepository
from src.rag.service import RAGService


def get_rag_repository(db: AsyncSession = Depends(get_db)) -> RAGRepository:
    return RAGRepository(db)


def get_rag_service(
    repo: RAGRepository = Depends(get_rag_repository),
) -> RAGService:
    return RAGService(repo)
