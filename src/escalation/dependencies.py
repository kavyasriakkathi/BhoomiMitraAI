from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.escalation.repository import EscalationRepository
from src.escalation.service import EscalationService


def get_escalation_repository(db: AsyncSession = Depends(get_db)) -> EscalationRepository:
    return EscalationRepository(db)


def get_escalation_service(
    repo: EscalationRepository = Depends(get_escalation_repository),
) -> EscalationService:
    return EscalationService(repo)
