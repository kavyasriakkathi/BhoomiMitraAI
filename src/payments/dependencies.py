"""
FastAPI Dependencies for Payment module.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.payments.service import PaymentService


def get_payment_service(db: AsyncSession = Depends(get_db)) -> PaymentService:
    """Provide a PaymentService instance bound to request DB session."""
    return PaymentService(db)
