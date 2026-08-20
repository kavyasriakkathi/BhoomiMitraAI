from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.orders.repository import OrderRepository
from src.orders.service import OrderService


def get_order_repository(db: AsyncSession = Depends(get_db)) -> OrderRepository:
    return OrderRepository(db)


def get_order_service(
    repo: OrderRepository = Depends(get_order_repository),
) -> OrderService:
    return OrderService(repo)
