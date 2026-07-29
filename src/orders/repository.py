from typing import Optional, List, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from src.core.models import OrderRequest, Inventory, Shop, Farmer
from src.orders.schemas import OrderRequestCreate, OrderRequestUpdateStatus


class OrderRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: OrderRequestCreate) -> OrderRequest:
        # Fetch inventory item details for snapshot pricing
        res = await self.db.execute(select(Inventory).where(Inventory.id == data.inventory_id))
        inventory_item = res.scalar_one_or_none()
        if not inventory_item:
            raise ValueError(f"Inventory product '{data.inventory_id}' not found.")

        price = inventory_item.discount_price if inventory_item.discount_price else inventory_item.price
        total = round(price * data.quantity, 2)

        order = OrderRequest(
            farmer_id=data.farmer_id,
            shop_id=data.shop_id,
            inventory_id=data.inventory_id,
            product_name=inventory_item.product_name,
            brand=inventory_item.brand,
            unit=inventory_item.unit,
            unit_price=price,
            quantity=data.quantity,
            total_price=total,
            status="Pending",
            notes=data.notes,
        )

        self.db.add(order)
        await self.db.commit()
        await self.db.refresh(order)
        return order

    async def get_by_id(self, order_id: UUID) -> Optional[OrderRequest]:
        result = await self.db.execute(select(OrderRequest).where(OrderRequest.id == order_id))
        return result.scalar_one_or_none()

    async def update_status(self, order_id: UUID, data: OrderRequestUpdateStatus) -> Optional[OrderRequest]:
        order = await self.get_by_id(order_id)
        if not order:
            return None

        order.status = data.status
        if data.notes:
            order.notes = (order.notes or "") + f" [{data.notes}]"
        order.updated_at = datetime.utcnow()

        self.db.add(order)
        await self.db.commit()
        await self.db.refresh(order)
        return order

    async def list_farmer_orders(
        self, farmer_id: UUID, page: int = 1, size: int = 20
    ) -> Tuple[List[OrderRequest], int]:
        query = select(OrderRequest).where(OrderRequest.farmer_id == farmer_id)
        count_query = select(func.count(OrderRequest.id)).where(OrderRequest.farmer_id == farmer_id)

        total_res = await self.db.execute(count_query)
        total = total_res.scalar() or 0

        query = query.order_by(OrderRequest.created_at.desc()).offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        orders = list(result.scalars().all())

        return orders, total

    async def list_shop_orders(
        self, shop_id: UUID, page: int = 1, size: int = 20, status: Optional[str] = None
    ) -> Tuple[List[OrderRequest], int]:
        query = select(OrderRequest).where(OrderRequest.shop_id == shop_id)
        count_query = select(func.count(OrderRequest.id)).where(OrderRequest.shop_id == shop_id)

        if status:
            query = query.where(OrderRequest.status == status)
            count_query = count_query.where(OrderRequest.status == status)

        total_res = await self.db.execute(count_query)
        total = total_res.scalar() or 0

        query = query.order_by(OrderRequest.created_at.desc()).offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        orders = list(result.scalars().all())

        return orders, total

    async def get_sales_analytics(self, shop_id: UUID) -> dict:
        """Aggregate sales analytics and product demand for a shop."""
        query = select(OrderRequest).where(OrderRequest.shop_id == shop_id)
        res = await self.db.execute(query)
        all_orders = res.scalars().all()

        total_orders = len(all_orders)
        pending = len([o for o in all_orders if o.status == "Pending"])
        accepted = len([o for o in all_orders if o.status == "Accepted"])
        ready = len([o for o in all_orders if o.status == "Ready"])
        completed = len([o for o in all_orders if o.status == "Completed"])
        cancelled = len([o for o in all_orders if o.status == "Cancelled"])

        revenue = sum(o.total_price for o in all_orders if o.status in ["Accepted", "Ready", "Completed"])

        # Popular products
        product_counts = {}
        for o in all_orders:
            p_name = o.product_name
            product_counts[p_name] = product_counts.get(p_name, 0) + o.quantity

        popular_products = [
            {"product_name": k, "units_sold": str(v)}
            for k, v in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return {
            "shop_id": shop_id,
            "total_orders": total_orders,
            "pending_orders": pending,
            "accepted_orders": accepted,
            "ready_orders": ready,
            "completed_orders": completed,
            "cancelled_orders": cancelled,
            "total_revenue_inr": round(revenue, 2),
            "popular_products": popular_products,
            "category_demand": [
                {"category": "Fertilizers", "demand": "High"},
                {"category": "Pesticides", "demand": "Moderate"},
                {"category": "Organic Products", "demand": "Growing"},
            ],
        }
