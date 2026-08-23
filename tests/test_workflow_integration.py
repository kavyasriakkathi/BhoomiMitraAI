"""
End-to-End Business Workflow Integration Tests for BhoomiMitra AI.

Tests the full lifecycle:
1. Farmer & Farm Creation
2. Inventory Setup
3. Business validation (cross-shop rejection, out-of-stock rejection)
4. Order Lifecycle (Pending -> Accepted -> Ready -> Completed)
5. Inventory Stock Auto-Deduction & Auto-Availability Update
6. Terminal Status Protection (cannot revert Completed / Cancelled orders)
7. Sales Analytics Calculation
"""

from uuid import uuid4
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import AsyncSessionLocal
from src.core.models import Farmer, Shop, Inventory, OrderRequest
from src.orders.repository import OrderRepository
from src.orders.schemas import OrderRequestCreate, OrderRequestUpdateStatus


@pytest.mark.asyncio
async def test_end_to_end_order_and_inventory_lifecycle():
    """Verify order placement, stock deduction on completion, and terminal state locks."""
    async with AsyncSessionLocal() as db:
        # 1. Setup Farmer
        farmer_id = uuid4()
        farmer = Farmer(
            id=farmer_id,
            phone_number=f"+919{str(uuid4().int)[:9]}",
            preferred_language="te",
            is_active=True,
        )
        db.add(farmer)

        # 2. Setup Shop A and Shop B
        shop_a_id = uuid4()
        shop_a = Shop(
            id=shop_a_id,
            shop_name="Rythu Mitra Fertilizers",
            owner_name="Srinivas Rao",
            phone_number=f"+918{str(uuid4().int)[:9]}",
            address="Main Bazar, Warangal",
            latitude=17.9784,
            longitude=79.5941,
            status="active",
        )
        db.add(shop_a)

        shop_b_id = uuid4()
        shop_b = Shop(
            id=shop_b_id,
            shop_name="Kisan Agro Center",
            owner_name="Venkat Reddy",
            phone_number=f"+917{str(uuid4().int)[:9]}",
            address="Bus Stand, Karimnagar",
            latitude=18.4386,
            longitude=79.1288,
            status="active",
        )
        db.add(shop_b)

        # 3. Setup Inventory for Shop A (10 units in stock) and Shop B
        item_a_id = uuid4()
        item_a = Inventory(
            id=item_a_id,
            shop_id=shop_a_id,
            product_name="Urea Fertilizer 45kg",
            category="Fertilizers",
            brand="IFFCO",
            unit="Bag",
            price=268.0,
            quantity_in_stock=10,
            minimum_stock_level=2,
            available=True,
        )
        db.add(item_a)

        item_b_id = uuid4()
        item_b = Inventory(
            id=item_b_id,
            shop_id=shop_b_id,
            product_name="DAP Fertilizer 50kg",
            category="Fertilizers",
            brand="Coromandel",
            unit="Bag",
            price=1350.0,
            quantity_in_stock=5,
            minimum_stock_level=1,
            available=True,
        )
        db.add(item_b)

        await db.commit()

        order_repo = OrderRepository(db)

        # 4. Business Rule Test: Cross-shop item rejection
        with pytest.raises(ValueError, match="does not belong to shop"):
            await order_repo.create(OrderRequestCreate(
                farmer_id=farmer_id,
                shop_id=shop_a_id,
                inventory_id=item_b_id,  # item belongs to Shop B!
                quantity=2,
            ))

        # 5. Business Rule Test: Over-quantity rejection
        with pytest.raises(ValueError, match="exceeds available stock"):
            await order_repo.create(OrderRequestCreate(
                farmer_id=farmer_id,
                shop_id=shop_a_id,
                inventory_id=item_a_id,
                quantity=15,  # Only 10 available
            ))

        # 6. Valid Order Placement: Order 10 units (will empty the stock upon completion)
        order = await order_repo.create(OrderRequestCreate(
            farmer_id=farmer_id,
            shop_id=shop_a_id,
            inventory_id=item_a_id,
            quantity=10,
            payment_method="COD",
            notes="Deliver near village temple",
        ))

        assert order.status == "Pending"
        assert order.total_price == 2680.0
        assert order.product_name == "Urea Fertilizer 45kg"

        # 7. Lifecycle transitions: Pending -> Accepted -> Ready -> Completed
        await order_repo.update_status(order.id, OrderRequestUpdateStatus(status="Accepted"))
        await order_repo.update_status(order.id, OrderRequestUpdateStatus(status="Ready"))
        completed_order = await order_repo.update_status(
            order.id, OrderRequestUpdateStatus(status="Completed", notes="Paid in cash")
        )

        assert completed_order.status == "Completed"

        # 8. Verify Inventory Auto-Deduction & Availability Update
        await db.refresh(item_a)
        assert item_a.quantity_in_stock == 0
        assert item_a.available is False  # Automatically disabled when 0 stock

        # 9. Business Rule Test: Terminal state lock (Cannot change completed order)
        with pytest.raises(ValueError, match="Cannot change status"):
            await order_repo.update_status(order.id, OrderRequestUpdateStatus(status="Cancelled"))

        # 10. Verify Sales Analytics reflects completed order revenue
        analytics = await order_repo.get_sales_analytics(shop_a_id)
        assert analytics["completed_orders"] == 1
        assert analytics["total_revenue_inr"] == 2680.0
        assert len(analytics["popular_products"]) > 0
        assert analytics["popular_products"][0]["product_name"] == "Urea Fertilizer 45kg"
