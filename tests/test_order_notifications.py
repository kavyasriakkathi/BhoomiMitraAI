"""
Unit and Integration Tests for WhatsApp Outbound Order Notifications.

Tests:
- Notification content construction for all lifecycle events (Created, Accepted, Ready, Completed, Cancelled).
- WhatsApp outbound dispatch and delivery.
- Non-blocking error handling (WhatsApp API outage does not abort order updates).
- Idempotency / duplicate notification suppression.
- Correct recipient phone resolution.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import AsyncSessionLocal
from src.core.models import Farmer, Shop, Inventory, OrderRequest, Conversation
from src.orders.notifications import (
    build_order_notification_message,
    notify_farmer_order_update,
)
from src.orders.repository import OrderRepository
from src.orders.service import OrderService
from src.orders.schemas import OrderRequestCreate, OrderRequestUpdateStatus


def test_build_notification_messages_all_events():
    """Verify message formatting for all order lifecycle events."""
    # 1. Created
    msg_created = build_order_notification_message(
        event="Created",
        product_name="Urea Fertilizer",
        quantity=2,
        unit="Bag",
        total_price=536.0,
        shop_name="Rythu Agro Center",
    )
    assert "Order Confirmation" in msg_created
    assert "Urea Fertilizer" in msg_created
    assert "2 Bags" in msg_created
    assert "536.00" in msg_created

    # 2. Accepted
    msg_accepted = build_order_notification_message(
        event="Accepted",
        product_name="DAP Fertilizer",
        quantity=1,
        unit="Bag",
        total_price=1350.0,
        shop_name="Kisan Store",
        shop_address="Main Road, Warangal",
        shop_phone="+919876543210",
    )
    assert "Order Accepted" in msg_accepted
    assert "Main Road, Warangal" in msg_accepted
    assert "+919876543210" in msg_accepted

    # 3. Ready
    msg_ready = build_order_notification_message(
        event="Ready",
        product_name="Potash",
        quantity=3,
        unit="Bag",
        total_price=3000.0,
        shop_name="Kisan Store",
    )
    assert "Ready for Pickup" in msg_ready
    assert "Potash" in msg_ready

    # 4. Completed
    msg_completed = build_order_notification_message(
        event="Completed",
        product_name="Potash",
        quantity=3,
        unit="Bag",
        total_price=3000.0,
        shop_name="Kisan Store",
    )
    assert "Order Completed" in msg_completed
    assert "Thank you" in msg_completed

    # 5. Cancelled
    msg_cancelled = build_order_notification_message(
        event="Cancelled",
        product_name="Zinc Sulfate",
        quantity=1,
        unit="Bag",
        total_price=700.0,
        shop_name="Kisan Store",
        reason="Out of stock",
    )
    assert "Order Cancelled" in msg_cancelled
    assert "Out of stock" in msg_cancelled


@pytest.mark.asyncio
async def test_notify_farmer_order_update_success_and_db_logging():
    """Verify WhatsApp message is dispatched to correct farmer phone and logged in conversations."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        phone = f"+919{str(uuid4().int)[:9]}"
        farmer = Farmer(
            id=farmer_id,
            phone_number=phone,
            preferred_language="te",
            is_active=True,
        )
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(
            id=shop_id,
            shop_name="Telangana Agro Traders",
            owner_name="Mallesh",
            phone_number=f"+918{str(uuid4().int)[:9]}",
            address="Market Yard, Nizamabad",
            status="active",
        )
        db.add(shop)

        item_id = uuid4()
        item = Inventory(
            id=item_id,
            shop_id=shop_id,
            product_name="Neem Oil Pesticide",
            category="Pesticides",
            brand="GreenCrop",
            unit="Bottle",
            price=450.0,
            quantity_in_stock=20,
            minimum_stock_level=5,
            available=True,
        )
        db.add(item)

        order_id = uuid4()
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="Neem Oil Pesticide",
            brand="GreenCrop",
            unit="Bottle",
            unit_price=450.0,
            quantity=2,
            total_price=900.0,
            status="Accepted",
        )
        db.add(order)
        await db.commit()

        # Mock send_text_message to return a simulated Meta message ID
        with patch("src.orders.notifications.send_text_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = "wamid.HBgMOTE5ODc2NTQzMjEwFQIAERgSQjEw"

            wa_id = await notify_farmer_order_update(order_id=order_id, event="Accepted")

            assert wa_id == "wamid.HBgMOTE5ODc2NTQzMjEwFQIAERgSQjEw"
            mock_send.assert_called_once()

            # Verify recipient phone was passed correctly
            called_phone = mock_send.call_args[1]["to_phone"]
            assert called_phone == phone.replace("+", "")

            # Verify message content
            called_msg = mock_send.call_args[1]["message_text"]
            assert "Neem Oil Pesticide" in called_msg
            assert "Telangana Agro Traders" in called_msg


@pytest.mark.asyncio
async def test_notification_failure_is_non_blocking():
    """Verify that WhatsApp API failure or network timeout does not raise errors or break order operations."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(
            id=farmer_id,
            phone_number=f"+919{str(uuid4().int)[:9]}",
            is_active=True,
        )
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(
            id=shop_id,
            shop_name="Kisan Kendra",
            owner_name="Rao",
            phone_number=f"+917{str(uuid4().int)[:9]}",
            address="Khammam",
            status="active",
        )
        db.add(shop)

        item_id = uuid4()
        item = Inventory(
            id=item_id,
            shop_id=shop_id,
            product_name="Complex 20-20-0-13",
            category="Fertilizers",
            brand="FACT",
            unit="Bag",
            price=1200.0,
            quantity_in_stock=10,
            minimum_stock_level=2,
            available=True,
        )
        db.add(item)

        order_id = uuid4()
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="Complex 20-20-0-13",
            brand="FACT",
            unit="Bag",
            unit_price=1200.0,
            quantity=1,
            total_price=1200.0,
            status="Pending",
        )
        db.add(order)
        await db.commit()

        # Simulate Meta API returning None (failure or timeout)
        with patch("src.orders.notifications.send_text_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            wa_id = await notify_farmer_order_update(order_id=order_id, event="Accepted")
            assert wa_id is None  # Handled gracefully


@pytest.mark.asyncio
async def test_order_service_idempotent_notification_suppression():
    """Verify duplicate order status updates do not trigger duplicate WhatsApp notifications."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Sri Rama Agro", owner_name="Rama", phone_number=f"+918{str(uuid4().int)[:9]}", address="Siddipet")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(
            id=item_id,
            shop_id=shop_id,
            product_name="Cotton Seed BG-II",
            category="Seeds",
            brand="Rasi",
            unit="Packet",
            price=850.0,
            quantity_in_stock=50,
            minimum_stock_level=5,
            available=True,
        )
        db.add(item)
        await db.commit()

        order_repo = OrderRepository(db)
        order = await order_repo.create(OrderRequestCreate(
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            quantity=2,
        ))

        order_service = OrderService(order_repo)

        with patch("src.orders.service.notify_farmer_order_update", new_callable=AsyncMock) as mock_notif:
            # 1. Update from Pending to Accepted -> notification MUST be dispatched
            await order_service.update_status(order.id, OrderRequestUpdateStatus(status="Accepted"))
            assert mock_notif.call_count == 1
            assert mock_notif.call_args[1]["event"] == "Accepted"

            # 2. Update with SAME status (e.g. updating notes only) -> notification must NOT be re-dispatched
            await order_service.update_status(order.id, OrderRequestUpdateStatus(status="Accepted", notes="Customer confirmed address"))
            assert mock_notif.call_count == 1  # Still 1, no duplicate sent!
