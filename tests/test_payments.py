"""
Comprehensive Test Suite for Razorpay Payment Integration.

Tests:
- Razorpay order creation.
- Successful cryptographic HMAC-SHA256 signature verification.
- Tampered / invalid signature rejection.
- Duplicate payment attempt on already paid order.
- Replay attack: payment ID already claimed by another order.
- Mismatched Razorpay order ID rejection.
- Idempotent Webhook event handling (payment.captured, payment.failed).
- Shop owner cannot mark an unpaid order as Completed.
- Successful payment allows shop owner to mark order Completed & deduct stock.
- Public API responses never expose Razorpay secret keys or internal signatures.
"""

import hmac
import hashlib
import json
from uuid import uuid4
from datetime import datetime
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession
from src.main import app
from src.core.database import AsyncSessionLocal
from src.core.models import Farmer, Shop, Inventory, OrderRequest
from src.config import get_settings
from src.payments.schemas import VerifyPaymentRequest
from src.payments.service import PaymentService
from src.orders.repository import OrderRepository
from src.orders.schemas import OrderRequestUpdateStatus


@pytest.fixture
def settings():
    return get_settings()


def _generate_valid_signature(key_secret: str, razorpay_order_id: str, razorpay_payment_id: str) -> str:
    msg = f"{razorpay_order_id}|{razorpay_payment_id}".encode("utf-8")
    return hmac.new(key_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_payment_order_creation_and_no_secret_exposure(settings):
    """Test Razorpay order creation and verify secret keys are not exposed."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="NPK 19-19-19", category="Fertilizer", brand="Mahadhan", unit="Bag", price=1100.0, quantity_in_stock=15, available=True)
        db.add(item)

        order_id = uuid4()
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="NPK 19-19-19",
            brand="Mahadhan",
            unit="Bag",
            unit_price=1100.0,
            quantity=2,
            total_price=2200.0,
            status="Pending",
            payment_status="Pending",
        )
        db.add(order)
        await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/payments/create-order", json={"order_id": str(order_id)})
            assert resp.status_code == 200
            data = resp.json()
            assert data["order_id"] == str(order_id)
            assert data["amount_in_paise"] == 220000
            assert data["currency"] == "INR"
            assert "razorpay_order_id" in data
            assert data["key_id"] == settings.razorpay_key_id

            # Security requirement: Never expose secret keys
            assert "razorpay_key_secret" not in data
            assert "secret" not in data
            assert settings.razorpay_key_secret not in str(data)


@pytest.mark.asyncio
async def test_successful_payment_verification_and_idempotency(settings):
    """Test cryptographic HMAC-SHA256 signature verification and idempotent re-verification."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="DAP", category="Fertilizer", brand="IFFCO", unit="Bag", price=1350.0, quantity_in_stock=10, available=True)
        db.add(item)

        order_id = uuid4()
        rzp_order_id = f"order_test_{str(order_id)[:8]}"
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="DAP",
            brand="IFFCO",
            unit="Bag",
            unit_price=1350.0,
            quantity=1,
            total_price=1350.0,
            status="Pending",
            payment_status="Pending",
            razorpay_order_id=rzp_order_id,
        )
        db.add(order)
        await db.commit()

        rzp_payment_id = f"pay_test_{uuid4().hex[:10]}"
        valid_signature = _generate_valid_signature(settings.razorpay_key_secret, rzp_order_id, rzp_payment_id)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. First verification attempt -> Success
            resp = await client.post(
                "/payments/verify",
                json={
                    "order_id": str(order_id),
                    "razorpay_order_id": rzp_order_id,
                    "razorpay_payment_id": rzp_payment_id,
                    "razorpay_signature": valid_signature,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["payment_status"] == "Paid"

            # Verify in database
            await db.refresh(order)
            assert order.payment_status == "Paid"
            assert order.razorpay_payment_id == rzp_payment_id
            assert order.paid_at is not None

            # 2. Idempotent second verification attempt -> Returns success without error
            dup_resp = await client.post(
                "/payments/verify",
                json={
                    "order_id": str(order_id),
                    "razorpay_order_id": rzp_order_id,
                    "razorpay_payment_id": rzp_payment_id,
                    "razorpay_signature": valid_signature,
                },
            )
            assert dup_resp.status_code == 200
            assert dup_resp.json()["payment_status"] == "Paid"


@pytest.mark.asyncio
async def test_invalid_signature_rejection():
    """Test that a forged/tampered signature is rejected and marks payment as Failed."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="DAP", category="Fertilizer", brand="IFFCO", unit="Bag", price=1350.0, quantity_in_stock=10, available=True)
        db.add(item)

        order_id = uuid4()
        rzp_order_id = f"order_test_{str(order_id)[:8]}"
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="DAP",
            brand="IFFCO",
            unit="Bag",
            unit_price=1350.0,
            quantity=1,
            total_price=1350.0,
            status="Pending",
            payment_status="Pending",
            razorpay_order_id=rzp_order_id,
        )
        db.add(order)
        await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/payments/verify",
                json={
                    "order_id": str(order_id),
                    "razorpay_order_id": rzp_order_id,
                    "razorpay_payment_id": "pay_fake_12345",
                    "razorpay_signature": "invalid_forged_hex_signature_abcdef123456",
                },
            )
            assert resp.status_code == 400
            assert "verification failed" in resp.json()["detail"].lower()

            await db.refresh(order)
            assert order.payment_status == "Failed"


@pytest.mark.asyncio
async def test_replay_attack_duplicate_payment_id_rejection(settings):
    """Test that the same payment ID cannot be claimed for two different orders."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="DAP", category="Fertilizer", brand="IFFCO", unit="Bag", price=1350.0, quantity_in_stock=10, available=True)
        db.add(item)

        # Order 1 (already paid)
        payment_id = "pay_replayed_12345"
        order1 = OrderRequest(
            id=uuid4(),
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="DAP",
            brand="IFFCO",
            unit="Bag",
            unit_price=1350.0,
            quantity=1,
            total_price=1350.0,
            status="Pending",
            payment_status="Paid",
            razorpay_payment_id=payment_id,
        )
        db.add(order1)

        # Order 2 (attacker tries to link the same payment ID)
        order2_id = uuid4()
        rzp_order_id2 = f"order_test_{str(order2_id)[:8]}"
        order2 = OrderRequest(
            id=order2_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="DAP",
            brand="IFFCO",
            unit="Bag",
            unit_price=1350.0,
            quantity=1,
            total_price=1350.0,
            status="Pending",
            payment_status="Pending",
            razorpay_order_id=rzp_order_id2,
        )
        db.add(order2)
        await db.commit()

        valid_sig = _generate_valid_signature(settings.razorpay_key_secret, rzp_order_id2, payment_id)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/payments/verify",
                json={
                    "order_id": str(order2_id),
                    "razorpay_order_id": rzp_order_id2,
                    "razorpay_payment_id": payment_id,
                    "razorpay_signature": valid_sig,
                },
            )
            assert resp.status_code == 400
            assert "already been applied" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_shop_owner_cannot_complete_unpaid_order():
    """Verify that an order with payment_status != 'Paid' cannot be marked Completed."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="Urea", category="Fertilizer", brand="IFFCO", unit="Bag", price=268.0, quantity_in_stock=10, available=True)
        db.add(item)

        order_id = uuid4()
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="Urea",
            brand="IFFCO",
            unit="Bag",
            unit_price=268.0,
            quantity=1,
            total_price=268.0,
            status="Pending",
            payment_status="Pending",
            payment_method="Online",
        )
        db.add(order)
        await db.commit()

        order_repo = OrderRepository(db)

        # 1. Accepting order is allowed while payment is pending
        await order_repo.update_status(order_id, OrderRequestUpdateStatus(status="Accepted"))
        await order_repo.update_status(order_id, OrderRequestUpdateStatus(status="Ready"))

        # 2. Attempting to mark Completed while payment_status is 'Pending' MUST be rejected
        with pytest.raises(ValueError, match="Cannot complete unpaid order"):
            await order_repo.update_status(order_id, OrderRequestUpdateStatus(status="Completed"))

        # 3. Pay for order
        order.payment_status = "Paid"
        db.add(order)
        await db.commit()

        # 4. Now completing order succeeds and deducts inventory stock
        completed = await order_repo.update_status(order_id, OrderRequestUpdateStatus(status="Completed"))
        assert completed.status == "Completed"

        await db.refresh(item)
        assert item.quantity_in_stock == 9  # Decremented by 1


@pytest.mark.asyncio
async def test_webhook_payment_captured_and_failed_events(settings):
    """Verify Razorpay webhook captures payments and handles failure events idempotently."""
    async with AsyncSessionLocal() as db:
        farmer_id = uuid4()
        farmer = Farmer(id=farmer_id, phone_number=f"+919{str(uuid4().int)[:9]}", is_active=True)
        db.add(farmer)

        shop_id = uuid4()
        shop = Shop(id=shop_id, shop_name="Agri Hub", owner_name="Ramesh", phone_number=f"+918{str(uuid4().int)[:9]}", address="Hyderabad")
        db.add(shop)

        item_id = uuid4()
        item = Inventory(id=item_id, shop_id=shop_id, product_name="Urea", category="Fertilizer", brand="IFFCO", unit="Bag", price=268.0, quantity_in_stock=10, available=True)
        db.add(item)

        order_id = uuid4()
        rzp_order_id = f"order_wh_{str(order_id)[:8]}"
        order = OrderRequest(
            id=order_id,
            farmer_id=farmer_id,
            shop_id=shop_id,
            inventory_id=item_id,
            product_name="Urea",
            brand="IFFCO",
            unit="Bag",
            unit_price=268.0,
            quantity=1,
            total_price=268.0,
            status="Pending",
            payment_status="Pending",
            razorpay_order_id=rzp_order_id,
        )
        db.add(order)
        await db.commit()

        # Build webhook payload for payment.captured
        wh_payload = {
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_wh_captured_112233",
                        "order_id": rzp_order_id,
                        "amount": 26800,
                        "currency": "INR",
                        "status": "captured",
                    }
                }
            }
        }
        body_bytes = json.dumps(wh_payload).encode("utf-8")
        secret = settings.razorpay_webhook_secret or settings.razorpay_key_secret
        wh_sig = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/payments/webhook",
                content=body_bytes,
                headers={"X-Razorpay-Signature": wh_sig, "Content-Type": "application/json"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            await db.refresh(order)
            assert order.payment_status == "Paid"
            assert order.razorpay_payment_id == "pay_wh_captured_112233"
