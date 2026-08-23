"""
BhoomiMitra AI — Razorpay Payment Service.

Handles Razorpay Order Creation, HMAC-SHA256 Signature Verification,
Replay-Attack Protection, Idempotency, and Webhook Processing.
"""

import hmac
import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID
import httpx
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from src.config import get_settings
from src.core.logging import logger
from src.core.models import OrderRequest, Farmer
from src.payments.schemas import (
    PaymentOrderResponse,
    VerifyPaymentRequest,
    VerifyPaymentResponse,
    PaymentWebhookResponse,
)


class PaymentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def create_payment_order(self, order_id: UUID) -> PaymentOrderResponse:
        """Create a Razorpay order for a given BhoomiMitra order."""
        res = await self.db.execute(select(OrderRequest).where(OrderRequest.id == order_id))
        order = res.scalar_one_or_none()
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order '{order_id}' not found."
            )

        if order.payment_status == "Paid":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Order '{order_id}' is already paid."
            )

        if order.status == "Cancelled":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot initiate payment for a cancelled order."
            )

        farmer_res = await self.db.execute(select(Farmer).where(Farmer.id == order.farmer_id))
        farmer = farmer_res.scalar_one_or_none()
        farmer_phone = farmer.phone_number if farmer else None

        amount_in_paise = int(round(order.total_price * 100))

        # Check if active credentials exist; call Razorpay API or generate order
        razorpay_order_id = None
        key_id = self.settings.razorpay_key_id
        key_secret = self.settings.razorpay_key_secret

        if key_id and key_secret and not key_id.startswith("rzp_test_bhoomimitra_mock"):
            try:
                url = "https://api.razorpay.com/v1/orders"
                payload = {
                    "amount": amount_in_paise,
                    "currency": "INR",
                    "receipt": f"order_{str(order.id)[:8]}",
                    "notes": {
                        "order_id": str(order.id),
                        "farmer_id": str(order.farmer_id),
                        "product_name": order.product_name,
                    },
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, auth=(key_id, key_secret), json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    razorpay_order_id = data.get("id")
                else:
                    logger.error(f"[RAZORPAY ERROR] API failed ({resp.status_code}): {resp.text}")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Payment gateway communication failed. Please try again.",
                    )
            except httpx.RequestError as err:
                logger.exception(f"[RAZORPAY GATEWAY ERROR] Network error contacting Razorpay: {err}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Could not reach payment gateway. Please check connection.",
                )
        else:
            # Deterministic mock/test order identifier for local dev/testing
            razorpay_order_id = f"order_mock_{str(order.id).replace('-', '')[:14]}"

        # Record razorpay_order_id in order record
        order.razorpay_order_id = razorpay_order_id
        order.updated_at = datetime.utcnow()
        self.db.add(order)
        await self.db.commit()
        await self.db.refresh(order)

        logger.info(f"[PAYMENT ORDER CREATED] Order {order_id} bound to Razorpay Order {razorpay_order_id} (₹{order.total_price})")

        return PaymentOrderResponse(
            order_id=order.id,
            razorpay_order_id=razorpay_order_id,
            amount_in_paise=amount_in_paise,
            currency="INR",
            key_id=key_id,
            product_name=order.product_name,
            customer_phone=farmer_phone,
        )

    async def verify_payment(self, data: VerifyPaymentRequest) -> VerifyPaymentResponse:
        """Verify Razorpay HMAC-SHA256 signature and record payment."""
        res = await self.db.execute(select(OrderRequest).where(OrderRequest.id == data.order_id))
        order = res.scalar_one_or_none()
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order '{data.order_id}' not found."
            )

        # Idempotency check: If order is already Paid with the exact same payment_id, return success
        if order.payment_status == "Paid":
            if order.razorpay_payment_id == data.razorpay_payment_id:
                return VerifyPaymentResponse(
                    success=True,
                    message="Payment already verified.",
                    order_id=order.id,
                    payment_status="Paid",
                    paid_at=order.paid_at,
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This order has already been paid.",
            )

        # Security Check: Ensure order_id is bound to the correct razorpay_order_id
        if order.razorpay_order_id and order.razorpay_order_id != data.razorpay_order_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Razorpay Order ID does not match the order record.",
            )

        # Security Check: Replay Attack Prevention (Ensure payment_id is not already used on another order)
        dup_check = await self.db.execute(
            select(OrderRequest).where(
                and_(
                    OrderRequest.razorpay_payment_id == data.razorpay_payment_id,
                    OrderRequest.id != data.order_id,
                )
            )
        )
        if dup_check.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment transaction has already been applied to a different order.",
            )

        # Security Check: Cryptographic HMAC-SHA256 Signature Verification
        key_secret = self.settings.razorpay_key_secret
        msg = f"{data.razorpay_order_id}|{data.razorpay_payment_id}".encode("utf-8")
        expected_sig = hmac.new(key_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_sig, data.razorpay_signature):
            logger.warning(
                f"[PAYMENT VERIFICATION FAILED] Signature mismatch for Order {data.order_id}. "
                f"Expected {expected_sig[:8]}..., received {data.razorpay_signature[:8]}..."
            )
            order.payment_status = "Failed"
            order.updated_at = datetime.utcnow()
            self.db.add(order)
            await self.db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment signature verification failed.",
            )

        # Mark payment as Paid
        paid_time = datetime.utcnow()
        order.payment_status = "Paid"
        order.razorpay_order_id = data.razorpay_order_id
        order.razorpay_payment_id = data.razorpay_payment_id
        order.razorpay_signature = data.razorpay_signature
        order.paid_at = paid_time
        order.updated_at = paid_time

        self.db.add(order)
        await self.db.commit()
        await self.db.refresh(order)

        logger.info(f"[PAYMENT VERIFIED SUCCESS] Order {order.id} marked as Paid (Payment ID: {data.razorpay_payment_id})")

        return VerifyPaymentResponse(
            success=True,
            message="Payment verified successfully.",
            order_id=order.id,
            payment_status="Paid",
            paid_at=paid_time,
        )

    async def handle_webhook(self, raw_body: bytes, signature: Optional[str]) -> PaymentWebhookResponse:
        """Process incoming Razorpay Webhook events securely and idempotently."""
        if not signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing X-Razorpay-Signature header.",
            )

        secret = self.settings.razorpay_webhook_secret or self.settings.razorpay_key_secret
        expected_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_sig, signature):
            logger.warning("[PAYMENT WEBHOOK ERROR] Invalid Razorpay webhook signature.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook signature.",
            )

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload.",
            )

        event = payload.get("event")
        entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        rzp_order_id = entity.get("order_id")
        rzp_payment_id = entity.get("id")

        if not rzp_order_id:
            rzp_order_id = payload.get("payload", {}).get("order", {}).get("entity", {}).get("id")

        logger.info(f"[PAYMENT WEBHOOK RECEIVED] Event: '{event}', Razorpay Order: '{rzp_order_id}'")

        if rzp_order_id:
            res = await self.db.execute(
                select(OrderRequest).where(OrderRequest.razorpay_order_id == rzp_order_id)
            )
            order = res.scalar_one_or_none()
            if order:
                if event in ["payment.captured", "order.paid"] and order.payment_status != "Paid":
                    order.payment_status = "Paid"
                    if rzp_payment_id:
                        order.razorpay_payment_id = rzp_payment_id
                    order.paid_at = datetime.utcnow()
                    order.updated_at = datetime.utcnow()
                    self.db.add(order)
                    await self.db.commit()
                    logger.info(f"[PAYMENT WEBHOOK APPLIED] Order {order.id} marked as Paid via webhook.")
                elif event == "payment.failed" and order.payment_status == "Pending":
                    order.payment_status = "Failed"
                    order.updated_at = datetime.utcnow()
                    self.db.add(order)
                    await self.db.commit()
                    logger.info(f"[PAYMENT WEBHOOK APPLIED] Order {order.id} marked as Failed via webhook.")

        return PaymentWebhookResponse(status="ok")
