"""
FastAPI Router for Razorpay Payment Integration.
"""

from fastapi import APIRouter, Depends, Header, Request, status
from typing import Optional
from src.payments.schemas import (
    CreatePaymentOrderRequest,
    PaymentOrderResponse,
    VerifyPaymentRequest,
    VerifyPaymentResponse,
    PaymentWebhookResponse,
)
from src.payments.service import PaymentService
from src.payments.dependencies import get_payment_service

router = APIRouter()


@router.post(
    "/create-order",
    response_model=PaymentOrderResponse,
    status_code=status.HTTP_200_OK,
    summary="Create Razorpay Payment Order",
    description="Initiates a Razorpay payment order for a given BhoomiMitra order.",
)
async def create_payment_order(
    payload: CreatePaymentOrderRequest,
    service: PaymentService = Depends(get_payment_service),
):
    return await service.create_payment_order(payload.order_id)


@router.post(
    "/verify",
    response_model=VerifyPaymentResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify Razorpay Payment Signature",
    description="Validates cryptographic HMAC-SHA256 signature and records payment status as Paid.",
)
async def verify_payment(
    payload: VerifyPaymentRequest,
    service: PaymentService = Depends(get_payment_service),
):
    return await service.verify_payment(payload)


@router.post(
    "/webhook",
    response_model=PaymentWebhookResponse,
    status_code=status.HTTP_200_OK,
    summary="Razorpay Webhook Listener",
    description="Asynchronously receives and verifies Razorpay webhook events to sync payment statuses.",
)
async def payment_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(None, alias="X-Razorpay-Signature"),
    service: PaymentService = Depends(get_payment_service),
):
    raw_body = await request.body()
    return await service.handle_webhook(raw_body=raw_body, signature=x_razorpay_signature)
