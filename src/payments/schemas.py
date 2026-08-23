"""
Pydantic Schemas for Razorpay Payment Integration.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class CreatePaymentOrderRequest(BaseModel):
    order_id: UUID = Field(..., description="The BhoomiMitra Order UUID to pay for")


class PaymentOrderResponse(BaseModel):
    order_id: UUID
    razorpay_order_id: str
    amount_in_paise: int
    currency: str = "INR"
    key_id: str = Field(..., description="Public Razorpay Key ID for frontend checkout")
    product_name: str
    customer_phone: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class VerifyPaymentRequest(BaseModel):
    order_id: UUID = Field(..., description="The BhoomiMitra Order UUID")
    razorpay_order_id: str = Field(..., description="Razorpay Order ID")
    razorpay_payment_id: str = Field(..., description="Razorpay Payment ID")
    razorpay_signature: str = Field(..., description="Razorpay HMAC-SHA256 signature from checkout")


class VerifyPaymentResponse(BaseModel):
    success: bool
    message: str
    order_id: UUID
    payment_status: str
    paid_at: Optional[datetime] = None


class PaymentWebhookResponse(BaseModel):
    status: str = "ok"
