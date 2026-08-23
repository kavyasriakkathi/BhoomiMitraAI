"""
WhatsApp Outbound Order Notifications.

Handles asynchronous, idempotent delivery of order status updates to farmers via WhatsApp.
Reuses existing WhatsApp Cloud API client infrastructure from src/gateway/whatsapp_client.py.
"""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import OrderRequest, Farmer, Shop, Conversation
from src.core.database import AsyncSessionLocal
from src.gateway.whatsapp_client import send_text_message
from src.core.logging import logger


def build_order_notification_message(
    event: str,
    product_name: str,
    quantity: int,
    unit: str,
    total_price: float,
    shop_name: str,
    shop_address: Optional[str] = None,
    shop_phone: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """Build clear, farmer-friendly WhatsApp message for order events."""
    if event in ["Created", "Pending"]:
        return (
            f"🌾 *BhoomiMitra Order Confirmation*\n"
            f"Your order request has been submitted to the shop owner!\n\n"
            f"📦 *Product:* {product_name}\n"
            f"🔢 *Quantity:* {quantity} {unit}s\n"
            f"💰 *Total:* ₹{total_price:.2f}\n"
            f"🏬 *Shop:* {shop_name}\n"
            f"⏳ *Status:* Pending Confirmation"
        )
    elif event == "Accepted":
        shop_info = f"\n📍 *Address:* {shop_address}" if shop_address else ""
        phone_info = f"\n📞 *Shop Contact:* {shop_phone}" if shop_phone else ""
        return (
            f"🌾 *BhoomiMitra Order Accepted*\n"
            f"Your order has been accepted by the shop owner!\n\n"
            f"📦 *Product:* {product_name}\n"
            f"🔢 *Quantity:* {quantity} {unit}s\n"
            f"💰 *Total:* ₹{total_price:.2f}\n"
            f"🏬 *Shop:* {shop_name}{shop_info}{phone_info}\n"
            f"⏳ *Status:* Accepted (Stock is being prepared)"
        )
    elif event == "Ready":
        shop_info = f"\n📍 *Address:* {shop_address}" if shop_address else ""
        return (
            f"🌾 *Your Order is Ready for Pickup!*\n"
            f"Please visit the shop to collect your items.\n\n"
            f"🏬 *Shop:* {shop_name}{shop_info}\n"
            f"📦 *Product:* {product_name}\n"
            f"🔢 *Quantity:* {quantity} {unit}s\n"
            f"💰 *Amount to Pay:* ₹{total_price:.2f}"
        )
    elif event == "Completed":
        return (
            f"🌾 *BhoomiMitra Order Completed*\n"
            f"Thank you for purchasing with BhoomiMitra AI!\n\n"
            f"📦 *Product:* {product_name} ({quantity} {unit}s)\n"
            f"💰 *Paid:* ₹{total_price:.2f}\n"
            f"🏬 *Shop:* {shop_name}\n"
            f"✅ *Status:* Completed"
        )
    elif event == "Cancelled":
        reason_text = f"\n📝 *Reason:* {reason}" if reason else "\n📝 *Reason:* Stock unavailable or cancelled by shop"
        return (
            f"⚠️ *BhoomiMitra Order Cancelled*\n"
            f"Your purchase request was cancelled.\n\n"
            f"📦 *Order:* {product_name} ({quantity} {unit}s)\n"
            f"🏬 *Shop:* {shop_name}{reason_text}"
        )
    return f"🌾 BhoomiMitra Order Update: Your order for {product_name} status is now {event}."


async def notify_farmer_order_update(
    order_id: UUID,
    event: str,
    reason: Optional[str] = None,
) -> Optional[str]:
    """
    Send an asynchronous WhatsApp notification to the farmer regarding order status changes.
    Non-blocking: delivery failures do not raise exceptions or abort database transactions.
    """
    try:
        async with AsyncSessionLocal() as db:
            # 1. Fetch Order with associated Farmer and Shop
            res = await db.execute(select(OrderRequest).where(OrderRequest.id == order_id))
            order = res.scalar_one_or_none()
            if not order:
                logger.warning(f"[ORDER NOTIFICATION] Order '{order_id}' not found.")
                return None

            farmer_res = await db.execute(select(Farmer).where(Farmer.id == order.farmer_id))
            farmer = farmer_res.scalar_one_or_none()
            if not farmer or not farmer.phone_number:
                logger.warning(f"[ORDER NOTIFICATION] Farmer or phone not found for order '{order_id}'.")
                return None

            shop_res = await db.execute(select(Shop).where(Shop.id == order.shop_id))
            shop = shop_res.scalar_one_or_none()
            shop_name = shop.shop_name if shop else "Agri Shop"
            shop_address = shop.address if shop else None
            shop_phone = shop.phone_number if shop else None

            # 2. Build message text
            msg_text = build_order_notification_message(
                event=event,
                product_name=order.product_name,
                quantity=order.quantity,
                unit=order.unit,
                total_price=order.total_price,
                shop_name=shop_name,
                shop_address=shop_address,
                shop_phone=shop_phone,
                reason=reason,
            )

            # Format phone number for Meta WhatsApp Cloud API (strip '+' and non-digits)
            raw_phone = farmer.phone_number.replace("+", "").replace(" ", "").replace("-", "").strip()

            # 3. Send WhatsApp message via existing client
            wa_msg_id = await send_text_message(to_phone=raw_phone, message_text=msg_text)

            # 4. Log in conversation table if delivered
            if wa_msg_id:
                conv = Conversation(
                    farmer_id=farmer.id,
                    message_id=f"order_notif_{order.id}_{event.lower()}_{wa_msg_id[-8:]}",
                    user_message=f"[Automated Order Notification: {event}]",
                    user_message_type="text",
                    ai_response=msg_text,
                    intent="order_status_notification",
                    outbound_message_id=wa_msg_id,
                    delivery_status="sent",
                )
                db.add(conv)
                await db.commit()
                logger.info(f"[ORDER NOTIFICATION DELIVERED] Order {order_id} ({event}) to {raw_phone} (wa_id={wa_msg_id})")

            return wa_msg_id

    except Exception as err:
        logger.exception(f"[ORDER NOTIFICATION ERROR] Non-blocking failure notifying farmer for order '{order_id}': {err}")
        return None
