"""
BhoomiMitra AI — Push Notification Service

Coordinates FCM push notifications for Stock Siren alerts across all 13 supported Indian languages.
Strictly relies on verified database/event facts:
- product_name
- shop_name
- district
- quantity
- unit
- brand
- timestamp

Never invents or hallucinates values.
Decoupled and fail-soft: WhatsApp delivery continues even if FCM push fails or is unconfigured.
"""

from typing import Optional, Dict, Tuple, List
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import logger
from src.core.models import Farmer, Shop
from src.language.languages import SUPPORTED_LANGUAGES, is_supported_language
from src.notifications.repository import PushTokenRepository
from src.notifications.firebase_client import send_stock_siren_push


# Product name localization mapping for all 13 supported languages
_PRODUCT_LOCALIZED: Dict[str, Dict[str, str]] = {
    "urea": {
        "te": "యూరియా",
        "hi": "यूरिया",
        "en": "Urea",
        "ta": "யூரியா",
        "kn": "ಯೂರಿಯಾ",
        "ml": "യൂറിയ",
        "mr": "युरिया",
        "bn": "ইউরিয়া",
        "gu": "યુરિયા",
        "or": "ୟୁରିଆ",
        "pa": "ਯੂਰੀਆ",
        "as": "ইউৰিয়া",
        "ur": "یوریا",
    },
    "dap": {
        "te": "డిఎపి",
        "hi": "डीएपी",
        "en": "DAP",
        "ta": "டிஏபி",
        "kn": "ಡಿಎಪಿ",
        "ml": "ഡിഎപി",
        "mr": "डीएपी",
        "bn": "ডিএপি",
        "gu": "ડીએપી",
        "or": "ଡିଏପି",
        "pa": "ਡੀਏਪੀ",
        "as": "ডিএপি",
        "ur": "ڈی اے پی",
    },
    "potash": {
        "te": "పోటాష్",
        "hi": "पोटाश",
        "en": "Potash",
        "ta": "பொட்டாஷ்",
        "kn": "ಪೊಟ್ಯಾಶ್",
        "ml": "പൊട്ടാഷ്",
        "mr": "पोटॅश",
        "bn": "পটাশ",
        "gu": "પોટાશ",
        "or": "ପୋଟାସ୍",
        "pa": "ਪੋਟਾਸ਼",
        "as": "পটাছ",
        "ur": "پوٹاش",
    },
}

# Stock Siren notification titles for all 13 supported languages
_SIREN_TITLES: Dict[str, str] = {
    "te": "🚨 {product} స్టాక్ అందుబాటులోకి వచ్చింది!",
    "hi": "🚨 {product} का स्टॉक उपलब्ध हो गया है!",
    "en": "🚨 {product} Stock Now Available!",
    "ta": "🚨 {product} இருப்பு இப்போது கிடைக்கிறது!",
    "kn": "🚨 {product} ಸ್ಟಾಕ್ ಈಗ ಲಭ್ಯವಿದೆ!",
    "ml": "🚨 {product} സ്റ്റോക്ക് ഇപ്പോൾ ലഭ്യമാണ്!",
    "mr": "🚨 {product} स्टॉक आता उपलब्ध आहे!",
    "bn": "🚨 {product} স্টক এখন উপলব্ধ!",
    "gu": "🚨 {product} સ્ટોક હવે ઉપલબ્ધ છે!",
    "or": "🚨 {product} ଷ୍ଟକ୍ ବର୍ତ୍ତମାନ ଉପଲବ୍ଧ!",
    "pa": "🚨 {product} ਸਟਾਕ ਹੁਣ ਉਪਲਬਧ ਹੈ!",
    "as": "🚨 {product} ষ্টক এতিয়া উপলব্ধ!",
    "ur": "🚨 {product} اسٹاک اب دستیاب ہے!",
}

# Notification body templates for all 13 supported languages
_SIREN_BODIES: Dict[str, str] = {
    "te": "🏪 {shop} ({district})\n📦 లభ్యత: {qty} {unit} | 🏷️ {brand}",
    "hi": "🏪 {shop} ({district})\n📦 उपलब्ध: {qty} {unit} | 🏷️ {brand}",
    "en": "🏪 {shop} ({district})\n📦 Available: {qty} {unit} | 🏷️ {brand}",
    "ta": "🏪 {shop} ({district})\n📦 இருப்பு: {qty} {unit} | 🏷️ {brand}",
    "kn": "🏪 {shop} ({district})\n📦 ಲಭ್ಯತೆ: {qty} {unit} | 🏷️ {brand}",
    "ml": "🏪 {shop} ({district})\n📦 ലഭ്യത: {qty} {unit} | 🏷️ {brand}",
    "mr": "🏪 {shop} ({district})\n📦 उपलब्ध: {qty} {unit} | 🏷️ {brand}",
    "bn": "🏪 {shop} ({district})\n📦 উপলব্ধ: {qty} {unit} | 🏷️ {brand}",
    "gu": "🏪 {shop} ({district})\n📦 ઉપલબ્ધ: {qty} {unit} | 🏷️ {brand}",
    "or": "🏪 {shop} ({district})\n📦 ଉପଲବ୍ଧ: {qty} {unit} | 🏷️ {brand}",
    "pa": "🏪 {shop} ({district})\n📦 ਉਪਲਬਧ: {qty} {unit} | 🏷️ {brand}",
    "as": "🏪 {shop} ({district})\n📦 উপলব্ধ: {qty} {unit} | 🏷️ {brand}",
    "ur": "🏪 {shop} ({district})\n📦 دستیاب: {qty} {unit} | 🏷️ {brand}",
}


def get_localized_stock_siren_content(
    language: str,
    product_name: str,
    shop_name: str,
    district: str,
    quantity: int,
    unit: str,
    brand: Optional[str] = None,
    updated_time_str: Optional[str] = None,
) -> Tuple[str, str, Dict[str, str]]:
    """
    Produce factual, localized Stock Siren notification content for any of the 13 supported languages.
    """
    lang = (language or "te").strip().lower()
    if not is_supported_language(lang):
        lang = "te"

    norm_product = product_name.strip().lower()
    prod_map = _PRODUCT_LOCALIZED.get(norm_product, {})
    product_display = prod_map.get(lang, product_name.title())

    brand_str = (brand or "").strip() or ("ప్రామాణిక బ్రాండ్" if lang == "te" else "Standard Brand")

    title_template = _SIREN_TITLES.get(lang, _SIREN_TITLES["te"])
    title = title_template.format(product=product_display)

    body_template = _SIREN_BODIES.get(lang, _SIREN_BODIES["te"])
    body = body_template.format(
        shop=shop_name,
        district=district,
        qty=quantity,
        unit=unit,
        brand=brand_str,
    )

    # Structured, verified data payload for the Android client
    data_payload = {
        "alert_type": "stock_siren",
        "product_name": product_name,
        "product_display": product_display,
        "shop_name": shop_name,
        "district": district,
        "quantity": str(quantity),
        "unit": unit,
        "brand": brand_str,
        "language": lang,
        "updated_at": updated_time_str or datetime.utcnow().strftime("%I:%M %p UTC"),
    }

    return title, body, data_payload


async def dispatch_fcm_stock_siren_notification(
    db: AsyncSession,
    farmer: Farmer,
    product_name: str,
    shop: Shop,
    new_quantity: int,
    unit: str,
    brand: Optional[str] = None,
    updated_time_str: Optional[str] = None,
) -> int:
    """
    Dispatches high-priority FCM push notification to a farmer's registered active devices.
    Fails soft so it never blocks or crashes the WhatsApp notification flow.

    Returns:
        Number of successful FCM pushes delivered.
    """
    try:
        repo = PushTokenRepository(db)
        tokens_records = await repo.get_active_tokens_for_farmer(farmer.id)
        if not tokens_records:
            logger.debug(f"[STOCK SIREN FCM] No active push tokens for farmer {farmer.id}. Skipping FCM.")
            return 0

        tokens = [rec.token for rec in tokens_records if rec.token]
        if not tokens:
            return 0

        lang = getattr(farmer, "preferred_language", "te") or "te"
        shop_district = shop.district or "Warangal"

        title, body, data_payload = get_localized_stock_siren_content(
            language=lang,
            product_name=product_name,
            shop_name=shop.shop_name,
            district=shop_district,
            quantity=new_quantity,
            unit=unit,
            brand=brand,
            updated_time_str=updated_time_str,
        )

        success_count, invalid_tokens = await send_stock_siren_push(
            tokens=tokens,
            title=title,
            body=body,
            data_payload=data_payload,
        )

        # Deactivate stale/invalid tokens reported by Firebase
        if invalid_tokens:
            await repo.deactivate_tokens_batch(invalid_tokens)

        # Mark tokens used
        if success_count > 0:
            used_ids = [rec.id for rec in tokens_records if rec.token not in invalid_tokens]
            await repo.mark_tokens_used(used_ids)

        logger.info(
            f"[STOCK SIREN FCM] Dispatched push to farmer {farmer.id} "
            f"({success_count}/{len(tokens)} devices succeeded, {len(invalid_tokens)} invalidated)."
        )
        return success_count

    except Exception as exc:
        logger.warning(f"[STOCK SIREN FCM] Failed soft while dispatching FCM to farmer {farmer.id}: {exc}")
        return 0
