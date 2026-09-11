from __future__ import annotations

"""
BhoomiMitra AI — Proactive Stock Availability Alert Service

Enables farmers to subscribe to out-of-stock input alerts (e.g. Urea, DAP).
When shop owners update stock from 0 to >0, matching active alert subscribers
in that district receive immediate, factual WhatsApp notifications.
"""

from typing import Optional, List, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_

from src.core.logging import logger
from src.core.models import StockAlert, Farmer, Shop, Inventory
from src.shops.service import (
    _PRODUCT_MAPPING,
    _KNOWN_DISTRICTS,
    _extract_district_from_query,
    _resolve_farmer_location,
    _detect_product_from_query,
    enrich_response_with_shops,
)


def _normalize_product_name(query_or_product: str) -> str:
    """Normalize input name to canonical key (e.g. 'యూరియా' -> 'urea')."""
    if not query_or_product:
        return "urea"
    q_lower = query_or_product.strip().lower()
    for kw, norm in _PRODUCT_MAPPING.items():
        if kw in q_lower or kw in query_or_product:
            return norm
    return q_lower


def _canonicalize_district(district_text: Optional[str]) -> Optional[str]:
    """
    Normalizes district names across English, Telugu, and casing/whitespace.
    Uses curated _KNOWN_DISTRICTS mapping (e.g. 'వరంగల్' -> 'Warangal', ' warangal ' -> 'Warangal').
    Returns canonical district name if recognized, or cleaned original string if unknown.
    Does not invent districts.
    """
    if not district_text or not str(district_text).strip():
        return None
    cleaned = str(district_text).strip()
    lower = cleaned.lower()
    if lower in _KNOWN_DISTRICTS:
        return _KNOWN_DISTRICTS[lower]
    if cleaned in _KNOWN_DISTRICTS:
        return _KNOWN_DISTRICTS[cleaned]
    # Check substring match against known districts (e.g. from address string)
    extracted = _extract_district_from_query(cleaned)
    if extracted:
        return extracted
    return cleaned


def _get_district_match_variants(district_text: Optional[str]) -> List[str]:
    """
    Returns all recognized bilingual variants for a district.
    E.g., for 'Warangal' or 'వరంగల్', returns:
    ['Warangal', 'warangal', 'వరంగల్', 'hanamkonda', 'హనుమకొండ', ...]
    """
    if not district_text or not str(district_text).strip():
        return []
    cleaned = str(district_text).strip()
    canon = _canonicalize_district(cleaned) or cleaned
    variants = {canon, canon.lower(), cleaned, cleaned.lower()}
    for kw, target_canon in _KNOWN_DISTRICTS.items():
        if target_canon.lower() == canon.lower():
            variants.add(kw)
            variants.add(target_canon)
    return list(variants)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-action keywords: Cancel / Stop & List Active Alerts
# ─────────────────────────────────────────────────────────────────────────────

CANCEL_KEYWORDS_TE = [
    "అలర్ట్ ఆపండి", "అలర్ట్స్ ఆపండి", "స్టాక్ అలర్ట్ రద్దు", "నోటిఫికేషన్ ఆపండి",
    "అలర్ట్ వద్దు", "అలర్ట్ క్యాన్సిల్", "స్టాక్ అలర్ట్ ఆపండి",
]

CANCEL_KEYWORDS_EN = [
    "stop alert", "stop alerts", "cancel stock alert", "cancel alert", "cancel urea stock alert",
    "stop urea alert", "unsubscribe alert", "disable alert", "turn off alert",
]

CANCEL_KEYWORDS_TANGLISH = [
    "alert aapandi", "alert apandi", "alert vaddu", "alert cancel cheyandi",
    "stop urea alert", "notification aapandi",
]

LIST_KEYWORDS_TE = [
    "నా అలర్ట్స్", "నా స్టాక్ అలర్ట్స్", "యాక్టివ్ అలర్ట్స్", "ఏ అలర్ట్స్ ఉన్నాయి",
    "నా అలర్ట్ లు", "అలర్ట్స్ లిస్ట్",
]

LIST_KEYWORDS_EN = [
    "my alerts", "active alerts", "show my alerts", "what alerts do i have",
    "list my alerts", "my stock alerts",
]

LIST_KEYWORDS_TANGLISH = [
    "naa alerts", "na alerts", "active alerts chupinchandi", "my alerts enti",
]


def detect_stock_alert_action(query_text: str) -> str:
    """
    Returns 'cancel', 'list', or 'subscribe' based on farmer message.
    """
    q_lower = query_text.lower()

    if (
        any(k in query_text for k in CANCEL_KEYWORDS_TE)
        or any(k in q_lower for k in CANCEL_KEYWORDS_EN)
        or any(k in q_lower for k in CANCEL_KEYWORDS_TANGLISH)
    ):
        return "cancel"

    if (
        any(k in query_text for k in LIST_KEYWORDS_TE)
        or any(k in q_lower for k in LIST_KEYWORDS_EN)
        or any(k in q_lower for k in LIST_KEYWORDS_TANGLISH)
    ):
        return "list"

    return "subscribe"


# ─────────────────────────────────────────────────────────────────────────────
# Core Database Operations
# ─────────────────────────────────────────────────────────────────────────────

async def create_or_reactivate_alert(
    db: AsyncSession,
    farmer: Farmer,
    product_name: str,
    district: str,
    state: Optional[str] = None,
) -> Tuple[StockAlert, bool]:
    """
    Create a new stock alert or reactivate an existing inactive alert for
    (farmer_id, product_name, district).

    Returns:
        (StockAlert, is_new_or_reactivated: bool)
    """
    norm_product = _normalize_product_name(product_name)
    clean_dist = district.strip()
    clean_state = state.strip() if state else "Telangana"
    dist_variants = _get_district_match_variants(clean_dist)

    # Check for existing alert for this farmer + product + district (bilingual matching)
    stmt = select(StockAlert).where(
        StockAlert.farmer_id == farmer.id,
        StockAlert.product_name == norm_product,
        or_(*[StockAlert.district.ilike(v) for v in dist_variants]) if dist_variants else StockAlert.district.ilike(clean_dist),
    )
    result = await db.execute(stmt)
    existing_alert = result.scalar_one_or_none()

    if existing_alert:
        if existing_alert.is_active:
            logger.info(
                f"[STOCK ALERT] Farmer {farmer.id} already has active alert for '{norm_product}' in '{clean_dist}'."
            )
            return existing_alert, False
        else:
            # Reactivate
            existing_alert.is_active = True
            existing_alert.notified_at = None
            existing_alert.updated_at = datetime.utcnow()
            db.add(existing_alert)
            await db.commit()
            await db.refresh(existing_alert)
            logger.info(
                f"[STOCK ALERT] Reactivated stock alert {existing_alert.id} for farmer {farmer.id} ('{norm_product}' in '{clean_dist}')."
            )
            return existing_alert, True

    # Create new alert
    new_alert = StockAlert(
        farmer_id=farmer.id,
        product_name=norm_product,
        district=clean_dist,
        state=clean_state,
        is_active=True,
    )
    db.add(new_alert)
    await db.commit()
    await db.refresh(new_alert)
    logger.info(
        f"[STOCK ALERT] Created new stock alert {new_alert.id} for farmer {farmer.id} ('{norm_product}' in '{clean_dist}')."
    )
    return new_alert, True


async def cancel_alert(
    db: AsyncSession,
    farmer: Farmer,
    product_name: Optional[str] = None,
    district: Optional[str] = None,
) -> int:
    """
    Cancel active stock alert(s) for a farmer.
    Returns number of alerts deactivated.
    """
    conditions = [StockAlert.farmer_id == farmer.id, StockAlert.is_active == True]

    if product_name:
        norm_prod = _normalize_product_name(product_name)
        conditions.append(StockAlert.product_name == norm_prod)
    if district:
        dist_variants = _get_district_match_variants(district)
        if dist_variants:
            conditions.append(or_(*[StockAlert.district.ilike(v) for v in dist_variants]))
        else:
            conditions.append(StockAlert.district.ilike(district.strip()))

    stmt = select(StockAlert).where(and_(*conditions))
    result = await db.execute(stmt)
    active_alerts = list(result.scalars().all())

    if not active_alerts:
        return 0

    for alert in active_alerts:
        alert.is_active = False
        alert.updated_at = datetime.utcnow()
        db.add(alert)

    await db.commit()
    logger.info(f"[STOCK ALERT] Deactivated {len(active_alerts)} alert(s) for farmer {farmer.id}.")
    return len(active_alerts)


async def list_farmer_alerts(db: AsyncSession, farmer: Farmer) -> List[StockAlert]:
    """Retrieve all active stock alerts for a farmer."""
    stmt = (
        select(StockAlert)
        .where(StockAlert.farmer_id == farmer.id, StockAlert.is_active == True)
        .order_by(StockAlert.created_at.desc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# Conversational Pipeline Integration Handler
# ─────────────────────────────────────────────────────────────────────────────

async def handle_stock_alert_query(
    db: AsyncSession,
    user_message: str,
    ai_response: str,
    farmer: Optional[Farmer],
    language: str = "te",
) -> str:
    """
    Deterministic handler for FarmerIntent.STOCK_ALERT:
    1. Sub-action = cancel -> cancel matching active alert(s).
    2. Sub-action = list -> return list of active subscriptions.
    3. Sub-action = subscribe:
       - Check if product is ALREADY available in farmer's district.
       - If in-stock: return immediate shop lookup results.
       - If out-of-stock: create/reactivate StockAlert and return confirmation.
    """
    if not farmer:
        # Fallback if no farmer identity is linked
        if language == "te":
            return "🔔 స్టాక్ అలర్ట్ నమోదు చేయడానికి మీ ఫోన్ నంబర్ వివరాలు అవసరం."
        return "🔔 Phone registration is required to set up stock availability alerts."

    action = detect_stock_alert_action(user_message)
    logger.info(f"[STOCK ALERT] Detected sub-action: '{action}' for farmer {farmer.id}")

    matched_product = _detect_product_from_query(user_message, ai_response) or "urea"
    norm_product = _normalize_product_name(matched_product)
    product_display = "యూరియా" if (norm_product == "urea" and language == "te") else norm_product.upper()

    # 1. CANCEL SUB-ACTION
    if action == "cancel":
        count = await cancel_alert(db, farmer, product_name=norm_product)
        if count > 0:
            if language == "te":
                return f"✅ మీ {product_display} స్టాక్ అలర్ట్ విజయవంతంగా రద్దు చేయబడింది."
            return f"✅ Your {norm_product.title()} stock alert has been successfully cancelled."
        else:
            if language == "te":
                return f"ℹ️ ప్రస్తుతం మీకు యాక్టివ్ {product_display} స్టాక్ అలర్ట్స్ ఏవీ లేవు."
            return f"ℹ️ You do not have any active {norm_product.title()} stock alerts."

    # 2. LIST SUB-ACTION
    if action == "list":
        active_alerts = await list_farmer_alerts(db, farmer)
        if not active_alerts:
            if language == "te":
                return "ℹ️ ప్రస్తుతం మీకు యాక్టివ్ స్టాక్ అలర్ట్స్ లేవు."
            return "ℹ️ You currently have no active stock alerts."

        if language == "te":
            lines = ["🔔 మీ యాక్టివ్ స్టాక్ అలర్ట్స్:\n"]
            for idx, a in enumerate(active_alerts, 1):
                prod_te = "యూరియా" if a.product_name == "urea" else a.product_name.title()
                lines.append(f"{idx}. {prod_te} — {a.district}\n   స్థితి: యాక్టివ్ (Active)")
            return "\n".join(lines)
        else:
            lines = ["🔔 Your Active Stock Alerts:\n"]
            for idx, a in enumerate(active_alerts, 1):
                lines.append(f"{idx}. {a.product_name.title()} — {a.district}\n   Status: Active")
            return "\n".join(lines)

    # 3. SUBSCRIBE SUB-ACTION
    # Resolve location
    loc_res = await _resolve_farmer_location(db, farmer, query_text=user_message)
    if len(loc_res) == 5:
        _, _, district, state, _ = loc_res
    else:
        _, _, district, state = loc_res

    district = district or "Warangal"
    state = state or "Telangana"

    # Step A: IMMEDIATE AVAILABILITY CHECK
    # Check if this product is ALREADY in stock in this district
    dist_variants = _get_district_match_variants(district)
    shop_district_filters = [Shop.district.ilike(v) for v in dist_variants]

    stmt_check = (
        select(Inventory, Shop)
        .join(Shop, Inventory.shop_id == Shop.id)
        .where(
            Inventory.available == True,
            Inventory.quantity_in_stock > 0,
            Shop.status == "active",
            or_(*shop_district_filters) if shop_district_filters else Shop.district.ilike(district.strip()),
            or_(
                Inventory.product_name.ilike(f"%{norm_product}%"),
                Inventory.category.ilike(f"%{norm_product}%"),
            ),
        )
    )
    res_check = await db.execute(stmt_check)
    in_stock_rows = res_check.all()

    if in_stock_rows:
        logger.info(
            f"[STOCK ALERT] '{norm_product}' is ALREADY IN STOCK in {district} ({len(in_stock_rows)} items). "
            "Returning immediate shop availability rather than creating redundant future alert."
        )
        # Use existing shop lookup and formatting logic
        shop_msg = await enrich_response_with_shops(db, user_message, "", farmer)
        if shop_msg and "🏬" in shop_msg:
            return shop_msg

    # Step B: Product is currently out-of-stock in farmer's district -> Create/Reactivate StockAlert
    alert, is_new = await create_or_reactivate_alert(
        db=db,
        farmer=farmer,
        product_name=norm_product,
        district=district,
        state=state,
    )

    district_display = district
    if language == "te":
        if district.lower() == "warangal":
            district_display = "వరంగల్"
        elif district.lower() == "guntur":
            district_display = "గుంటూరు"
        elif district.lower() == "khammam":
            district_display = "ఖమ్మం"
        elif district.lower() == "karimnagar":
            district_display = "కరీంనగర్"
        elif district.lower() == "nizamabad":
            district_display = "నిజామాబాద్"

        return (
            f"🔔 {product_display} స్టాక్ అలర్ట్ యాక్టివ్ అయింది.\n\n"
            f"📍 ప్రాంతం: {district_display}\n"
            f"ℹ️ మీ ప్రాంతంలోని డీలర్ల వద్ద {product_display} స్టాక్ అందుబాటులోకి రాగానే WhatsApp ద్వారా మీకు వెంటనే సమాచారం అందిస్తాము."
        )
    else:
        return (
            f"🔔 Your {norm_product.title()} stock alert is active.\n\n"
            f"📍 District: {district}\n"
            f"ℹ️ I will notify you on WhatsApp as soon as {norm_product.title()} becomes available at registered shops in your area."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Proactive Notification Dispatch (Triggered when Shop Restocks)
# ─────────────────────────────────────────────────────────────────────────────

async def trigger_stock_alert_notifications(
    inventory_item_id: UUID,
    shop_id: UUID,
    product_name: str,
    new_quantity: int,
    unit: str,
    brand: Optional[str] = None,
    db_session: Optional[AsyncSession] = None,
) -> int:
    """
    Triggered when an inventory item transitions from 0 to > 0.
    Finds matching active StockAlert subscriptions for (product_name, shop.district).
    Dispatches WhatsApp notifications and deactivates alerts on success.

    Returns:
        Number of farmers successfully notified.
    """
    from src.core.database import AsyncSessionLocal
    from src.gateway.whatsapp_client import (
        send_text_message,
        upload_media_bytes,
        send_audio_message,
    )
    from src.language.service import synthesize_speech
    from src.config import get_settings

    norm_product = _normalize_product_name(product_name)
    logger.info(
        f"[STOCK ALERT TRIGGER] Checking restock event for '{product_name}' (norm='{norm_product}'), "
        f"qty={new_quantity} {unit} at shop {shop_id}"
    )

    async def _execute(db: AsyncSession) -> int:
        # 1. Fetch Shop details
        shop_res = await db.execute(select(Shop).where(Shop.id == shop_id))
        shop = shop_res.scalar_one_or_none()
        if not shop:
            logger.warning(f"[STOCK ALERT TRIGGER] Shop {shop_id} not found.")
            return 0

        # Safe district resolution: shop.district or safe whitelist fallback from shop.address
        shop_district = (shop.district or "").strip()
        if not shop_district and shop.address:
            extracted_from_addr = _extract_district_from_query(shop.address)
            if extracted_from_addr:
                shop_district = extracted_from_addr
                logger.info(
                    f"[STOCK ALERT TRIGGER] Shop {shop_id} district is empty; "
                    f"resolved '{shop_district}' from shop address '{shop.address}'."
                )

        if not shop_district:
            logger.warning(
                f"[STOCK ALERT TRIGGER] Shop {shop_id} has no resolvable district from district or address."
            )
            return 0

        district_variants = _get_district_match_variants(shop_district)

        # 2. Query active matching alerts across bilingual variants
        stmt = (
            select(StockAlert, Farmer)
            .join(Farmer, StockAlert.farmer_id == Farmer.id)
            .where(
                StockAlert.is_active == True,
                StockAlert.product_name == norm_product,
                or_(*[StockAlert.district.ilike(v) for v in district_variants]) if district_variants else StockAlert.district.ilike(shop_district),
            )
        )
        res = await db.execute(stmt)
        matching_subscribers = res.all()

        if not matching_subscribers:
            logger.info(
                f"[STOCK ALERT TRIGGER] No active subscriptions found for '{norm_product}' in '{shop_district}'."
            )
            return 0

        logger.info(
            f"[STOCK ALERT TRIGGER] Found {len(matching_subscribers)} active subscriber(s) for "
            f"'{norm_product}' in '{shop_district}'."
        )

        redis_client = None
        settings = get_settings()
        if settings.redis_url:
            try:
                import redis.asyncio as aioredis
                redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
            except Exception as r_conn_err:
                logger.debug(f"Redis client initialization error: {r_conn_err}")
                redis_client = None

        # Prepare Stock Siren voice notification once per batch (batch-level caching)
        batch_media_id: Optional[str] = None
        try:
            prod_te_voice = "యూరియా" if norm_product == "urea" else product_name
            # Short, urgent Telugu voice script
            voice_script = (
                f"రైతు సోదరులారా, అత్యవసర స్టాక్ సమాచారం. "
                f"{shop_district} లోని {shop.shop_name} వద్ద {prod_te_voice} స్టాక్ అందుబాటులోకి వచ్చింది. "
                f"ప్రస్తుతం {new_quantity} {unit}ల స్టాక్ ఉంది. "
                f"స్టాక్ త్వరగా అయిపోయే అవకాశం ఉంది, వెంటనే దుకాణాన్ని సంప్రదించండి."
            )
            audio_bytes = await synthesize_speech(voice_script, language_code="te-IN")
            if audio_bytes:
                batch_media_id = await upload_media_bytes(
                    file_bytes=audio_bytes,
                    mime_type="audio/ogg",
                    filename="stock_siren_alert.ogg",
                )
        except Exception as audio_prep_err:
            logger.warning(f"[STOCK SIREN] Voice alert synthesis/upload skipped: {audio_prep_err}")
            batch_media_id = None

        notified_count = 0

        for alert, farmer in matching_subscribers:
            # Duplicate protection: Redis lock per alert + inventory event
            lock_key = f"stock_alert_lock:{alert.id}:{inventory_item_id}"
            if redis_client:
                try:
                    acquired = await redis_client.set(lock_key, "1", ex=86400, nx=True)
                    if not acquired:
                        logger.info(f"[STOCK ALERT TRIGGER] Duplicate suppressed by Redis lock: {lock_key}")
                        continue
                except Exception as r_err:
                    logger.debug(f"Redis lock check error: {r_err}")

            lang = farmer.preferred_language or "te"
            updated_time_str = datetime.utcnow().strftime("%I:%M %p").lstrip("0")

            # Format Stock Siren restock notification message
            if lang == "te":
                prod_te = "యూరియా" if norm_product == "urea" else product_name
                brand_str = brand if brand else "ప్రామాణిక బ్రాండ్"
                phone_str = shop.phone_number if shop.phone_number else "లభ్యత లేదు"
                notification_text = (
                    f"🚨🔔 అత్యవసర స్టాక్ అలర్ట్ (Stock Siren)!\n\n"
                    f"{prod_te} ప్రస్తుతం మీ ప్రాంతంలో ({shop_district}) స్టాక్ అందుబాటులోకి వచ్చింది.\n\n"
                    f"🏪 దుకాణం: {shop.shop_name}\n"
                    f"📍 జిల్లా / చిరునామా: {shop_district}, {shop.address}\n"
                    f"📦 ధృవీకరించిన ప్రస్తుత స్టాక్: {new_quantity} {unit}s\n"
                    f"🏷️ బ్రాండ్: {brand_str}\n"
                    f"🕒 స్టాక్ అప్‌డేట్ సమయం: {updated_time_str} UTC\n"
                    f"📞 ఫోన్: {phone_str}\n\n"
                    f"⚠️ హెచ్చరిక: స్టాక్ త్వరగా అయిపోయే అవకాశం ఉంది! వెళ్లే ముందు వెంటనే దుకాణానికి ఫోన్ చేసి నిర్ధారించుకోండి."
                )
            else:
                brand_str = brand if brand else "Standard Brand"
                phone_str = shop.phone_number if shop.phone_number else "N/A"
                notification_text = (
                    f"🚨🔔 Urgent Stock Siren!\n\n"
                    f"{norm_product.title()} is now back IN STOCK in your area ({shop_district}).\n\n"
                    f"🏪 Shop: {shop.shop_name}\n"
                    f"📍 District / Address: {shop_district}, {shop.address}\n"
                    f"📦 Verified Stock: {new_quantity} {unit}s\n"
                    f"🏷️ Brand: {brand_str}\n"
                    f"🕒 Updated: {updated_time_str} UTC\n"
                    f"📞 Phone: {phone_str}\n\n"
                    f"⚠️ Warning: High demand product, stock may sell out quickly! Please call the shop immediately to reserve or confirm."
                )

            # Send WhatsApp text message
            wa_msg_id = await send_text_message(
                to_phone=farmer.phone_number,
                message_text=notification_text,
            )

            if wa_msg_id:
                # Successfully sent text -> Deactivate alert and stamp notification time
                alert.is_active = False
                alert.notified_at = datetime.utcnow()
                alert.updated_at = datetime.utcnow()
                db.add(alert)
                notified_count += 1
                logger.info(
                    f"[STOCK ALERT TRIGGER] Successfully notified farmer {farmer.phone_number} "
                    f"(Alert ID {alert.id}, WA ID {wa_msg_id})."
                )

                # Send Stock Siren Voice Audio (Fail-soft)
                if batch_media_id:
                    try:
                        await send_audio_message(
                            to_phone=farmer.phone_number,
                            media_id=batch_media_id,
                        )
                    except Exception as wa_audio_err:
                        logger.warning(
                            f"[STOCK SIREN] Audio dispatch to {farmer.phone_number} failed soft: {wa_audio_err}"
                        )
            else:
                logger.error(
                    f"[STOCK ALERT TRIGGER] Failed to deliver WhatsApp message to {farmer.phone_number}. "
                    "Alert remains active."
                )

        if notified_count > 0:
            await db.commit()

        return notified_count

    if db_session:
        return await _execute(db_session)
    else:
        async with AsyncSessionLocal() as session:
            return await _execute(session)
