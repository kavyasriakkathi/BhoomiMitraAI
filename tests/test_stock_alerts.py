import pytest
import pytest_asyncio
import asyncio
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import Farmer, FarmerProfile, Farm, Shop, Inventory, StockAlert, Conversation, UserAccount
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.shops.stock_alerts import (
    detect_stock_alert_action,
    create_or_reactivate_alert,
    cancel_alert,
    list_farmer_alerts,
    handle_stock_alert_query,
    trigger_stock_alert_notifications,
    resolve_shop_district,
    _normalize_product_name,
    _canonicalize_district,
    _get_district_match_variants,
)
from src.inventory.service import InventoryService
from src.inventory.repository import InventoryRepository
from src.inventory.schemas import StockUpdatePayload, InventoryCreate, InventoryUpdate
from src.shops.repository import ShopRepository
from src.shops.schemas import ShopUpdate
from src.auth.dependencies import verify_shop_access, get_token_from_request
from src.auth.security import create_access_token, decode_access_token
from src.core.exceptions import BhoomiMitraException
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from src.core.database import Base


@pytest_asyncio.fixture
async def db_session():
    """Provides isolated in-memory SQLite async database session for stock alert tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_telugu_stock_alert_detection():
    engine = AIDecisionEngine()
    queries = [
        # Exact production string
        "యూరియా స్టాక్లోకి వస్తే నాకు చెప్పండి",
        # Mobile keyboard variant with invisible ZWNJ (\u200c)
        "యూరియా స్టాక్\u200cలోకి వస్తే నాకు చెప్పండి",
        # Mobile keyboard variant with ZWJ (\u200d)
        "యూరియా స్టాక్\u200dలోకి వస్తే నాకు చెప్పండి",
        # Space-separated variant
        "యూరియా స్టాక్ లోకి వస్తే నాకు చెప్పండి",
        # Space-separated with ZWNJ variant
        "యూరియా స్టాక్\u200c లోకి వస్తే నాకు చెప్పండి",
        # Whitespace variant
        "యూరియా  స్టాక్లోకి   వస్తే  నాకు  చెప్పండి",
        "యూరియా స్టాక్ వస్తే చెప్పండి",
        "యూరియా దొరికితే నాకు చెప్పండి",
        "యూరియాకు అలర్ట్ పెట్టండి",
        "యూరియా అందుబాటులోకి వస్తే చెప్పండి",
    ]
    for q in queries:
        intents = engine.detect_all_intents(q)
        assert FarmerIntent.STOCK_ALERT in intents, f"Failed to detect STOCK_ALERT for: '{q}'"


def test_english_stock_alert_detection():
    engine = AIDecisionEngine()
    queries = [
        "Alert me when urea is in stock",
        "Notify me when urea is available",
        "Tell me when urea is available",
        "Set a urea stock alert",
        "When urea comes in stock, tell me",
    ]
    for q in queries:
        intents = engine.detect_all_intents(q)
        assert FarmerIntent.STOCK_ALERT in intents, f"Failed to detect STOCK_ALERT for: '{q}'"


def test_tanglish_stock_alert_detection():
    engine = AIDecisionEngine()
    queries = [
        "Urea stock vasthe cheppandi",
        "Urea dorikithe cheppandi",
        "Urea ki alert pettandi",
        "Urea stock vachaka cheppandi",
    ]
    for q in queries:
        intents = engine.detect_all_intents(q)
        assert FarmerIntent.STOCK_ALERT in intents, f"Failed to detect STOCK_ALERT for: '{q}'"


def test_stop_alert_detection():
    queries = [
        "యూరియా అలర్ట్ ఆపండి",
        "స్టాక్ అలర్ట్ రద్దు చేయండి",
        "యూరియా నోటిఫికేషన్ ఆపండి",
        "Stop urea alert",
        "Cancel urea stock alert",
        "alert aapandi",
    ]
    for q in queries:
        action = detect_stock_alert_action(q)
        assert action == "cancel", f"Failed to detect cancel action for: '{q}'"


def test_list_alert_detection():
    queries = [
        "నా అలర్ట్స్ ఏంటి?",
        "నా స్టాక్ అలర్ట్స్",
        "యాక్టివ్ అలర్ట్స్",
        "Show my active alerts",
        "What alerts do I have?",
        "my alerts",
    ]
    for q in queries:
        action = detect_stock_alert_action(q)
        assert action == "list", f"Failed to detect list action for: '{q}'"


def test_existing_shop_lookup_remains_unchanged():
    engine = AIDecisionEngine()
    query = "నా దగ్గర యూరియా ఎక్కడ దొరుకుతుంది?"
    intents = engine.detect_all_intents(query)
    assert FarmerIntent.SHOPS in intents
    assert FarmerIntent.STOCK_ALERT not in intents


def test_fertilizer_dosage_intent_remains_unchanged():
    engine = AIDecisionEngine()
    query = "వరిలో యూరియా ఎంత వేయాలి?"
    intents = engine.detect_all_intents(query)
    assert FarmerIntent.FERTILIZER in intents
    assert FarmerIntent.STOCK_ALERT not in intents


# ─────────────────────────────────────────────────────────────────────────────
# 2. Database & Subscription Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alert_creation_and_duplicate_prevention(db_session):
    farmer = Farmer(phone_number="919876543210", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    # 1. Create alert
    alert1, is_new1 = await create_or_reactivate_alert(
        db=db_session,
        farmer=farmer,
        product_name="యూరియా",
        district="Warangal",
    )
    assert is_new1 is True
    assert alert1.is_active is True
    assert alert1.product_name == "urea"
    assert alert1.district == "Warangal"

    # 2. Re-subscribing while active does NOT create duplicate
    alert2, is_new2 = await create_or_reactivate_alert(
        db=db_session,
        farmer=farmer,
        product_name="urea",
        district="Warangal",
    )
    assert is_new2 is False
    assert alert2.id == alert1.id

    # Verify only 1 record exists in DB
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1


@pytest.mark.asyncio
async def test_alert_cancellation_and_reactivation(db_session):
    farmer = Farmer(phone_number="919876543211", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    # 1. Create alert
    alert, _ = await create_or_reactivate_alert(
        db=db_session,
        farmer=farmer,
        product_name="urea",
        district="Warangal",
    )
    assert alert.is_active is True

    # 2. Cancel alert
    cancelled_count = await cancel_alert(db_session, farmer, product_name="urea")
    assert cancelled_count == 1

    # Verify list is empty
    active_alerts = await list_farmer_alerts(db_session, farmer)
    assert len(active_alerts) == 0

    # 3. Reactivate alert
    reactivated_alert, is_new = await create_or_reactivate_alert(
        db=db_session,
        farmer=farmer,
        product_name="urea",
        district="Warangal",
    )
    assert is_new is True
    assert reactivated_alert.is_active is True
    assert reactivated_alert.id == alert.id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Immediate Fulfillment vs. Stock Out Behavior
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_immediate_fulfillment_when_already_in_stock(db_session):
    farmer = Farmer(phone_number="919876543212", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    # Create active shop with in-stock urea
    shop = Shop(
        shop_name="Kisan Seva Kendra",
        owner_name="Suresh",
        phone_number="9876500001",
        address="Station Road, Warangal",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()
    await db_session.refresh(shop)

    inv = Inventory(
        shop_id=shop.id,
        product_name="IFFCO Neem Coated Urea",
        category="Fertilizer",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=50,
        available=True,
    )
    db_session.add(inv)
    await db_session.commit()

    # Query alert in Warangal
    reply = await handle_stock_alert_query(
        db=db_session,
        user_message="వరంగల్లో యూరియా స్టాక్ వస్తే చెప్పండి",
        ai_response="",
        farmer=farmer,
        language="te",
    )

    # Should return immediate shop info rather than creating future alert
    assert "Kisan Seva Kendra" in reply or "🏬" in reply or "268" in reply
    # No future alert should be created
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_stock_out_creates_future_alert(db_session):
    farmer = Farmer(phone_number="919876543213", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    # No shops have urea in Karimnagar
    reply = await handle_stock_alert_query(
        db=db_session,
        user_message="కరీంనగర్ లో యూరియా స్టాక్లోకి వస్తే నాకు చెప్పండి",
        ai_response="",
        farmer=farmer,
        language="te",
    )

    assert "🔔" in reply
    assert "యాక్టివ్" in reply or "కరీంనగర్" in reply

    # Alert should be persisted
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1
    assert alerts[0].product_name == "urea"
    assert alerts[0].district == "Karimnagar"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Proactive Restock Trigger & WhatsApp Notification Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_restock_trigger_sends_notification_and_deactivates_alert(db_session):
    # Subscriber 1 in Warangal
    farmer1 = Farmer(phone_number="919876543214", preferred_language="te")
    db_session.add(farmer1)
    # Subscriber 2 in Warangal
    farmer2 = Farmer(phone_number="919876543215", preferred_language="en")
    db_session.add(farmer2)
    # Subscriber 3 in Guntur (different district)
    farmer3 = Farmer(phone_number="919876543216", preferred_language="te")
    db_session.add(farmer3)
    await db_session.commit()

    await create_or_reactivate_alert(db_session, farmer1, "urea", "Warangal")
    await create_or_reactivate_alert(db_session, farmer2, "urea", "Warangal")
    await create_or_reactivate_alert(db_session, farmer3, "urea", "Guntur")

    shop = Shop(
        shop_name="Bhoomi Agro Agencies",
        owner_name="Venkatesh",
        phone_number="9876500002",
        address="Market Yard, Warangal",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()
    await db_session.refresh(shop)

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizer",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()
    await db_session.refresh(inv)

    # Restock event: 0 -> 100 bags
    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.HBgL..."

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=100,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )

        # Warangal subscribers (farmer1 and farmer2) should be notified
        assert notified_count == 2
        assert mock_send.call_count == 2

        # Check call arguments
        phones_called = [call.kwargs["to_phone"] for call in mock_send.call_args_list]
        assert "919876543214" in phones_called
        assert "919876543215" in phones_called
        assert "919876543216" not in phones_called  # Guntur farmer must NOT be notified for Warangal restock

    # Check alert state: Warangal alerts should be deactivated
    f1_alerts = await list_farmer_alerts(db_session, farmer1)
    assert len(f1_alerts) == 0

    f2_alerts = await list_farmer_alerts(db_session, farmer2)
    assert len(f2_alerts) == 0

    # Guntur alert remains active
    f3_alerts = await list_farmer_alerts(db_session, farmer3)
    assert len(f3_alerts) == 1
    assert f3_alerts[0].is_active is True


@pytest.mark.asyncio
async def test_failed_whatsapp_send_keeps_alert_active(db_session):
    farmer = Farmer(phone_number="919876543217", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")

    shop = Shop(
        shop_name="Warangal Agri Centre",
        owner_name="Rao",
        phone_number="9876500003",
        address="Warangal",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        # Simulate send failure (returns None)
        mock_send.return_value = None

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=uuid4(),
            shop_id=shop.id,
            product_name="urea",
            new_quantity=50,
            unit="bag",
            db_session=db_session,
        )

        assert notified_count == 0

    # Alert MUST remain active for retry
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1
    assert alerts[0].is_active is True
    assert alerts[0].notified_at is None


@pytest.mark.asyncio
async def test_inventory_service_stock_update_transitions(db_session):
    """
    Test transition thresholds:
    0 -> 50: Trigger
    50 -> 40: No trigger
    40 -> 20: No trigger
    20 -> 0: No trigger
    0 -> 10: Trigger
    """
    shop = Shop(
        shop_name="Test Dealer",
        owner_name="Test Owner",
        phone_number="9876500004",
        address="Test Address",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()
    await db_session.refresh(shop)

    repo = InventoryRepository(db_session)
    service = InventoryService(repo)

    # Create out of stock item
    item = await service.add_product(
        InventoryCreate(
            shop_id=shop.id,
            product_name="Urea",
            category="Fertilizer",
            brand="IFFCO",
            unit="bag",
            price=268.0,
            quantity_in_stock=0,
            available=False,
        )
    )

    with patch("src.shops.stock_alerts.trigger_stock_alert_notifications", new_callable=AsyncMock) as mock_trigger:
        # 1. Update: 0 -> 50 (MUST trigger)
        await service.update_stock(item.id, StockUpdatePayload(quantity_in_stock=50, available=True))
        await asyncio.sleep(0.01)
        assert mock_trigger.call_count == 1

        mock_trigger.reset_mock()

        # 2. Update: 50 -> 40 (MUST NOT trigger)
        await service.update_stock(item.id, StockUpdatePayload(quantity_in_stock=40, available=True))
        await asyncio.sleep(0.01)
        assert mock_trigger.call_count == 0

        # 3. Update: 40 -> 20 (MUST NOT trigger)
        await service.update_stock(item.id, StockUpdatePayload(quantity_in_stock=20, available=True))
        await asyncio.sleep(0.01)
        assert mock_trigger.call_count == 0

        # 4. Update: 20 -> 0 (MUST NOT trigger)
        await service.update_stock(item.id, StockUpdatePayload(quantity_in_stock=0, available=False))
        await asyncio.sleep(0.01)
        assert mock_trigger.call_count == 0

        # 5. Update: 0 -> 10 (MUST trigger again)
        await service.update_stock(item.id, StockUpdatePayload(quantity_in_stock=10, available=True))
        await asyncio.sleep(0.01)
        assert mock_trigger.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Intent and Decision Engine Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_intent_stock_alert_and_market_price(db_session):
    farmer = Farmer(phone_number="919876543218", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()
    await db_session.refresh(farmer)

    engine = AIDecisionEngine()
    query = "యూరియా స్టాక్లోకి వస్తే చెప్పండి, వరంగల్లో పత్తి ధర ఎంత?"
    intents = engine.detect_all_intents(query)

    assert FarmerIntent.STOCK_ALERT in intents
    assert FarmerIntent.MARKET_PRICE in intents

    conv = Conversation(
        farmer_id=farmer.id,
        message_id="msg_test_multi_intent_stock",
        user_message=query,
    )
    db_session.add(conv)
    await db_session.commit()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_ai:
        mock_resp = MagicMock()
        mock_resp.response_text = "రైతు సోదరులకు నమస్కారం."
        mock_ai.return_value = mock_resp

        reply = await engine.process_message(db_session, farmer, conv)

        # Both stock alert confirmation and market price guidance should be present
        assert "🔔" in reply or "యూరియా" in reply
        assert "పత్తి" in reply or "ధర" in reply or "మార్కెట్" in reply


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bilingual District Normalization & Safe Address Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_bilingual_district_canonicalization_and_variants():
    """Unit tests verifying canonicalization across Telugu, English, casing, and addresses."""
    # Canonicalization
    assert _canonicalize_district("వరంగల్") == "Warangal"
    assert _canonicalize_district("Warangal") == "Warangal"
    assert _canonicalize_district("  warangal  ") == "Warangal"
    assert _canonicalize_district("Main Bazar, Warangal") == "Warangal"
    assert _canonicalize_district("గుంటూరు") == "Guntur"
    assert _canonicalize_district("Guntur") == "Guntur"
    # Does not invent or alter unknown text
    assert _canonicalize_district("Unknown Area 123") == "Unknown Area 123"
    assert _canonicalize_district(None) is None
    assert _canonicalize_district("") is None

    # resolve_shop_district resolution logic
    assert resolve_shop_district(None, "Main Bazar, Warangal") == "Warangal"
    assert resolve_shop_district("", "Main Bazar, Warangal") == "Warangal"
    assert resolve_shop_district("Main Bazar, Warangal", None) == "Warangal"
    assert resolve_shop_district("Warangal", None) == "Warangal"
    assert resolve_shop_district("వరంగల్", None) == "Warangal"
    assert resolve_shop_district(None, "వరంగల్, మెయిన్ బజార్") == "Warangal"
    assert resolve_shop_district("Unknown Area 123", "Main Bazar, Warangal") == "Warangal"
    # Unrelated or invalid address safely returns None
    assert resolve_shop_district(None, "Plot 42, Sector 9, Industrial Estate") is None
    assert resolve_shop_district("Unknown Area 123", "Plot 42") is None
    assert resolve_shop_district(None, None) is None
    assert resolve_shop_district("", "") is None

    # Matching variants
    w_variants = _get_district_match_variants("Warangal")
    assert "Warangal" in w_variants
    assert "warangal" in w_variants
    assert "వరంగల్" in w_variants

    te_variants = _get_district_match_variants("వరంగల్")
    assert "Warangal" in te_variants
    assert "warangal" in te_variants
    assert "వరంగల్" in te_variants
    assert set(w_variants) == set(te_variants)


@pytest.mark.asyncio
async def test_trigger_bilingual_telugu_alert_english_shop(db_session):
    """
    Telugu alert ('వరంగల్') must be matched and notified when English shop ('Warangal') restocks.
    """
    farmer = Farmer(phone_number="919876543301", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    # Alert created with Telugu district
    alert, is_new = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert is_new is True
    assert alert.is_active is True

    # Shop with English district
    shop = Shop(
        shop_name="Warangal Agri Center",
        owner_name="Ramesh",
        phone_number="9876510001",
        address="Market Road, Warangal",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea Fertilizer 45kg",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.test01"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )

        assert notified_count == 1
        assert mock_send.call_count == 1
        assert mock_send.call_args[1]["to_phone"] == "919876543301"

    # Verify alert deactivated
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_trigger_bilingual_english_alert_telugu_shop(db_session):
    """
    English alert ('Warangal') must be matched and notified when Telugu shop ('వరంగల్') restocks.
    """
    farmer = Farmer(phone_number="919876543302", preferred_language="en")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(
        db_session, farmer, "urea", "Warangal"
    )
    assert alert.is_active is True

    shop = Shop(
        shop_name="Telangana Agri",
        owner_name="Suresh",
        phone_number="9876510002",
        address="వరంగల్",
        district="వరంగల్",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.test02"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=50,
            unit="bag",
            db_session=db_session,
        )

        assert notified_count == 1
        assert mock_send.call_count == 1

    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_trigger_same_script_and_whitespace_case(db_session):
    """
    Verifies same-script matching and whitespace/casing normalization ('  warangal  ' == 'WARANGAL').
    """
    farmer1 = Farmer(phone_number="919876543303", preferred_language="te")
    farmer2 = Farmer(phone_number="919876543304", preferred_language="en")
    db_session.add_all([farmer1, farmer2])
    await db_session.commit()

    await create_or_reactivate_alert(db_session, farmer1, "urea", "  warangal  ")
    await create_or_reactivate_alert(db_session, farmer2, "urea", "వరంగల్")

    shop = Shop(
        shop_name="Kisan Store",
        owner_name="Owner",
        phone_number="9876510003",
        address="Town Hall Road",
        district="WARANGAL",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.test03"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=25,
            unit="bag",
            db_session=db_session,
        )

        assert notified_count == 2
        assert mock_send.call_count == 2


@pytest.mark.asyncio
async def test_trigger_shop_district_null_address_fallback(db_session):
    """
    Production condition test: shop.district is NULL, but shop.address contains 'Main Bazar, Warangal'.
    Alert district is 'వరంగల్'. Must safely resolve Warangal from address and notify subscriber.
    """
    farmer = Farmer(phone_number="919876543305", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert alert.is_active is True

    # Shop with district=None, but valid address
    shop = Shop(
        shop_name="Rythu Mitra Fertilizers",
        owner_name="Srinivas Rao",
        phone_number="9876510004",
        address="Main Bazar, Warangal",
        district=None,
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea Fertilizer 45kg",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.test04"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )

        assert notified_count == 1
        assert mock_send.call_count == 1
        assert mock_send.call_args[1]["to_phone"] == "919876543305"

    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_trigger_unrelated_district_never_matches(db_session):
    """
    Negative test: Warangal alert must never match a restock in Karimnagar.
    """
    farmer = Farmer(phone_number="919876543306", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )

    shop = Shop(
        shop_name="Karimnagar Agro",
        owner_name="Venkat",
        phone_number="9876510005",
        address="Bus Stand, Karimnagar",
        district="Karimnagar",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=50,
            unit="bag",
            db_session=db_session,
        )

        assert notified_count == 0
        assert mock_send.call_count == 0

    # Alert remains active
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1
    assert alerts[0].is_active is True


@pytest.mark.asyncio
async def test_trigger_missing_district_and_invalid_address_fails_safe(db_session):
    """
    Negative test: shop with district=None and address without any known district fails safe.
    """
    farmer = Farmer(phone_number="919876543307", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )

    shop = Shop(
        shop_name="Unknown Location Store",
        owner_name="Unknown",
        phone_number="9876510006",
        address="Plot 42, Sector 9, Industrial Estate",
        district=None,
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=50,
            unit="bag",
            db_session=db_session,
        )

        assert notified_count == 0
        assert mock_send.call_count == 0

    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1
    assert alerts[0].is_active is True


@pytest.mark.asyncio
async def test_bilingual_duplicate_alert_prevention(db_session):
    """
    Subscribing in Telugu ('వరంగల్') and then in English ('Warangal') must NOT create duplicate alerts.
    """
    farmer = Farmer(phone_number="919876543308", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert1, is_new1 = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert is_new1 is True

    # Resubscribe with English name
    alert2, is_new2 = await create_or_reactivate_alert(
        db_session, farmer, "urea", "Warangal"
    )
    assert is_new2 is False
    assert alert2.id == alert1.id

    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 1


@pytest.mark.asyncio
async def test_bilingual_cancel_alert(db_session):
    """
    Alert created with Telugu ('వరంగల్') can be cancelled using English ('Warangal').
    """
    farmer = Farmer(phone_number="919876543309", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert len(await list_farmer_alerts(db_session, farmer)) == 1

    cancelled = await cancel_alert(db_session, farmer, product_name="urea", district="Warangal")
    assert cancelled == 1
    assert len(await list_farmer_alerts(db_session, farmer)) == 0


@pytest.mark.asyncio
async def test_stock_siren_district_resolution_regression(db_session):
    """
    Regression test for Stock Siren district resolution:
    - Shop with address 'Main Bazar, Warangal' and district=None must resolve to Warangal.
    - Matches Telugu active StockAlert for 'వరంగల్'.
    - Urgent WhatsApp notification sent with verified quantity, shop, location, timestamp, and sell-out warning.
    - Alert deactivated and notified_at recorded.
    - Re-triggering does not send duplicate notifications.
    """
    farmer = Farmer(phone_number="919876543310", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, is_new = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert is_new is True
    assert alert.is_active is True

    # Shop with district=None, but address containing 'Main Bazar, Warangal'
    shop = Shop(
        id=uuid4(),
        shop_name="Rythu Mitra Fertilizers",
        owner_name="Srinivas Rao",
        phone_number="9876510010",
        address="Main Bazar, Warangal",
        district=None,
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    # Verify resolve_shop_district correctly resolves
    assert resolve_shop_district(shop.district, shop.address) == "Warangal"

    inv = Inventory(
        id=uuid4(),
        shop_id=shop.id,
        product_name="Urea Fertilizer 45kg",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.siren.test01"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )

        assert notified_count == 1
        assert mock_send.call_count == 1
        msg = mock_send.call_args[1]["message_text"]
        assert "Rythu Mitra Fertilizers" in msg
        assert "Warangal" in msg or "వరంగల్" in msg
        assert "Main Bazar, Warangal" in msg
        assert "50 bag" in msg
        assert "హెచ్చరిక" in msg or "Warning" in msg
        assert "UTC" in msg

    # Alert must now be inactive and notified_at must be populated
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0  # list_farmer_alerts returns only active alerts
    assert alert.is_active is False
    assert alert.notified_at is not None

    # Re-triggering must NOT duplicate notification
    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send2:
        second_notified = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )
        assert second_notified == 0
        assert mock_send2.call_count == 0


@pytest.mark.asyncio
async def test_production_unresolvable_location_update_and_stock_siren_trigger(db_session):
    """
    Simulates and validates the exact production situation:
    - Shop has district=None, address='korutla', latitude=17.9784, longitude=79.5941.
    - Active StockAlert exists for district='వరంగల్', product='urea'.
    - Restock event (qty 0 -> 50) fails because 'korutla' does not resolve to Warangal,
      and coordinates are not auto-inferred (preserving curated whitelist & avoiding arbitrary inferences).
    - 0 notifications are sent.
    - Shop owner performs legitimate location update (setting district='Warangal').
    - resolve_shop_district now correctly resolves to 'Warangal'.
    - Next restock event successfully triggers Stock Siren, notifying the farmer in Telugu via WhatsApp.
    - Also verifies alternative legitimate action: updating address to 'Warangal' when district is None.
    """
    farmer = Farmer(phone_number="919876543399", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, is_new = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert is_new is True
    assert alert.is_active is True

    # 1. Setup exact production shop initial state
    shop_repo = ShopRepository(db_session)
    prod_shop_id = uuid4()
    prod_shop = Shop(
        id=prod_shop_id,
        shop_name="Kisan Seva Kendra",
        owner_name="Mallesh Goud",
        phone_number="9876543999",
        address="korutla",
        district=None,
        latitude=17.9784,
        longitude=79.5941,
        state="Telangana",
        status="active",
    )
    db_session.add(prod_shop)
    await db_session.commit()

    inv = Inventory(
        id=uuid4(),
        shop_id=prod_shop.id,
        product_name="Urea Fertilizer 45kg",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    # Step 2: In initial state, district resolution fails ('korutla' not in curated whitelist)
    assert resolve_shop_district(prod_shop.district, prod_shop.address) is None

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        notified = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=prod_shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )
        assert notified == 0
        assert mock_send.call_count == 0

    # Farmer alert remains active because no restock alert could be dispatched for this district
    assert alert.is_active is True

    # Step 3: Shop owner updates district to 'Warangal' via legitimate update mechanism
    updated_shop = await shop_repo.update(prod_shop.id, ShopUpdate(district="Warangal"))
    assert updated_shop is not None
    assert updated_shop.district == "Warangal"
    assert updated_shop.address == "korutla"

    # Step 4: resolve_shop_district now successfully resolves to 'Warangal'
    assert resolve_shop_district(updated_shop.district, updated_shop.address) == "Warangal"

    # Step 5: Restock event now successfully triggers Stock Siren!
    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send_restock:
        mock_send_restock.return_value = "wamid.siren.test02"
        notified_restock = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=prod_shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )
        assert notified_restock == 1
        assert mock_send_restock.call_count == 1
        msg = mock_send_restock.call_args[1]["message_text"]
        assert "Kisan Seva Kendra" in msg
        assert "Warangal" in msg or "వరంగల్" in msg
        assert "50 bag" in msg
        assert "హెచ్చరిక" in msg or "Warning" in msg

    # Step 6: Verify alternative update mechanism (updating address to 'Warangal' when district=None)
    # Reset alert for testing address-only fallback
    alert.is_active = True
    alert.notified_at = None
    # Reset shop district to None, update address to 'Warangal'
    prod_shop.district = None
    prod_shop.address = "Warangal"
    db_session.add(prod_shop)
    await db_session.commit()

    assert resolve_shop_district(prod_shop.district, prod_shop.address) == "Warangal"

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send_addr:
        mock_send_addr.return_value = "wamid.siren.test03"
        notified_addr = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=prod_shop.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )
        assert notified_addr == 1
        assert mock_send_addr.call_count == 1


@pytest.mark.asyncio
async def test_shop_owner_authentication_lifecycle_and_address_update(db_session):
    """
    Validates the authentication lifecycle and profile update flow:
    1. Request without token/cookie returns None from get_token_from_request (causes 401).
    2. Expired JWT token raises 401 Invalid or expired authentication token.
    3. Unauthorized shop owner (Shop B) attempting to update Shop A raises 403 Access forbidden.
    4. Authenticated owner of Shop A passes verify_shop_access and updates address to 'Warangal'.
    5. 'Warangal' persists in the database record.
    6. Restock trigger resolves 'Warangal', matches Telugu subscriber for 'వరంగల్', and fires Stock Siren.
    """
    farmer = Farmer(phone_number="919876543388", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, is_new = await create_or_reactivate_alert(
        db_session, farmer, "urea", "వరంగల్"
    )
    assert is_new is True
    assert alert.is_active is True

    # Setup Shop A
    shop_repo = ShopRepository(db_session)
    shop_a_id = uuid4()
    shop_a = Shop(
        id=shop_a_id,
        shop_name="Warangal Agri Depot",
        owner_name="Mallanna",
        phone_number="9876543888",
        address="korutla",  # Initial unresolvable address
        district=None,
        state="Telangana",
        status="active",
    )
    db_session.add(shop_a)

    # Setup Shop B
    shop_b_id = uuid4()
    shop_b = Shop(
        id=shop_b_id,
        shop_name="Other Shop",
        owner_name="Other Owner",
        phone_number="9876543777",
        address="Some Address",
        district=None,
        state="Telangana",
        status="active",
    )
    db_session.add(shop_b)
    await db_session.commit()

    # User A (Owner of Shop A)
    user_a = UserAccount(
        id=uuid4(),
        email="owner.a@bhoomimitra.ai",
        password_hash="hashed_pw_test_a",
        role="shop_owner",
        shop_id=shop_a_id,
        is_active=True,
    )
    # User B (Owner of Shop B)
    user_b = UserAccount(
        id=uuid4(),
        email="owner.b@bhoomimitra.ai",
        password_hash="hashed_pw_test_b",
        role="shop_owner",
        shop_id=shop_b_id,
        is_active=True,
    )
    db_session.add_all([user_a, user_b])
    await db_session.commit()

    # Step 1: Unauthenticated request simulation
    mock_req_empty = MagicMock()
    mock_req_empty.headers = {}
    mock_req_empty.cookies = {}
    assert get_token_from_request(mock_req_empty) is None

    # Step 2: Expired token raises 401
    expired_token = create_access_token(
        {"sub": str(user_a.id), "role": user_a.role},
        expires_delta=timedelta(minutes=-30),
    )
    with pytest.raises(BhoomiMitraException) as exc_expired:
        decode_access_token(expired_token)
    assert exc_expired.value.status_code == 401
    assert "expired" in exc_expired.value.message.lower()

    # Step 3: Tenant Isolation: User B cannot modify Shop A
    with pytest.raises(BhoomiMitraException) as exc_tenant:
        verify_shop_access(user_b, shop_a_id)
    assert exc_tenant.value.status_code == 403
    assert "another shop" in exc_tenant.value.message.lower()

    # Step 4: Legitimate Owner A passes tenant authorization
    verify_shop_access(user_a, shop_a_id)  # Does not raise!

    # Step 5: Update Address to 'Warangal' and verify persistence
    updated = await shop_repo.update(shop_a_id, ShopUpdate(address="Warangal"))
    assert updated is not None
    assert updated.address == "Warangal"
    assert updated.district is None

    # Re-fetch from DB to confirm persistence
    persisted = await shop_repo.get_by_id(shop_a_id)
    assert persisted.address == "Warangal"

    # Step 6: Verify Stock Siren resolves 'Warangal' and triggers restock notification
    assert resolve_shop_district(persisted.district, persisted.address) == "Warangal"

    inv = Inventory(
        id=uuid4(),
        shop_id=shop_a.id,
        product_name="Urea Fertilizer 45kg",
        category="Fertilizers",
        brand="IFFCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wamid.siren.auth.test"
        notified = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop_a.id,
            product_name="Urea Fertilizer 45kg",
            new_quantity=50,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )
        assert notified == 1
        assert mock_send.call_count == 1
        msg = mock_send.call_args[1]["message_text"]
        assert "Warangal Agri Depot" in msg
        assert "Warangal" in msg or "వరంగల్" in msg
        assert "50 bag" in msg
