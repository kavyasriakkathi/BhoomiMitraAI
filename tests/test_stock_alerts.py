import pytest
import pytest_asyncio
import asyncio
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import Farmer, FarmerProfile, Farm, Shop, Inventory, StockAlert, Conversation
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.shops.stock_alerts import (
    detect_stock_alert_action,
    create_or_reactivate_alert,
    cancel_alert,
    list_farmer_alerts,
    handle_stock_alert_query,
    trigger_stock_alert_notifications,
    _normalize_product_name,
)
from src.inventory.service import InventoryService
from src.inventory.repository import InventoryRepository
from src.inventory.schemas import StockUpdatePayload, InventoryCreate, InventoryUpdate
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
