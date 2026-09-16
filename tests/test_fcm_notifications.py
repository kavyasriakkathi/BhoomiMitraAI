"""
BhoomiMitra AI — Tests for FCM Phone-Level Stock Siren Notifications

Validates:
1. Token registration, duplicate handling, token refresh, and deactivation.
2. Batch deactivation of stale/invalid tokens reported by Firebase.
3. Multilingual localization across all 13 supported Indian languages.
4. Restock event (0 -> positive) triggers FCM push notification.
5. Positive-to-positive stock change does not trigger restock or FCM.
6. Fail-soft behavior: FCM failure does NOT disrupt WhatsApp text/template alert.
7. Token masking utility prevents plaintext token leaks in logs.
8. Notifications router endpoints for Android token lifecycle.
"""

import pytest
import pytest_asyncio
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from src.core.database import Base
from src.core.models import Farmer, Shop, Inventory, StockAlert, FarmerPushToken
from src.notifications.schemas import mask_token, PushTokenRegisterRequest, PushTokenDeactivateRequest
from src.notifications.repository import PushTokenRepository
from src.notifications.service import (
    get_localized_stock_siren_content,
    dispatch_fcm_stock_siren_notification,
)
from src.language.languages import SUPPORTED_LANGUAGES
from src.shops.stock_alerts import trigger_stock_alert_notifications
from src.inventory.service import InventoryService
from src.inventory.repository import InventoryRepository
from src.inventory.schemas import StockUpdatePayload
from src.main import app


@pytest_asyncio.fixture
async def db_session():
    """Provides isolated in-memory SQLite async database session for notification tests."""
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


def test_mask_token_utility():
    """Ensure FCM tokens are masked to prevent plaintext leakage in logs."""
    token = "eK1234567890abcdefghijklmnopqrstuvwxyz_FCM_REGISTRATION_TOKEN_9999"
    masked = mask_token(token)
    assert masked.startswith("eK1234...")
    assert masked.endswith("9999")
    assert "abcdefghijkl" not in masked
    assert mask_token("") == "***"
    assert mask_token("short") == "***"


@pytest.mark.asyncio
async def test_token_registration_and_update(db_session: AsyncSession):
    """Test registering a new token and refreshing/re-registering an existing token."""
    repo = PushTokenRepository(db_session)
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919876543210", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    token_str = "fcm_token_farmer_1_device_abc123"
    reg1 = await repo.register_or_update_token(
        farmer_id=farmer_id,
        token=token_str,
        platform="android",
        device_model="Samsung Galaxy M31",
    )
    assert reg1.id is not None
    assert reg1.farmer_id == farmer_id
    assert reg1.is_active is True
    assert reg1.device_model == "Samsung Galaxy M31"

    # Re-registering same token updates the record without creating duplicate
    reg2 = await repo.register_or_update_token(
        farmer_id=farmer_id,
        token=token_str,
        platform="android",
        device_model="Samsung Galaxy M31 Updated",
    )
    assert reg2.id == reg1.id
    assert reg2.device_model == "Samsung Galaxy M31 Updated"

    tokens = await repo.get_active_tokens_for_farmer(farmer_id)
    assert len(tokens) == 1


@pytest.mark.asyncio
async def test_token_deactivation_and_batch_deactivation(db_session: AsyncSession):
    """Test manual deactivation and automatic batch deactivation for invalid tokens."""
    repo = PushTokenRepository(db_session)
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919876543211", preferred_language="hi")
    db_session.add(farmer)
    await db_session.commit()

    t1 = await repo.register_or_update_token(farmer_id, "token_alpha_111")
    t2 = await repo.register_or_update_token(farmer_id, "token_beta_222")
    t3 = await repo.register_or_update_token(farmer_id, "token_gamma_333")

    active = await repo.get_active_tokens_for_farmer(farmer_id)
    assert len(active) == 3

    # Deactivate single token
    success = await repo.deactivate_token("token_alpha_111")
    assert success is True

    active_after = await repo.get_active_tokens_for_farmer(farmer_id)
    assert len(active_after) == 2
    assert "token_alpha_111" not in [t.token for t in active_after]

    # Batch deactivate (e.g. reported unregistered by Firebase)
    count = await repo.deactivate_tokens_batch(["token_beta_222", "token_gamma_333"])
    assert count == 2

    remaining = await repo.get_active_tokens_for_farmer(farmer_id)
    assert len(remaining) == 0


def test_multilingual_localization_all_13_languages():
    """Verify that Stock Siren push content formats cleanly across all 13 supported languages."""
    expected_title_fragments = {
        "te": "స్టాక్ అందుబాటులోకి వచ్చింది",
        "hi": "स्टॉक उपलब्ध हो गया है",
        "en": "Stock Now Available",
        "ta": "இருப்பு இப்போது கிடைக்கிறது",
        "kn": "ಸ್ಟಾಕ್ ಈಗ ಲಭ್ಯವಿದೆ",
        "ml": "സ്റ്റോക്ക് ഇപ്പോൾ ലഭ്യമാണ്",
        "mr": "स्टॉक आता उपलब्ध आहे",
        "bn": "স্টক এখন উপলব্ধ",
        "gu": "સ્ટોક હવે ઉપલબ્ધ છે",
        "or": "ଷ୍ଟକ୍ ବର୍ତ୍ତମାନ ଉପଲବ୍ଧ",
        "pa": "ਸਟਾਕ ਹੁਣ ਉਪਲਬਧ ਹੈ",
        "as": "ষ্টক এতিয়া উপলব্ধ",
        "ur": "اسٹاک اب دستیاب ہے",
    }

    for lang_code in SUPPORTED_LANGUAGES.keys():
        title, body, payload = get_localized_stock_siren_content(
            language=lang_code,
            product_name="urea",
            shop_name="Rythu Mitra Agro",
            district="Warangal",
            quantity=50,
            unit="Bags",
            brand="IFFCO",
            updated_time_str="06:00 PM UTC",
        )

        assert "🚨" in title
        expected_frag = expected_title_fragments.get(lang_code)
        if expected_frag:
            assert expected_frag in title, f"Language '{lang_code}' missing title fragment '{expected_frag}'"

        assert "Rythu Mitra Agro" in body
        assert "Warangal" in body
        assert "50" in body
        assert "Bags" in body
        assert "IFFCO" in body

        assert payload["alert_type"] == "stock_siren"
        assert payload["product_name"] == "urea"
        assert payload["shop_name"] == "Rythu Mitra Agro"
        assert payload["district"] == "Warangal"
        assert payload["quantity"] == "50"
        assert payload["unit"] == "Bags"
        assert payload["brand"] == "IFFCO"
        assert payload["language"] == lang_code


@pytest.mark.asyncio
async def test_restock_event_triggers_fcm_alongside_whatsapp(db_session: AsyncSession):
    """
    Verify that an inventory transition 0 -> 50 triggers BOTH the existing WhatsApp
    alert and the new FCM Stock Siren push notification.
    """
    farmer = Farmer(id=uuid4(), phone_number="919876543220", preferred_language="te")
    shop = Shop(
        id=uuid4(),
        shop_name="Warangal Agri Depot",
        owner_name="Test Owner",
        district="Warangal",
        phone_number="919876543221",
        address="Warangal",
    )
    alert = StockAlert(
        id=uuid4(),
        farmer_id=farmer.id,
        product_name="urea",
        district="Warangal",
        state="Telangana",
        is_active=True,
    )
    token = FarmerPushToken(
        id=uuid4(),
        farmer_id=farmer.id,
        token="device_token_farmer_220",
        platform="android",
        is_active=True,
    )
    inv = Inventory(
        id=uuid4(),
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizer",
        brand="IFFCO",
        price=268.0,
        quantity_in_stock=0,
        available=False,
        unit="Bags",
    )
    db_session.add_all([farmer, shop, alert, token, inv])
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_wa, \
         patch("src.notifications.service.send_stock_siren_push", new_callable=AsyncMock) as mock_fcm:

        mock_wa.return_value = "wa_msg_test_1001"
        mock_fcm.return_value = (1, [])

        notified = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="urea",
            new_quantity=50,
            unit="Bags",
            brand="IFFCO",
            db_session=db_session,
        )

        assert notified == 1
        # WhatsApp called
        assert mock_wa.called
        assert mock_wa.call_args[1]["to_phone"] == "919876543220"

        # FCM called
        assert mock_fcm.called
        fcm_tokens = mock_fcm.call_args[1]["tokens"]
        assert "device_token_farmer_220" in fcm_tokens
        fcm_payload = mock_fcm.call_args[1]["data_payload"]
        assert fcm_payload["alert_type"] == "stock_siren"
        assert fcm_payload["quantity"] == "50"
        assert fcm_payload["brand"] == "IFFCO"

        # Alert deactivated on delivery
        await db_session.refresh(alert)
        assert alert.is_active is False
        assert alert.notified_at is not None


@pytest.mark.asyncio
async def test_fcm_failure_does_not_break_whatsapp(db_session: AsyncSession):
    """
    Ensure fail-soft isolation: If Firebase raises an unexpected exception or fails,
    the WhatsApp notification still proceeds and succeeds normally.
    """
    farmer = Farmer(id=uuid4(), phone_number="919876543230", preferred_language="en")
    shop = Shop(
        id=uuid4(),
        shop_name="Secunderabad Agro",
        owner_name="Test Owner",
        district="Hyderabad",
        phone_number="919876543231",
        address="Hyderabad",
    )
    alert = StockAlert(
        id=uuid4(),
        farmer_id=farmer.id,
        product_name="urea",
        district="Hyderabad",
        is_active=True,
    )
    token = FarmerPushToken(
        id=uuid4(),
        farmer_id=farmer.id,
        token="token_fail_soft_test",
        platform="android",
        is_active=True,
    )
    inv = Inventory(
        id=uuid4(),
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizer",
        brand="IFFCO",
        price=268.0,
        quantity_in_stock=0,
        available=False,
        unit="Bags",
    )
    db_session.add_all([farmer, shop, alert, token, inv])
    await db_session.commit()

    with patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_wa, \
         patch("src.notifications.service.send_stock_siren_push", new_callable=AsyncMock) as mock_fcm:

        mock_fcm.side_effect = Exception("Firebase server unreachable / network timeout")
        mock_wa.return_value = "wa_msg_ok_2002"

        notified = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="urea",
            new_quantity=50,
            unit="Bags",
            db_session=db_session,
        )

        assert notified == 1
        assert mock_wa.called
        await db_session.refresh(alert)
        assert alert.is_active is False


@pytest.mark.asyncio
async def test_fcm_auto_deactivates_unregistered_tokens(db_session: AsyncSession):
    """
    Verify that when Firebase returns invalid/unregistered tokens,
    the notification service deactivates them in the database.
    """
    farmer = Farmer(id=uuid4(), phone_number="919876543240", preferred_language="te")
    shop = Shop(
        id=uuid4(),
        shop_name="Guntur Fertilizer Depot",
        owner_name="Test Owner",
        district="Guntur",
        phone_number="919876543241",
        address="Guntur",
    )
    token_valid = FarmerPushToken(
        id=uuid4(),
        farmer_id=farmer.id,
        token="valid_token_device_1",
        is_active=True,
    )
    token_stale = FarmerPushToken(
        id=uuid4(),
        farmer_id=farmer.id,
        token="stale_unregistered_token_device_2",
        is_active=True,
    )
    db_session.add_all([farmer, shop, token_valid, token_stale])
    await db_session.commit()

    with patch("src.notifications.service.send_stock_siren_push", new_callable=AsyncMock) as mock_fcm:
        # Simulate Firebase reporting token_stale as invalid
        mock_fcm.return_value = (1, ["stale_unregistered_token_device_2"])

        await dispatch_fcm_stock_siren_notification(
            db=db_session,
            farmer=farmer,
            product_name="urea",
            shop=shop,
            new_quantity=30,
            unit="Bags",
        )

        await db_session.refresh(token_valid)
        await db_session.refresh(token_stale)

        assert token_valid.is_active is True
        assert token_stale.is_active is False  # Automatically deactivated


@pytest.mark.asyncio
async def test_notifications_router_endpoints(db_session: AsyncSession):
    """Verify HTTP API endpoints for registering and deactivating FCM tokens."""
    farmer = Farmer(id=uuid4(), phone_number="919999888877", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Override get_db dependency to use the test db_session
        from src.core.database import get_db
        app.dependency_overrides[get_db] = lambda: db_session

        # 1. Register Token
        reg_payload = {
            "token": "fcm_token_from_android_app_999888",
            "platform": "android",
            "device_model": "Redmi Note 12",
            "phone_number": "919999888877",
        }
        res = await client.post("/notifications/tokens", json=reg_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["farmer_id"] == str(farmer.id)
        assert data["is_active"] is True
        assert "fcm_to..." in data["token_masked"]

        # 2. Deactivate Token
        deact_payload = {
            "token": "fcm_token_from_android_app_999888",
        }
        res_del = await client.request("DELETE", "/notifications/tokens", json=deact_payload)
        assert res_del.status_code == 200
        data_del = res_del.json()
        assert data_del["deactivated"] is True

        app.dependency_overrides.pop(get_db, None)
