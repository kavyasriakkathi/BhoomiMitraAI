"""
Unit and integration tests for Stock Siren (Urgent Stock Alert with WhatsApp Audio).
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from src.core.database import Base
from src.language.service import LanguageService, synthesize_speech
from src.gateway.whatsapp_client import upload_media_bytes, send_audio_message


@pytest_asyncio.fixture
async def db_session():
    """Provides isolated in-memory SQLite async database session for tests."""
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



# =============================================================================
# 1. ISOLATED TTS TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_synthesize_speech_empty_or_whitespace():
    """Empty or whitespace text returns None immediately."""
    assert await synthesize_speech("") is None
    assert await synthesize_speech("   ") is None
    assert await synthesize_speech(None) is None


@pytest.mark.asyncio
async def test_synthesize_speech_success():
    """Valid text synthesizes OGG_OPUS audio via Google Cloud TTS."""
    mock_response = MagicMock()
    mock_response.audio_content = b"OGG_OPUS_AUDIO_BYTES_TEST"

    mock_client = MagicMock()
    mock_client.synthesize_speech = AsyncMock(return_value=mock_response)

    with patch.object(LanguageService, "google_tts_client", new_callable=lambda: mock_client):
        service = LanguageService()
        result = await service.synthesize_speech(
            text="యూరియా స్టాక్ అందుబాటులోకి వచ్చింది",
            language_code="te-IN"
        )
        assert result == b"OGG_OPUS_AUDIO_BYTES_TEST"
        assert mock_client.synthesize_speech.called


@pytest.mark.asyncio
async def test_synthesize_speech_fails_soft_on_error():
    """TTS failure catches exceptions and returns None safely."""
    mock_client = MagicMock()
    mock_client.synthesize_speech = AsyncMock(side_effect=RuntimeError("Google TTS quota exceeded"))

    with patch.object(LanguageService, "google_tts_client", new_callable=lambda: mock_client):
        service = LanguageService()
        result = await service.synthesize_speech(text="యూరియా", language_code="te-IN")
        assert result is None


# =============================================================================
# 2. ISOLATED WHATSAPP OUTBOUND MEDIA & AUDIO TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_upload_media_bytes_empty():
    """Empty payload returns None immediately."""
    assert await upload_media_bytes(b"") is None
    assert await upload_media_bytes(None) is None


@pytest.mark.asyncio
async def test_upload_media_bytes_success():
    """Successful media upload returns Meta media_id."""
    fake_response = Response(
        status_code=200,
        json={"id": "media-wa-test-999"},
        request=MagicMock()
    )

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_phone_number_id = "mock_phone_id"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 10.0
        mock_post.return_value = fake_response

        media_id = await upload_media_bytes(b"RAW_AUDIO_CONTENT", mime_type="audio/ogg")
        assert media_id == "media-wa-test-999"
        assert mock_post.called


@pytest.mark.asyncio
async def test_upload_media_bytes_fails_soft():
    """Upload failure logs error and returns None without raising."""
    fake_response = Response(
        status_code=500,
        text="Internal Server Error",
        request=MagicMock()
    )

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_phone_number_id = "mock_phone_id"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 10.0
        mock_post.return_value = fake_response

        media_id = await upload_media_bytes(b"RAW_AUDIO_CONTENT")
        assert media_id is None


@pytest.mark.asyncio
async def test_send_audio_message_validation():
    """Missing recipient or media references returns None."""
    assert await send_audio_message("") is None
    assert await send_audio_message("919876543210") is None


@pytest.mark.asyncio
async def test_send_audio_message_with_media_id():
    """Sends audio message payload with media_id."""
    fake_response = Response(
        status_code=200,
        json={"messages": [{"id": "wam-audio-12345"}]},
        request=MagicMock()
    )

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_phone_number_id = "mock_phone_id"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 10.0
        mock_post.return_value = fake_response

        res = await send_audio_message(
            to_phone="919876543210",
            media_id="media-wa-test-999"
        )
        assert res == "wam-audio-12345"
        kwargs = mock_post.call_args[1]
        assert kwargs["json"]["type"] == "audio"
        assert kwargs["json"]["audio"]["id"] == "media-wa-test-999"
        assert kwargs["json"]["to"] == "919876543210"


@pytest.mark.asyncio
async def test_send_audio_message_fails_soft():
    """Audio send failure returns None without raising."""
    fake_response = Response(
        status_code=400,
        text="Bad Request",
        request=MagicMock()
    )

    with patch("src.gateway.whatsapp_client.get_settings") as mock_settings, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_settings.return_value.whatsapp_api_token = "mock_token"
        mock_settings.return_value.whatsapp_phone_number_id = "mock_phone_id"
        mock_settings.return_value.whatsapp_api_timeout_seconds = 10.0
        mock_post.return_value = fake_response

        res = await send_audio_message(
            to_phone="919876543210",
            media_id="media-wa-test-999"
        )
        assert res is None


# =============================================================================
# 3. STOCK SIREN INTEGRATION TESTS
# =============================================================================

from src.core.models import Farmer, Shop, Inventory, StockAlert
from src.shops.stock_alerts import (
    create_or_reactivate_alert,
    trigger_stock_alert_notifications,
    list_farmer_alerts,
)


@pytest.mark.asyncio
async def test_stock_siren_triggers_text_and_audio(db_session):
    """
    Restock triggers urgent Stock Siren text with urgency warning,
    verified quantity, updated timestamp, and sends WhatsApp audio.
    """
    farmer = Farmer(phone_number="919876549901", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(
        db_session, farmer, "urea", "Warangal"
    )
    assert alert.is_active is True

    shop = Shop(
        shop_name="Rythu Bandhu Agro",
        owner_name="Suresh",
        phone_number="9876540001",
        address="Market Yard, Warangal",
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

    with patch("src.language.service.synthesize_speech", new_callable=AsyncMock) as mock_tts, \
         patch("src.gateway.whatsapp_client.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
         patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_text, \
         patch("src.gateway.whatsapp_client.send_audio_message", new_callable=AsyncMock) as mock_audio:

        mock_tts.return_value = b"FAKE_OGG_BYTES"
        mock_upload.return_value = "media-wa-siren-100"
        mock_text.return_value = "wam-text-siren-100"
        mock_audio.return_value = "wam-audio-siren-100"

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
        assert mock_tts.called
        assert mock_upload.called
        assert mock_text.call_count == 1
        assert mock_audio.call_count == 1

        # Verify urgent text notification contents
        sent_text = mock_text.call_args[1]["message_text"]
        assert "🚨🔔" in sent_text
        assert "Stock Siren" in sent_text
        assert "Rythu Bandhu Agro" in sent_text
        assert "50 bag" in sent_text
        assert "స్టాక్ అప్‌డేట్ సమయం" in sent_text
        assert "స్టాక్ త్వరగా అయిపోయే అవకాశం ఉంది" in sent_text

        # Verify audio call parameters
        audio_kwargs = mock_audio.call_args[1]
        assert audio_kwargs["to_phone"] == "919876549901"
        assert audio_kwargs["media_id"] == "media-wa-siren-100"

    # Alert should be marked inactive
    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_stock_siren_audio_fails_soft_preserves_text(db_session):
    """
    If TTS synthesis or media upload fails, text delivery still succeeds,
    and alert is properly deactivated.
    """
    farmer = Farmer(phone_number="919876549902", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")

    shop = Shop(
        shop_name="Kisan Agro Warangal",
        owner_name="Srinivas",
        phone_number="9876540002",
        address="Station Road, Warangal",
        district="Warangal",
        state="Telangana",
        status="active",
    )
    db_session.add(shop)
    await db_session.commit()

    inv = Inventory(
        shop_id=shop.id,
        product_name="Urea",
        category="Fertilizers",
        brand="KRIBHCO",
        unit="bag",
        price=268.0,
        quantity_in_stock=0,
        available=False,
    )
    db_session.add(inv)
    await db_session.commit()

    with patch("src.language.service.synthesize_speech", new_callable=AsyncMock) as mock_tts, \
         patch("src.gateway.whatsapp_client.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
         patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_text, \
         patch("src.gateway.whatsapp_client.send_audio_message", new_callable=AsyncMock) as mock_audio:

        # Audio synthesis fails
        mock_tts.return_value = None
        mock_text.return_value = "wam-text-only-200"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=30,
            unit="bag",
            brand="KRIBHCO",
            db_session=db_session,
        )

        assert notified_count == 1
        assert mock_text.call_count == 1
        assert mock_audio.call_count == 0  # Not attempted because media_id is None

    alerts = await list_farmer_alerts(db_session, farmer)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_stock_siren_batch_audio_caching(db_session):
    """
    When multiple farmers match a restock event, TTS synthesis and
    Meta media upload are performed only ONCE for the entire batch.
    """
    farmer1 = Farmer(phone_number="919876549911", preferred_language="te")
    farmer2 = Farmer(phone_number="919876549912", preferred_language="te")
    db_session.add_all([farmer1, farmer2])
    await db_session.commit()

    await create_or_reactivate_alert(db_session, farmer1, "urea", "Warangal")
    await create_or_reactivate_alert(db_session, farmer2, "urea", "Warangal")

    shop = Shop(
        shop_name="Telangana Agro Warangal",
        owner_name="Anil",
        phone_number="9876540003",
        address="Bus Stand, Warangal",
        district="Warangal",
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

    with patch("src.language.service.synthesize_speech", new_callable=AsyncMock) as mock_tts, \
         patch("src.gateway.whatsapp_client.upload_media_bytes", new_callable=AsyncMock) as mock_upload, \
         patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_text, \
         patch("src.gateway.whatsapp_client.send_audio_message", new_callable=AsyncMock) as mock_audio:

        mock_tts.return_value = b"BATCH_OGG_BYTES"
        mock_upload.return_value = "media-batch-300"
        mock_text.return_value = "wam-text-batch"
        mock_audio.return_value = "wam-audio-batch"

        notified_count = await trigger_stock_alert_notifications(
            inventory_item_id=inv.id,
            shop_id=shop.id,
            product_name="Urea",
            new_quantity=100,
            unit="bag",
            brand="IFFCO",
            db_session=db_session,
        )

        assert notified_count == 2
        # Crucial verification: TTS and upload called only ONCE for the batch!
        assert mock_tts.call_count == 1
        assert mock_upload.call_count == 1
        # Text and audio sent to both farmers
        assert mock_text.call_count == 2
        assert mock_audio.call_count == 2


# =============================================================================
# 4. STOCK SIREN OUTSIDE-24H TEMPLATE FALLBACK & RACE RECOVERY TESTS
# =============================================================================

from src.shops.stock_alerts import handle_stock_alert_status_update


@pytest.mark.asyncio
async def test_stock_siren_webhook_131047_retry_success(db_session):
    """
    Standard asynchronous 131047 fallback:
    Webhook reports error 131047 with valid outbound metadata in Redis.
    Dispatches approved template, marks alert inactive, and caches correlation.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540005", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {
        "stock_alert_outbound:wamid.prod.text.01": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-test-id",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "11:00 AM",
            "used_template": False,
        })
    }

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))
    mock_redis.set = AsyncMock(side_effect=lambda k, v, **kw: redis_store.update({k: v}) or True)
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    status_payload = {
        "id": "wamid.prod.text.01",
        "status": "failed",
        "recipient_id": "919876540005",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        mock_tmpl.return_value = "wamid.prod.tmpl.01"

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is True
        assert mock_tmpl.call_count == 1
        assert mock_tmpl.call_args[1]["to_phone"] == "919876540005"
        params = mock_tmpl.call_args[1]["parameters"]
        assert params[0] == "Urea Fertilizer 45kg"
        assert params[3] == "50 bag"

    await db_session.refresh(alert)
    assert alert.is_active is False
    assert alert.notified_at is not None


@pytest.mark.asyncio
async def test_stock_siren_webhook_race_metadata_delayed_then_recovered(db_session):
    """
    Part 5 Tests 1 & 2: Webhook arrives before outbound Redis write completes.
    Initially Redis returns None, but metadata becomes available on retry.
    The webhook recovers metadata and successfully dispatches template fallback.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540008", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {}
    get_calls = 0

    async def mock_redis_get(key):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            return None
        return redis_store.get(key)

    async def mock_redis_set(key, val, **kwargs):
        if kwargs.get("nx") and key in redis_store:
            return False
        redis_store[key] = val
        return True

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=mock_redis_get)
    mock_redis.set = AsyncMock(side_effect=mock_redis_set)
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    redis_store["stock_alert_outbound:wamid.race.01"] = json.dumps({
        "alert_id": str(alert.id),
        "farmer_id": str(farmer.id),
        "inventory_item_id": "inv-test-race",
        "phone_number": farmer.phone_number,
        "product_name": "Urea Fertilizer 45kg",
        "shop_name": "Warangal Agri Hub",
        "district": "Warangal",
        "new_quantity": 50,
        "unit": "bag",
        "updated_time_str": "11:30 AM",
        "used_template": False,
    })

    status_payload = {
        "id": "wamid.race.01",
        "status": "failed",
        "recipient_id": "919876540008",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        mock_tmpl.return_value = "wamid.tmpl.race.success"
        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)

        assert res is True
        assert mock_tmpl.call_count == 1
        assert get_calls >= 2

    await db_session.refresh(alert)
    assert alert.is_active is False
    assert alert.notified_at is not None


@pytest.mark.asyncio
async def test_stock_siren_webhook_permanent_metadata_loss_does_not_reactivate_multiple_alerts(db_session):
    """
    Requirement 7: Complete Redis metadata loss after bounded retries:
    - Does NOT reactivate multiple alerts for the farmer.
    - Two different product alerts ('urea' and 'dap') for the same farmer remain unchanged.
    - Does NOT send an uncorrelated template to the farmer.
    - Logs correlation failure and safely returns False.
    """
    from datetime import datetime
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540009", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    # Farmer has two different product alerts recently notified
    alert_urea, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert_urea.is_active = False
    alert_urea.notified_at = datetime.utcnow()

    alert_dap, _ = await create_or_reactivate_alert(db_session, farmer, "dap", "Warangal")
    alert_dap.is_active = False
    alert_dap.notified_at = datetime.utcnow()

    db_session.add_all([alert_urea, alert_dap])
    await db_session.commit()

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)  # Outbound metadata completely unavailable

    status_payload = {
        "id": "wamid.missing.01",
        "status": "failed",
        "recipient_id": "919876540009",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is False
        # Crucial safety: No uncorrelated template was sent!
        assert mock_tmpl.call_count == 0

    # Crucial safety: Alerts remain untouched, preventing arbitrary guessing/bulk reactivation
    await db_session.refresh(alert_urea)
    await db_session.refresh(alert_dap)
    assert alert_urea.is_active is False
    assert alert_dap.is_active is False


@pytest.mark.asyncio
async def test_stock_siren_webhook_concurrent_workers_prevents_duplicate_template(db_session):
    """
    Part 5 Tests 5, 6, 7, 9: Two workers process duplicate 131047 webhooks simultaneously.
    Atomic Redis claim (stock_alert_template_fallback:{msg_id} with NX=True) guarantees
    only Worker A acquires the lock and sends the template; Worker B safely skips sending.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540010", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {
        "stock_alert_outbound:wamid.dup.01": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-dup-test",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "12:00 PM",
            "used_template": False,
        })
    }

    async def mock_redis_set(key, val, **kwargs):
        if kwargs.get("nx") and key in redis_store:
            return False
        redis_store[key] = val
        return True

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))
    mock_redis.set = AsyncMock(side_effect=mock_redis_set)
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    status_payload = {
        "id": "wamid.dup.01",
        "status": "failed",
        "recipient_id": "919876540010",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        mock_tmpl.return_value = "wamid.tmpl.dup.success"

        res_worker_a = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        res_worker_b = await handle_stock_alert_status_update(status_payload, db_session=db_session)

        assert res_worker_a is True
        assert res_worker_b is True
        assert mock_tmpl.call_count == 1
        assert redis_store.get("stock_alert_template_fallback:wamid.dup.01") == "completed"


@pytest.mark.asyncio
async def test_stock_siren_webhook_duplicate_after_success_skips_sending(db_session):
    """
    Part 5 Test 10: Duplicate webhook arrives after template fallback has already succeeded.
    Must not send another template message.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540012", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {
        "stock_alert_outbound:wamid.orig.success.01": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-dup-success",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "01:00 PM",
            "used_template": True,
            "fallback_template_wa_id": "wamid.tmpl.already.sent",
        }),
        "stock_alert_template_fallback:wamid.orig.success.01": "completed",
    }

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))

    status_payload = {
        "id": "wamid.orig.success.01",
        "status": "failed",
        "recipient_id": "919876540012",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is True
        assert mock_tmpl.call_count == 0


@pytest.mark.asyncio
async def test_stock_siren_webhook_template_failure_releases_fallback_lock(db_session):
    """
    Part 5 Test 8: When template send fails, the fallback lock must be deleted
    from Redis and the StockAlert reactivated, allowing a controlled future retry.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540011", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {
        "stock_alert_outbound:wamid.fail.01": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-fail-test",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "12:30 PM",
            "used_template": False,
        })
    }

    async def mock_redis_set(key, val, **kwargs):
        if kwargs.get("nx") and key in redis_store:
            return False
        redis_store[key] = val
        return True

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))
    mock_redis.set = AsyncMock(side_effect=mock_redis_set)
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    status_payload = {
        "id": "wamid.fail.01",
        "status": "failed",
        "recipient_id": "919876540011",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        mock_tmpl.return_value = None

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is False
        assert mock_tmpl.call_count == 1
        assert "stock_alert_template_fallback:wamid.fail.01" not in redis_store

    await db_session.refresh(alert)
    assert alert.is_active is True
    assert alert.notified_at is None


@pytest.mark.asyncio
async def test_stock_siren_webhook_template_own_131047_does_not_recurse(db_session):
    """
    Part 5 Test 11: The template message itself reports 131047.
    Must not recursively send another template. Alert must remain active and lock cleared.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540013", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    redis_store = {
        "stock_alert_outbound:wamid.tmpl.failed.01": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-recurse-test",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "01:30 PM",
            "used_template": True,
        }),
        f"stock_alert_lock:{alert.id}:inv-recurse-test": "1",
    }

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    status_payload = {
        "id": "wamid.tmpl.failed.01",
        "status": "failed",
        "recipient_id": "919876540013",
        "errors": [{"code": 131047, "message": "24-hour window expired"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is False
        assert mock_tmpl.call_count == 0

    await db_session.refresh(alert)
    assert alert.is_active is True
    assert alert.notified_at is None
    assert f"stock_alert_lock:{alert.id}:inv-recurse-test" not in redis_store


@pytest.mark.asyncio
async def test_stock_siren_webhook_non_131047_failure_reactivates_alert_without_template(db_session):
    """
    Part 5 Test 13: Handle Meta error 131047 specifically.
    When a non-131047 error (e.g. 131026 undeliverable) is reported by Meta webhook,
    template retry is NOT triggered and the alert is reactivated.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540006", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    alert.is_active = False
    db_session.add(alert)
    await db_session.commit()

    mock_redis = MagicMock()
    redis_store = {
        "stock_alert_outbound:wamid.prod.text.02": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-test-id",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "11:00 AM",
            "used_template": False,
        })
    }

    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))
    mock_redis.delete = AsyncMock(side_effect=lambda k: redis_store.pop(k, None))

    status_payload = {
        "id": "wamid.prod.text.02",
        "status": "failed",
        "recipient_id": "919876540006",
        "errors": [{"code": 131026, "message": "Message Undeliverable"}],
    }

    with patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl, \
         patch("redis.asyncio.from_url", return_value=mock_redis):

        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is False
        assert mock_tmpl.call_count == 0

    await db_session.refresh(alert)
    assert alert.is_active is True
    assert alert.notified_at is None


@pytest.mark.asyncio
async def test_stock_siren_webhook_confirmed_delivery_status(db_session):
    """
    When Meta webhook reports 'delivered' or 'sent', confirms successful delivery
    and marks StockAlert as deactivated.
    """
    import json
    from src.core.models import Farmer
    from src.shops.stock_alerts import create_or_reactivate_alert

    farmer = Farmer(phone_number="919876540007", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")

    mock_redis = MagicMock()
    redis_store = {
        "stock_alert_outbound:wamid.prod.text.03": json.dumps({
            "alert_id": str(alert.id),
            "farmer_id": str(farmer.id),
            "inventory_item_id": "inv-test-id",
            "phone_number": farmer.phone_number,
            "product_name": "Urea Fertilizer 45kg",
            "shop_name": "Warangal Agri Hub",
            "district": "Warangal",
            "new_quantity": 50,
            "unit": "bag",
            "updated_time_str": "11:00 AM",
            "used_template": False,
        })
    }

    mock_redis.get = AsyncMock(side_effect=lambda k: redis_store.get(k))

    status_payload = {
        "id": "wamid.prod.text.03",
        "status": "delivered",
        "recipient_id": "919876540007",
    }

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        res = await handle_stock_alert_status_update(status_payload, db_session=db_session)
        assert res is True

    await db_session.refresh(alert)
    assert alert.is_active is False
    assert alert.notified_at is not None


@pytest.mark.asyncio
async def test_stock_siren_synchronous_131047_fallback(db_session):
    """
    Part 5 Test 12: Existing synchronous 131047 fallback still works.
    When send_text_message raises WhatsApp24HourWindowExceeded, send_template_message is invoked.
    """
    from src.core.models import Farmer, Shop, Inventory
    from src.shops.stock_alerts import create_or_reactivate_alert, trigger_stock_alert_notifications
    from src.gateway.whatsapp_client import WhatsApp24HourWindowExceeded

    farmer = Farmer(phone_number="919876549999", preferred_language="te")
    db_session.add(farmer)
    await db_session.commit()

    alert, _ = await create_or_reactivate_alert(db_session, farmer, "urea", "Warangal")
    shop = Shop(
        shop_name="Agri Care",
        owner_name="Ramu",
        phone_number="9876549999",
        address="Warangal",
        district="Warangal",
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

    with patch("src.language.service.synthesize_speech", new_callable=AsyncMock) as mock_tts, \
         patch("src.gateway.whatsapp_client.send_text_message", new_callable=AsyncMock) as mock_text, \
         patch("src.gateway.whatsapp_client.send_template_message", new_callable=AsyncMock) as mock_tmpl:

        mock_tts.return_value = None
        mock_text.side_effect = WhatsApp24HourWindowExceeded("24-hour window expired (Meta error 131047)")
        mock_tmpl.return_value = "wamid.sync.tmpl.01"

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
        assert mock_text.call_count == 1
        assert mock_tmpl.call_count == 1

    await db_session.refresh(alert)
    assert alert.is_active is False
    assert alert.notified_at is not None


def test_stock_siren_dosage_safety_remains_unchanged():
    """
    Part 5 Test 15: Verify that chemical dosage and safety logic remains completely unchanged.
    System prompt rules and dosage-sensitive query classification must remain strictly intact.
    """
    from src.ai.prompts import BHOOMIMITRA_SYSTEM_PROMPT
    from src.ai.service import is_dosage_sensitive_query

    assert "NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages" in BHOOMIMITRA_SYSTEM_PROMPT
    assert "I am not 100% sure about the exact dosage. Please consult your local agriculture officer" in BHOOMIMITRA_SYSTEM_PROMPT

    assert is_dosage_sensitive_query("వరిలో యూరియా ఎంత వేయాలి?") is True
    assert is_dosage_sensitive_query("How many kg urea per acre for paddy?") is True
    assert is_dosage_sensitive_query("What is the market price of tomato?") is False
