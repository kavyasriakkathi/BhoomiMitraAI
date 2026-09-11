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

