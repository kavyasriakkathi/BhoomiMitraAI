import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from src.main import app
from src.config import get_settings
from src.core.database import Base
from src.core.models import Farmer, Conversation
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import (
    store_incoming_message,
    process_message_pipeline,
)

client = TestClient(app)


@patch("src.gateway.router.get_settings")
def test_webhook_verification_success(mock_get_settings):
    mock_get_settings.return_value.whatsapp_verify_token = "bhoomimitra_verify_2026"
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "bhoomimitra_verify_2026",
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 200
    assert response.text == "115820120"


def test_webhook_verification_failure():
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "115820120",
        },
    )
    assert response.status_code == 403


def test_whatsapp_health_endpoint():
    response = client.get("/webhook/whatsapp/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "whatsapp_configured" in data["data"]
    assert "phone_number_id_configured" in data["data"]
    assert "phone_number_id" in data["data"]
    assert "gemini_configured" in data["data"]


@patch("src.gateway.router.process_message_pipeline")
def test_post_webhook_text_message_success(mock_pipeline):
    mock_pipeline.return_value = None

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1993680168018884",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "1211671805365875"
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Farmer"},
                                    "wa_id": "919876543210"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.HBgLTEST12345",
                                    "timestamp": "1700000000",
                                    "text": {"body": "Hi BhoomiMitra test"},
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }

    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["messages_queued"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TESTS: Duplicate Message & Webhook Concurrency Race Protection
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy.pool import StaticPool


@pytest.mark.asyncio
async def test_store_incoming_message_duplicate_rollback():
    """
    Verify that calling store_incoming_message with an existing message_id
    catches IntegrityError, executes rollback, returns None, and leaves the session clean.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with test_session_maker() as db:
        farmer = Farmer(phone_number="+919876543210")
        db.add(farmer)
        await db.commit()
        await db.refresh(farmer)

        msg = ParsedIncomingMessage(
            phone_number="+919876543210",
            message_id="wamid.DUPLICATE_ROLLBACK_TEST",
            timestamp="1700000000",
            message_type="text",
            text_content="First delivery",
        )

        # 1. First insert should succeed
        conv1 = await store_incoming_message(db, farmer, msg)
        assert conv1 is not None
        assert conv1.message_id == "wamid.DUPLICATE_ROLLBACK_TEST"

        # 2. Duplicate insert with same message_id should catch IntegrityError, rollback, and return None
        conv2 = await store_incoming_message(db, farmer, msg)
        assert conv2 is None

        # 3. Verify session is NOT in PendingRollbackError state and can execute subsequent queries
        res = await db.execute(
            select(Conversation).where(Conversation.message_id == "wamid.DUPLICATE_ROLLBACK_TEST")
        )
        saved_convs = list(res.scalars().all())
        assert len(saved_convs) == 1

    await test_engine.dispose()


@pytest.mark.asyncio
@patch("src.gateway.service.send_text_message")
@patch("src.gateway.service.process_text_message")
@patch("src.gateway.service.mark_message_as_read")
async def test_process_message_pipeline_concurrent_duplicate_race(
    mock_mark_read,
    mock_process_text,
    mock_send_text,
    tmp_path,
):
    """
    Verify that when two concurrent background pipelines run for the exact same message_id:
    - Only ONE conversation row is saved in the database
    - AI processing (process_text_message) is executed only once
    - WhatsApp outbound message (send_text_message) is dispatched only once
    - No IntegrityError or PendingRollbackError escapes unhandled
    """
    db_file = tmp_path / "test_concurrent_webhook.db"
    test_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    test_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Pre-seed the farmer in database
    async with test_session_maker() as db:
        farmer = Farmer(phone_number="+919111222333")
        db.add(farmer)
        await db.commit()

    mock_send_text.return_value = "outbound_meta_wamid_999"
    mock_process_text.return_value = "Namaste, this is BhoomiMitra."
    mock_mark_read.return_value = None

    msg1 = ParsedIncomingMessage(
        phone_number="+919111222333",
        message_id="wamid.CONCURRENT_RACE_TEST_101",
        timestamp="1700000001",
        message_type="text",
        text_content="Tomato leaf curl disease question",
        sender_name="Srinivas",
    )
    msg2 = ParsedIncomingMessage(
        phone_number="+919111222333",
        message_id="wamid.CONCURRENT_RACE_TEST_101",
        timestamp="1700000001",
        message_type="text",
        text_content="Tomato leaf curl disease question",
        sender_name="Srinivas",
    )

    with patch("src.gateway.service.AsyncSessionLocal", test_session_maker):
        # Run both tasks concurrently
        await asyncio.gather(
            process_message_pipeline(msg1, "Srinivas"),
            process_message_pipeline(msg2, "Srinivas"),
        )

    # Verify database state
    async with test_session_maker() as db:
        res = await db.execute(
            select(Conversation).where(Conversation.message_id == "wamid.CONCURRENT_RACE_TEST_101")
        )
        conversations = list(res.scalars().all())
        assert len(conversations) == 1
        assert conversations[0].delivery_status == "sent"
        assert conversations[0].outbound_message_id == "outbound_meta_wamid_999"

    # Verify AI generation and outbound dispatch were only executed once
    assert mock_process_text.call_count == 1
    assert mock_send_text.call_count == 1

    await test_engine.dispose()
