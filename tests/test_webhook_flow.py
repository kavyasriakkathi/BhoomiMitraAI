import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from sqlalchemy.exc import IntegrityError

from src.main import app
from src.core.models import Farmer
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

@pytest.mark.asyncio
async def test_process_message_pipeline_duplicate_stage1_skips_ai():
    """Verify that if a message_id is already in DB, Stage 1 immediately skips the pipeline."""
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DUPLICATE_STAGE1_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Where to buy urea?",
    )

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=True) as mock_dup, \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_farmer, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_dup.assert_awaited_once()
        mock_farmer.assert_not_called()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_pipeline_concurrent_duplicate_db_constraint_aborts():
    """
    Verify that if two requests race past Stage 1, the unique DB constraint in Stage 4
    rejects the second insert and exits immediately without calling Gemini or sending a reply.
    """
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.CONCURRENT_RACE_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Where to buy urea?",
    )

    mock_farmer_obj = Farmer(id=uuid.uuid4(), phone_number="919876543210")

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer_obj), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=None) as mock_store, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        await process_message_pipeline(parsed)

        mock_store.assert_awaited_once()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_store_incoming_message_catches_integrity_error():
    """Verify that store_incoming_message rolls back and returns None on IntegrityError."""
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("duplicate key value violates unique constraint", None, None)
    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210")
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.INTEGRITY_ERR_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Test integrity",
    )

    result = await store_incoming_message(mock_db, mock_farmer, parsed)
    assert result is None
    mock_db.rollback.assert_awaited_once()


@patch("src.gateway.router.process_message_pipeline")
def test_post_webhook_duplicate_messages_in_payload_queued_once(mock_pipeline):
    """Verify that if the same message_id appears multiple times in a webhook payload, it is only queued once."""
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
                                    "id": "wamid.DUPLICATE_BATCH_ID_999",
                                    "timestamp": "1700000000",
                                    "text": {"body": "First instance"},
                                    "type": "text"
                                },
                                {
                                    "from": "919876543210",
                                    "id": "wamid.DUPLICATE_BATCH_ID_999",
                                    "timestamp": "1700000000",
                                    "text": {"body": "Second duplicate instance"},
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


@pytest.mark.asyncio
async def test_process_message_pipeline_in_flight_lock_aborts_concurrent_run():
    """Verify that if a message ID is already in-flight, a second concurrent task aborts at Stage 0."""
    from src.gateway.service import _IN_FLIGHT_MESSAGE_IDS

    msg_id = "wamid.CONCURRENT_IN_FLIGHT_TEST"
    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id=msg_id,
        timestamp="1700000000",
        message_type="text",
        text_content="Multi intent query",
    )

    _IN_FLIGHT_MESSAGE_IDS.add(msg_id)
    try:
        with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock) as mock_dup, \
             patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock) as mock_farmer, \
             patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

            await process_message_pipeline(parsed)

            mock_dup.assert_not_called()
            mock_farmer.assert_not_called()
            mock_send.assert_not_called()
    finally:
        _IN_FLIGHT_MESSAGE_IDS.discard(msg_id)

