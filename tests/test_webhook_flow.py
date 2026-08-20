import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.main import app
from src.config import get_settings

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
