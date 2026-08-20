import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime
from src.main import app
from src.conversation.schemas import ConversationResponse
from src.conversation.service import ConversationService
from src.conversation.dependencies import get_conversation_service

client = TestClient(app)

MOCK_TIMESTAMP = "2023-01-01T00:00:00Z"


@pytest.fixture
def mock_conversation_service():
    service = AsyncMock(spec=ConversationService)
    app.dependency_overrides[get_conversation_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_response(**overrides) -> ConversationResponse:
    """Helper to build a ConversationResponse with sensible defaults."""
    defaults = dict(
        id=uuid4(),
        farmer_id=uuid4(),
        message_id="wamid_test_001",
        user_message="My crop is turning yellow",
        user_message_type="text",
        ai_response="It could be a nitrogen deficiency.",
        intent="crop_disease",
        confidence_score=0.92,
        outbound_message_id=None,
        delivery_status="pending",
        created_at=MOCK_TIMESTAMP,
    )
    defaults.update(overrides)
    return ConversationResponse(**defaults)


# ---- CREATE ----

def test_create_conversation(mock_conversation_service):
    conv = _mock_response()
    mock_conversation_service.create_conversation.return_value = conv

    response = client.post("/conversations", json={
        "farmer_id": str(conv.farmer_id),
        "message_id": conv.message_id,
        "user_message": conv.user_message,
        "user_message_type": "text",
    })

    assert response.status_code == 201
    assert response.json()["message_id"] == conv.message_id
    assert response.json()["farmer_id"] == str(conv.farmer_id)


def test_create_conversation_missing_message_id():
    """POST /conversations without message_id should return 422."""
    response = client.post("/conversations", json={
        "farmer_id": str(uuid4()),
    })
    assert response.status_code == 422


def test_create_conversation_invalid_message_type():
    """POST /conversations with invalid user_message_type should return 422."""
    response = client.post("/conversations", json={
        "farmer_id": str(uuid4()),
        "message_id": "wamid_test_002",
        "user_message_type": "video",
    })
    assert response.status_code == 422


def test_create_conversation_invalid_confidence_too_high():
    """Confidence score > 1.0 should return 422."""
    response = client.post("/conversations", json={
        "farmer_id": str(uuid4()),
        "message_id": "wamid_test_003",
        "confidence_score": 1.5,
    })
    assert response.status_code == 422


def test_create_conversation_invalid_confidence_negative():
    """Negative confidence score should return 422."""
    response = client.post("/conversations", json={
        "farmer_id": str(uuid4()),
        "message_id": "wamid_test_004",
        "confidence_score": -0.1,
    })
    assert response.status_code == 422


# ---- READ (single) ----

def test_get_conversation(mock_conversation_service):
    conv = _mock_response()
    mock_conversation_service.get_conversation.return_value = conv

    response = client.get(f"/conversations/{conv.id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(conv.id)


# ---- READ (list) ----

def test_get_conversations(mock_conversation_service):
    mock_conversation_service.get_conversations.return_value = (0, [])

    response = client.get("/conversations?page=1&size=10")

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []


def test_get_farmer_conversations(mock_conversation_service):
    farmer_id = uuid4()
    conv = _mock_response(farmer_id=farmer_id)
    mock_conversation_service.get_farmer_conversations.return_value = (1, [conv])

    response = client.get(f"/conversations/farmer/{farmer_id}?page=1&size=10")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["farmer_id"] == str(farmer_id)


# ---- UPDATE ----

def test_update_conversation(mock_conversation_service):
    conv = _mock_response(delivery_status="delivered")
    mock_conversation_service.update_conversation.return_value = conv

    response = client.put(f"/conversations/{conv.id}", json={
        "delivery_status": "delivered",
    })

    assert response.status_code == 200
    assert response.json()["delivery_status"] == "delivered"


def test_update_conversation_invalid_delivery_status():
    """PUT /conversations/{id} with invalid delivery_status should return 422."""
    conv_id = uuid4()
    response = client.put(f"/conversations/{conv_id}", json={
        "delivery_status": "exploded",
    })
    assert response.status_code == 422


# ---- DELETE ----

def test_delete_conversation(mock_conversation_service):
    conv_id = uuid4()
    mock_conversation_service.delete_conversation.return_value = None

    response = client.delete(f"/conversations/{conv_id}")

    assert response.status_code == 204
