import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from src.main import app
from src.farmers.schemas import FarmerResponse, FarmerCreate
from src.farmers.service import FarmerService
from src.farmers.dependencies import get_farmer_service

client = TestClient(app)

@pytest.fixture
def mock_farmer_service():
    service = AsyncMock(spec=FarmerService)
    app.dependency_overrides[get_farmer_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def test_create_farmer(mock_farmer_service):
    farmer_id = uuid4()
    mock_farmer = FarmerResponse(
        id=farmer_id,
        phone_number="+11234567890",
        preferred_language="te",
        is_active=True,
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )
    mock_farmer_service.create_farmer.return_value = mock_farmer

    response = client.post("/farmers", json={"phone_number": "+11234567890", "preferred_language": "te", "is_active": True})
    
    assert response.status_code == 201
    assert response.json()["id"] == str(farmer_id)
    assert response.json()["phone_number"] == "+11234567890"

def test_create_farmer_validation_error():
    response = client.post("/farmers", json={"phone_number": "invalid_phone"})
    assert response.status_code == 422

def test_get_farmer(mock_farmer_service):
    farmer_id = uuid4()
    mock_farmer = FarmerResponse(
        id=farmer_id,
        phone_number="+11234567890",
        preferred_language="te",
        is_active=True,
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )
    mock_farmer_service.get_farmer.return_value = mock_farmer

    response = client.get(f"/farmers/{farmer_id}")
    
    assert response.status_code == 200
    assert response.json()["id"] == str(farmer_id)

def test_get_farmers(mock_farmer_service):
    mock_farmer_service.get_farmers.return_value = (0, [])
    response = client.get("/farmers?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []

def test_update_farmer(mock_farmer_service):
    farmer_id = uuid4()
    mock_farmer = FarmerResponse(
        id=farmer_id,
        phone_number="+19876543210",
        preferred_language="en",
        is_active=False,
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )
    mock_farmer_service.update_farmer.return_value = mock_farmer

    response = client.put(f"/farmers/{farmer_id}", json={"preferred_language": "en", "is_active": False})
    
    assert response.status_code == 200
    assert response.json()["preferred_language"] == "en"
    assert response.json()["is_active"] == False

def test_delete_farmer(mock_farmer_service):
    farmer_id = uuid4()
    mock_farmer_service.delete_farmer.return_value = None

    response = client.delete(f"/farmers/{farmer_id}")
    
    assert response.status_code == 204


# --- Phone number validation tests for PUT /farmers/{id} ---

def test_update_farmer_valid_phone(mock_farmer_service):
    """PUT /farmers/{id} with a valid E.164 phone number should succeed."""
    farmer_id = uuid4()
    mock_farmer = FarmerResponse(
        id=farmer_id,
        phone_number="+919789658731",
        preferred_language="te",
        is_active=True,
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )
    mock_farmer_service.update_farmer.return_value = mock_farmer

    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": "+919789658731"})

    assert response.status_code == 200
    assert response.json()["phone_number"] == "+919789658731"


def test_update_farmer_invalid_phone_short():
    """PUT /farmers/{id} with '150' (too short, no '+') should return 422."""
    farmer_id = uuid4()
    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": "150"})

    assert response.status_code == 422


def test_update_farmer_invalid_phone_too_short_with_plus():
    """PUT /farmers/{id} with '+2015184' (too few digits) should return 422."""
    farmer_id = uuid4()
    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": "+2015184"})

    assert response.status_code == 422


def test_update_farmer_invalid_phone_no_plus():
    """PUT /farmers/{id} with '917896587312' (missing '+' prefix) should return 422."""
    farmer_id = uuid4()
    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": "917896587312"})

    assert response.status_code == 422


def test_update_farmer_invalid_phone_alpha():
    """PUT /farmers/{id} with alphabetic characters should return 422."""
    farmer_id = uuid4()
    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": "abc"})

    assert response.status_code == 422


def test_update_farmer_invalid_phone_empty():
    """PUT /farmers/{id} with an empty string phone number should return 422."""
    farmer_id = uuid4()
    response = client.put(f"/farmers/{farmer_id}", json={"phone_number": ""})

    assert response.status_code == 422


# --- Phone number validation tests for POST /farmers ---

def test_create_farmer_invalid_phone_no_plus():
    """POST /farmers with '917896587312' (missing '+' prefix) should return 422."""
    response = client.post("/farmers", json={"phone_number": "917896587312"})
    assert response.status_code == 422


def test_create_farmer_invalid_phone_too_short():
    """POST /farmers with '+2015184' (too few digits) should return 422."""
    response = client.post("/farmers", json={"phone_number": "+2015184"})
    assert response.status_code == 422
