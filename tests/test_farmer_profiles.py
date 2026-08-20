import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from src.main import app
from src.farmer_profiles.schemas import FarmerProfileResponse
from src.farmer_profiles.service import FarmerProfileService
from src.farmer_profiles.dependencies import get_farmer_profile_service

client = TestClient(app)

@pytest.fixture
def mock_profile_service():
    service = AsyncMock(spec=FarmerProfileService)
    app.dependency_overrides[get_farmer_profile_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def test_create_farmer_profile(mock_profile_service):
    profile_id = uuid4()
    farmer_id = uuid4()
    mock_profile = FarmerProfileResponse(
        id=profile_id,
        farmer_id=farmer_id,
        full_name="Test Farmer",
        state="Telangana",
        district="Hyderabad",
        current_crop="Paddy",
        land_size_acres=2.5
    )
    mock_profile_service.create_profile.return_value = mock_profile

    response = client.post("/farmer-profiles", json={
        "farmer_id": str(farmer_id),
        "full_name": "Test Farmer",
        "state": "Telangana",
        "district": "Hyderabad",
        "current_crop": "Paddy",
        "land_size_acres": 2.5
    })
    
    assert response.status_code == 201
    assert response.json()["id"] == str(profile_id)
    assert response.json()["farmer_id"] == str(farmer_id)

def test_get_farmer_profile(mock_profile_service):
    profile_id = uuid4()
    farmer_id = uuid4()
    mock_profile = FarmerProfileResponse(
        id=profile_id,
        farmer_id=farmer_id,
        full_name="Test Farmer",
        state="Telangana",
        district="Hyderabad",
        current_crop="Paddy",
        land_size_acres=2.5
    )
    mock_profile_service.get_profile.return_value = mock_profile

    response = client.get(f"/farmer-profiles/{profile_id}")
    
    assert response.status_code == 200
    assert response.json()["id"] == str(profile_id)

def test_get_farmer_profiles(mock_profile_service):
    mock_profile_service.get_profiles.return_value = (0, [])
    response = client.get("/farmer-profiles?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []

def test_get_profile_by_farmer(mock_profile_service):
    profile_id = uuid4()
    farmer_id = uuid4()
    mock_profile = FarmerProfileResponse(
        id=profile_id,
        farmer_id=farmer_id,
        full_name="Test Farmer",
        state="Telangana",
        district="Hyderabad",
        current_crop="Paddy",
        land_size_acres=2.5
    )
    mock_profile_service.get_profile_by_farmer.return_value = mock_profile

    response = client.get(f"/farmer-profiles/farmer/{farmer_id}")
    
    assert response.status_code == 200
    assert response.json()["farmer_id"] == str(farmer_id)

def test_update_farmer_profile(mock_profile_service):
    profile_id = uuid4()
    farmer_id = uuid4()
    mock_profile = FarmerProfileResponse(
        id=profile_id,
        farmer_id=farmer_id,
        full_name="Test Farmer Updated",
        state="Telangana",
        district="Hyderabad",
        current_crop="Cotton",
        land_size_acres=5.0
    )
    mock_profile_service.update_profile.return_value = mock_profile

    response = client.put(f"/farmer-profiles/{profile_id}", json={
        "full_name": "Test Farmer Updated",
        "current_crop": "Cotton",
        "land_size_acres": 5.0
    })
    
    assert response.status_code == 200
    assert response.json()["full_name"] == "Test Farmer Updated"
    assert response.json()["current_crop"] == "Cotton"

def test_delete_farmer_profile(mock_profile_service):
    profile_id = uuid4()
    mock_profile_service.delete_profile.return_value = None

    response = client.delete(f"/farmer-profiles/{profile_id}")
    
    assert response.status_code == 204
