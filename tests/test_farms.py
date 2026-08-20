import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.farms.schemas import FarmResponse
from src.farms.service import FarmService
from src.farms.dependencies import get_farm_service

client = TestClient(app)
TS = "2023-01-01T00:00:00Z"

@pytest.fixture
def mock_farm_service():
    service = AsyncMock(spec=FarmService)
    app.dependency_overrides[get_farm_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def _resp(**kw) -> FarmResponse:
    defaults = dict(id=uuid4(), farmer_id=uuid4(), farm_name="North Plot",
        land_size_acres=5.0, soil_type="Black", irrigation_type="Drip",
        village="Rampur", district="Warangal", state="Telangana",
        latitude=17.9, longitude=79.5, created_at=TS, updated_at=TS)
    defaults.update(kw)
    return FarmResponse(**defaults)

# ---- CREATE ----

def test_create_farm(mock_farm_service):
    farm = _resp()
    mock_farm_service.create_farm.return_value = farm
    response = client.post("/farms", json={
        "farmer_id": str(farm.farmer_id), "farm_name": "North Plot",
        "land_size_acres": 5.0, "soil_type": "Black", "irrigation_type": "Drip",
        "village": "Rampur", "district": "Warangal", "state": "Telangana"})
    assert response.status_code == 201
    assert response.json()["farm_name"] == "North Plot"

def test_create_farm_invalid_land_size_zero():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "farm_name": "Bad Farm", "land_size_acres": 0})
    assert response.status_code == 422

def test_create_farm_invalid_land_size_negative():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "farm_name": "Bad Farm", "land_size_acres": -5})
    assert response.status_code == 422

def test_create_farm_invalid_soil_type():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "farm_name": "Test", "land_size_acres": 2.0,
        "soil_type": "Martian"})
    assert response.status_code == 422

def test_create_farm_invalid_irrigation_type():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "farm_name": "Test", "land_size_acres": 2.0,
        "irrigation_type": "Laser"})
    assert response.status_code == 422

def test_create_farm_invalid_latitude():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "farm_name": "Test", "land_size_acres": 2.0,
        "latitude": 91.0})
    assert response.status_code == 422

def test_create_farm_missing_farm_name():
    response = client.post("/farms", json={
        "farmer_id": str(uuid4()), "land_size_acres": 2.0})
    assert response.status_code == 422

# ---- READ ----

def test_get_farm(mock_farm_service):
    farm = _resp()
    mock_farm_service.get_farm.return_value = farm
    response = client.get(f"/farms/{farm.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(farm.id)

def test_get_farms(mock_farm_service):
    mock_farm_service.get_farms.return_value = (0, [])
    response = client.get("/farms?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 0

def test_get_farmer_farms(mock_farm_service):
    fid = uuid4()
    farm = _resp(farmer_id=fid)
    mock_farm_service.get_farmer_farms.return_value = (1, [farm])
    response = client.get(f"/farms/farmer/{fid}?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 1

# ---- UPDATE ----

def test_update_farm(mock_farm_service):
    farm = _resp(farm_name="South Plot")
    mock_farm_service.update_farm.return_value = farm
    response = client.put(f"/farms/{farm.id}", json={"farm_name": "South Plot"})
    assert response.status_code == 200
    assert response.json()["farm_name"] == "South Plot"

def test_update_farm_invalid_soil_type():
    response = client.put(f"/farms/{uuid4()}", json={"soil_type": "Plutonium"})
    assert response.status_code == 422

# ---- DELETE ----

def test_delete_farm(mock_farm_service):
    mock_farm_service.delete_farm.return_value = None
    response = client.delete(f"/farms/{uuid4()}")
    assert response.status_code == 204
