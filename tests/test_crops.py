import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.crops.schemas import CropResponse
from src.crops.service import CropService
from src.crops.dependencies import get_crop_service

client = TestClient(app)
TS = "2023-01-01T00:00:00Z"

@pytest.fixture
def mock_crop_service():
    service = AsyncMock(spec=CropService)
    app.dependency_overrides[get_crop_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def _resp(**kw) -> CropResponse:
    defaults = dict(id=uuid4(), farm_id=uuid4(), crop_name="Wheat",
        variety="Sharbati", sowing_date=None, season="Rabi", status="planned",
        created_at=TS, updated_at=TS)
    defaults.update(kw)
    return CropResponse(**defaults)

# ---- CREATE ----

def test_create_crop(mock_crop_service):
    crop = _resp()
    mock_crop_service.create_crop.return_value = crop
    response = client.post("/crops", json={
        "farm_id": str(crop.farm_id), "crop_name": "Wheat",
        "variety": "Sharbati", "season": "Rabi", "status": "planned"})
    assert response.status_code == 201
    assert response.json()["crop_name"] == "Wheat"

def test_create_crop_invalid_season():
    response = client.post("/crops", json={
        "farm_id": str(uuid4()), "crop_name": "Wheat", "season": "Winter"})
    assert response.status_code == 422

def test_create_crop_invalid_status():
    response = client.post("/crops", json={
        "farm_id": str(uuid4()), "crop_name": "Wheat", "status": "dead"})
    assert response.status_code == 422

def test_create_crop_missing_crop_name():
    response = client.post("/crops", json={
        "farm_id": str(uuid4()), "season": "Rabi"})
    assert response.status_code == 422

# ---- READ ----

def test_get_crop(mock_crop_service):
    crop = _resp()
    mock_crop_service.get_crop.return_value = crop
    response = client.get(f"/crops/{crop.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(crop.id)

def test_get_crops(mock_crop_service):
    mock_crop_service.get_crops.return_value = (0, [])
    response = client.get("/crops?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 0

def test_get_farm_crops(mock_crop_service):
    fid = uuid4()
    crop = _resp(farm_id=fid)
    mock_crop_service.get_farm_crops.return_value = (1, [crop])
    response = client.get(f"/crops/farm/{fid}?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 1

# ---- UPDATE ----

def test_update_crop(mock_crop_service):
    crop = _resp(crop_name="Rice")
    mock_crop_service.update_crop.return_value = crop
    response = client.put(f"/crops/{crop.id}", json={"crop_name": "Rice"})
    assert response.status_code == 200
    assert response.json()["crop_name"] == "Rice"

def test_update_crop_invalid_season():
    response = client.put(f"/crops/{uuid4()}", json={"season": "Spring"})
    assert response.status_code == 422

# ---- DELETE ----

def test_delete_crop(mock_crop_service):
    mock_crop_service.delete_crop.return_value = None
    response = client.delete(f"/crops/{uuid4()}")
    assert response.status_code == 204
