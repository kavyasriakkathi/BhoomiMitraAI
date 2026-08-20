import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.crop_health.schemas import CropHealthResponse
from src.crop_health.service import CropHealthService
from src.crop_health.dependencies import get_crop_health_service

client = TestClient(app)
TS = "2023-01-01T00:00:00Z"

@pytest.fixture
def mock_crop_health_service():
    service = AsyncMock(spec=CropHealthService)
    app.dependency_overrides[get_crop_health_service] = lambda: service
    yield service
    app.dependency_overrides.clear()

def _resp(**kw) -> CropHealthResponse:
    defaults = dict(id=uuid4(), crop_id=uuid4(), farmer_id=uuid4(), image_url=None,
        symptoms="Yellowing leaves", disease_name="Nitrogen Deficiency",
        diagnosis_result="Nitrogen deficiency detected.",
        treatment_recommendation="Apply nitrogen rich fertilizer.",
        confidence_score=0.9, created_at=TS, updated_at=TS)
    defaults.update(kw)
    return CropHealthResponse(**defaults)

# ---- CREATE ----

def test_create_diagnosis(mock_crop_health_service):
    diagnosis = _resp()
    mock_crop_health_service.create_diagnosis.return_value = diagnosis
    response = client.post("/crop-health", json={
        "crop_id": str(diagnosis.crop_id), "farmer_id": str(diagnosis.farmer_id),
        "symptoms": "Yellowing leaves", "diagnosis_result": "Nitrogen deficiency detected.",
        "treatment_recommendation": "Apply nitrogen rich fertilizer."})
    assert response.status_code == 201
    assert response.json()["symptoms"] == "Yellowing leaves"

def test_create_diagnosis_empty_symptoms():
    response = client.post("/crop-health", json={
        "crop_id": str(uuid4()), "farmer_id": str(uuid4()),
        "symptoms": "  ", "diagnosis_result": "Nitrogen deficiency detected.",
        "treatment_recommendation": "Apply nitrogen rich fertilizer."})
    assert response.status_code == 422

# ---- READ ----

def test_get_diagnosis(mock_crop_health_service):
    diagnosis = _resp()
    mock_crop_health_service.get_diagnosis.return_value = diagnosis
    response = client.get(f"/crop-health/{diagnosis.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(diagnosis.id)

def test_get_diagnoses(mock_crop_health_service):
    mock_crop_health_service.get_diagnoses.return_value = (0, [])
    response = client.get("/crop-health?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 0

def test_get_crop_diagnoses(mock_crop_health_service):
    cid = uuid4()
    diagnosis = _resp(crop_id=cid)
    mock_crop_health_service.get_crop_diagnoses.return_value = (1, [diagnosis])
    response = client.get(f"/crop-health/crop/{cid}?page=1&size=10")
    assert response.status_code == 200
    assert response.json()["total"] == 1

# ---- UPDATE ----

def test_update_diagnosis(mock_crop_health_service):
    diagnosis = _resp(symptoms="Brown spots")
    mock_crop_health_service.update_diagnosis.return_value = diagnosis
    response = client.put(f"/crop-health/{diagnosis.id}", json={"symptoms": "Brown spots"})
    assert response.status_code == 200
    assert response.json()["symptoms"] == "Brown spots"

def test_update_diagnosis_invalid_confidence():
    response = client.put(f"/crop-health/{uuid4()}", json={"confidence_score": 1.5})
    assert response.status_code == 422

# ---- DELETE ----

def test_delete_diagnosis(mock_crop_health_service):
    mock_crop_health_service.delete_diagnosis.return_value = None
    response = client.delete(f"/crop-health/{uuid4()}")
    assert response.status_code == 204
