import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime
from src.main import app
from src.schemes.schemas import (
    GovernmentSchemeResponse,
    FarmerEligibilityResponse,
    SchemeEligibilityItem,
    SchemeApplicationResponse,
)
from src.schemes.service import SchemeService
from src.schemes.dependencies import get_scheme_service

client = TestClient(app)
TS = datetime.utcnow()


@pytest.fixture
def mock_scheme_service():
    service = AsyncMock(spec=SchemeService)
    app.dependency_overrides[get_scheme_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_scheme_response(**kwargs) -> GovernmentSchemeResponse:
    defaults = dict(
        id=uuid4(),
        scheme_name="PM-KISAN Samman Nidhi",
        scheme_code="PM_KISAN",
        category="Direct Income Support",
        state="All India",
        district=None,
        crop_type="All Crops",
        min_land_acres=0.0,
        max_land_acres=5.0,
        description="Minimum income support for small farmers.",
        benefits_summary="₹6,000 per year paid in 3 installments.",
        eligibility_criteria="Landholding farmers up to 5 acres.",
        required_documents="Aadhaar, Land Papers, Bank Passbook.",
        application_deadline=None,
        official_portal_url="https://pmkisan.gov.in",
        is_active=True,
        created_at=TS,
    )
    defaults.update(kwargs)
    return GovernmentSchemeResponse(**defaults)


def test_list_government_schemes(mock_scheme_service):
    """Test listing government schemes endpoint."""
    mock_scheme = _mock_scheme_response()
    mock_scheme_service.list_schemes.return_value = [mock_scheme]

    response = client.get("/schemes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["scheme_code"] == "PM_KISAN"


def test_evaluate_farmer_eligibility(mock_scheme_service):
    """Test AI Government Schemes eligibility assessment endpoint."""
    farmer_id = uuid4()
    mock_scheme = _mock_scheme_response()
    mock_eligibility = FarmerEligibilityResponse(
        farmer_id=farmer_id,
        farmer_name="Ramesh Gowda",
        state="Telangana",
        district="Jagtial",
        land_size_acres=2.5,
        total_schemes_evaluated=1,
        eligible_schemes_count=1,
        schemes=[
            SchemeEligibilityItem(
                scheme=mock_scheme,
                is_eligible=True,
                match_score_percentage=100,
                eligibility_reason="State and land size match criteria.",
                recommended_action="Apply via Meeseva portal.",
                voice_explanation="You are 100% eligible for PM-KISAN."
            )
        ]
    )
    mock_scheme_service.evaluate_farmer_eligibility.return_value = mock_eligibility

    response = client.get(f"/schemes/eligibility/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["farmer_id"] == str(farmer_id)
    assert data["eligible_schemes_count"] == 1
    assert data["schemes"][0]["is_eligible"] is True


def test_apply_for_scheme(mock_scheme_service):
    """Test applying for a government scheme endpoint."""
    farmer_id = uuid4()
    scheme_id = uuid4()
    mock_app_resp = SchemeApplicationResponse(
        id=uuid4(),
        farmer_id=farmer_id,
        scheme_id=scheme_id,
        status="Applied",
        notes="Applied via BhoomiMitra AI",
        scheme_name="PM-KISAN Samman Nidhi",
        created_at=TS,
        updated_at=TS,
    )
    mock_scheme_service.apply_for_scheme.return_value = mock_app_resp

    payload = {
        "farmer_id": str(farmer_id),
        "scheme_id": str(scheme_id),
        "notes": "Applied via BhoomiMitra AI"
    }
    response = client.post("/schemes/apply", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["farmer_id"] == str(farmer_id)
    assert data["status"] == "Applied"


def test_get_farmer_applications(mock_scheme_service):
    """Test retrieving farmer scheme applications endpoint."""
    farmer_id = uuid4()
    mock_app_resp = SchemeApplicationResponse(
        id=uuid4(),
        farmer_id=farmer_id,
        scheme_id=uuid4(),
        status="Applied",
        notes="Testing",
        scheme_name="PM-KISAN Samman Nidhi",
        created_at=TS,
        updated_at=TS,
    )
    mock_scheme_service.get_farmer_applications.return_value = [mock_app_resp]

    response = client.get(f"/schemes/farmer-applications/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["status"] == "Applied"
