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


# ===========================================================================
# NEW TESTS: Intent Detection, Formatting, Crop Prioritisation, Pipeline
# ===========================================================================

# ---------------------------------------------------------------------------
# 5. Intent Detection — English keywords
# ---------------------------------------------------------------------------

def test_scheme_intent_english():
    """English scheme keywords correctly trigger intent."""
    from src.schemes.service import _detect_scheme_intent

    assert _detect_scheme_intent("what government schemes are available?") is True
    assert _detect_scheme_intent("tell me about pm kisan") is True
    assert _detect_scheme_intent("i want to know about subsidies for farmers") is True
    assert _detect_scheme_intent("am i eligible for kcc?") is True


# ---------------------------------------------------------------------------
# 6. Intent Detection — Telugu keywords
# ---------------------------------------------------------------------------

def test_scheme_intent_telugu():
    """Telugu scheme keywords correctly trigger intent."""
    from src.schemes.service import _SCHEME_KEYWORDS_TE

    query_te = "రైతులకు ఏ ప్రభుత్వ పథకాలు ఉన్నాయి?"
    has_intent = any(kw in query_te for kw in _SCHEME_KEYWORDS_TE)
    assert has_intent, "Telugu scheme query should detect scheme intent"

    query_subsidy = "సబ్సిడీలు అందుబాటులో ఉన్నాయా?"
    has_intent_subsidy = any(kw in query_subsidy for kw in _SCHEME_KEYWORDS_TE)
    assert has_intent_subsidy, "Telugu subsidy query should detect scheme intent"


# ---------------------------------------------------------------------------
# 7. Non-scheme query does NOT trigger intent
# ---------------------------------------------------------------------------

def test_no_scheme_intent_skipped():
    """Pest or crop disease query does not trigger scheme intent."""
    from src.schemes.service import _detect_scheme_intent, _SCHEME_KEYWORDS_TE

    query = "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?"
    en_match = _detect_scheme_intent(query.lower())
    te_match = any(kw in query for kw in _SCHEME_KEYWORDS_TE)
    assert not en_match and not te_match, "Pest query should not trigger scheme intent"


# ---------------------------------------------------------------------------
# 8. Crop detection from query
# ---------------------------------------------------------------------------

def test_crop_detection_from_query():
    """Crop keywords extracted correctly from English and Telugu queries."""
    from src.schemes.service import _detect_crop_from_query

    assert _detect_crop_from_query("cotton crop scheme") == "Cotton"
    assert _detect_crop_from_query("పత్తి పంటకు పథకం") == "Cotton"
    assert _detect_crop_from_query("rice subsidy available?") == "Rice"
    assert _detect_crop_from_query("వరి రైతులకు సహాయం") == "Rice"
    assert _detect_crop_from_query("what schemes are available") is None


# ---------------------------------------------------------------------------
# 9. Crop prioritisation — never excludes "All Crops" schemes
# ---------------------------------------------------------------------------

def test_crop_sort_never_excludes_all_crops():
    """'All Crops' schemes always appear in results even when a crop is mentioned."""
    from src.schemes.service import _sort_schemes_by_crop_priority
    from unittest.mock import MagicMock

    all_crops_scheme = MagicMock()
    all_crops_scheme.crop_type = "All Crops"
    all_crops_scheme.scheme_name = "PM-KISAN"

    cotton_scheme = MagicMock()
    cotton_scheme.crop_type = "Cotton"
    cotton_scheme.scheme_name = "Cotton Support"

    sorted_schemes = _sort_schemes_by_crop_priority(
        [all_crops_scheme, cotton_scheme], mentioned_crop="Cotton"
    )

    names = [s.scheme_name for s in sorted_schemes]
    assert "PM-KISAN" in names, "'All Crops' scheme must never be excluded"
    assert names[0] == "Cotton Support", "Crop-specific match should rank first"


# ---------------------------------------------------------------------------
# 10. English scheme block formatting includes disclaimer
# ---------------------------------------------------------------------------

def test_scheme_formatting_english_has_disclaimer():
    """English formatted scheme block includes the safety disclaimer."""
    from src.schemes.service import _format_scheme_block, _EN_LABELS
    from unittest.mock import MagicMock
    from datetime import datetime, timedelta

    mock_scheme = MagicMock()
    mock_scheme.scheme_name = "PM-KISAN Samman Nidhi"
    mock_scheme.benefits_summary = "₹6,000 per year in 3 installments."
    mock_scheme.eligibility_criteria = "Farmers with up to 5 acres."
    mock_scheme.required_documents = "Aadhaar, Land Papers, Bank Passbook."
    mock_scheme.application_deadline = datetime.utcnow() + timedelta(days=60)
    mock_scheme.official_portal_url = "https://pmkisan.gov.in"

    block = _format_scheme_block(mock_scheme, _EN_LABELS, "en")
    assert "PM-KISAN Samman Nidhi" in block
    assert "₹6,000" in block
    assert "pmkisan.gov.in" in block
    # Disclaimer is appended at the full_block level, not inside scheme block


# ---------------------------------------------------------------------------
# 11. Telugu scheme block formatting uses Telugu labels
# ---------------------------------------------------------------------------

def test_scheme_formatting_telugu_labels():
    """Telugu formatted scheme block uses Telugu label keys."""
    from src.schemes.service import _format_scheme_block, _TE_LABELS
    from unittest.mock import MagicMock

    mock_scheme = MagicMock()
    mock_scheme.scheme_name = "రైతు బంధు"
    mock_scheme.benefits_summary = "ఎకరాకు ₹10,000 ప్రతి సంవత్సరం."
    mock_scheme.eligibility_criteria = "తెలంగాణ రాష్ట్రంలోని పట్టాదార్ రైతులు."
    mock_scheme.required_documents = "పట్టాదార్ పాస్ బుక్, ఆధార్ కార్డు."
    mock_scheme.application_deadline = None
    mock_scheme.official_portal_url = "https://rythubandhu.telangana.gov.in"

    block = _format_scheme_block(mock_scheme, _TE_LABELS, "te")
    assert "రైతు బంధు" in block
    assert "ప్రయోజనాలు" in block
    assert "అర్హత" in block
    assert "rythubandhu.telangana.gov.in" in block


# ---------------------------------------------------------------------------
# 12. seed_defaults_if_empty bug fix — self parameter present
# ---------------------------------------------------------------------------

def test_seed_defaults_if_empty_has_self_parameter():
    """Verify the seed_defaults_if_empty method correctly declares self."""
    import inspect
    from src.schemes.service import SchemeService

    sig = inspect.signature(SchemeService.seed_defaults_if_empty)
    params = list(sig.parameters.keys())
    assert "self" in params, (
        "seed_defaults_if_empty must declare 'self' as first parameter. "
        "The bug has not been fixed."
    )


# ---------------------------------------------------------------------------
# 13. Expired deadline formatting in Telugu and English
# ---------------------------------------------------------------------------

def test_scheme_expired_deadline_formatting_telugu_and_english():
    """Expired scheme deadlines show closed status with next cycle notice."""
    from src.schemes.service import _format_scheme_block, _TE_LABELS, _EN_LABELS
    from unittest.mock import MagicMock
    from datetime import datetime, timedelta

    past_date = datetime.utcnow() - timedelta(days=30)
    mock_scheme = MagicMock()
    mock_scheme.scheme_name = "PMFBY Kharif 2024"
    mock_scheme.benefits_summary = "Crop loss insurance coverage."
    mock_scheme.eligibility_criteria = "Notified crop farmers."
    mock_scheme.required_documents = "Aadhaar, Sowing Certificate."
    mock_scheme.application_deadline = past_date
    mock_scheme.official_portal_url = "https://pmfby.gov.in"

    block_te = _format_scheme_block(mock_scheme, _TE_LABELS, "te")
    assert "గడువు ముగిసింది" in block_te
    assert "PMFBY Kharif 2024" in block_te

    block_en = _format_scheme_block(mock_scheme, _EN_LABELS, "en")
    assert "Deadline Closed" in block_en
    assert "pmfby.gov.in" in block_en


# ---------------------------------------------------------------------------
# 14. Future deadline formatting
# ---------------------------------------------------------------------------

def test_scheme_future_deadline_formatting():
    """Future scheme deadlines show clean formatted date without expired warning."""
    from src.schemes.service import _format_scheme_block, _TE_LABELS
    from unittest.mock import MagicMock
    from datetime import datetime, timedelta

    future_date = datetime.utcnow() + timedelta(days=90)
    mock_scheme = MagicMock()
    mock_scheme.scheme_name = "PM-KUSUM"
    mock_scheme.benefits_summary = "Solar pump subsidy up to 90%."
    mock_scheme.eligibility_criteria = "Borewell owners."
    mock_scheme.required_documents = "Aadhaar, Land Records, NOC."
    mock_scheme.application_deadline = future_date
    mock_scheme.official_portal_url = "https://pmkusum.mnre.gov.in"

    block = _format_scheme_block(mock_scheme, _TE_LABELS, "te")
    assert "PM-KUSUM" in block
    assert "గడువు ముగిసింది" not in block
    assert "pmkusum.mnre.gov.in" in block


# ---------------------------------------------------------------------------
# 15. Extended scheme intent keywords
# ---------------------------------------------------------------------------

def test_extended_scheme_intent_keywords():
    """Verify newly added Telugu and English agricultural scheme keywords."""
    from src.schemes.service import _detect_scheme_intent, _SCHEME_KEYWORDS_TE

    # Telugu keywords
    assert any(kw in "రైతు భరోసా నిధులు ఎప్పుడు వస్తాయి?" for kw in _SCHEME_KEYWORDS_TE)
    assert any(kw in "పీఎం కిసాన్ ఈకేవైసీ ఎలా చేయాలి?" for kw in _SCHEME_KEYWORDS_TE)
    assert any(kw in "వ్యవసాయ రుణం సబ్సిడీ పథకాలు" for kw in _SCHEME_KEYWORDS_TE)

    # English keywords
    assert _detect_scheme_intent("is there any financial assistance for drip irrigation?")
    assert _detect_scheme_intent("tell me about rythu bharosa")
    assert _detect_scheme_intent("what krishi yojana benefits can i get?")


# ---------------------------------------------------------------------------
# 16. Scheme structured fields verification
# ---------------------------------------------------------------------------

def test_scheme_structured_fields_verification():
    """Verify that all default seeded schemes have non-empty required fields and valid URLs."""
    from src.schemes.repository import DEFAULT_INDIAN_SCHEMES

    for s in DEFAULT_INDIAN_SCHEMES:
        assert s["scheme_name"] and len(s["scheme_name"]) > 3
        assert s["benefits_summary"] and len(s["benefits_summary"]) > 5
        assert s["eligibility_criteria"] and len(s["eligibility_criteria"]) > 5
        assert s["required_documents"] and len(s["required_documents"]) > 5
        assert s["official_portal_url"].startswith("http")
