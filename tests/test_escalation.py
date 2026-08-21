import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime
from fastapi.testclient import TestClient
from src.main import app
from src.core.models import Farmer, Expert
from src.escalation.service import (
    _detect_escalation_intent,
    _detect_specialty_hint,
    _generate_ticket_id,
    _find_recent_pending_ticket,
    enrich_response_with_escalation,
    EscalationService,
)
from src.escalation.repository import EscalationRepository
from src.escalation.schemas import ExpertResponse, FarmerEscalationHistoryResponse
from src.escalation.dependencies import get_escalation_service

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Intent Detection Tests (English & Telugu)
# ---------------------------------------------------------------------------

def test_escalation_intent_english():
    """English escalation phrases trigger intent detection."""
    assert _detect_escalation_intent("i want to talk to an expert", "i want to talk to an expert") is True
    assert _detect_escalation_intent("connect me to an agriculture officer", "connect me to an agriculture officer") is True
    assert _detect_escalation_intent("can an agronomist call me back?", "can an agronomist call me back?") is True
    assert _detect_escalation_intent("i need human agent support", "i need human agent support") is True
    assert _detect_escalation_intent("please escalate this issue", "please escalate this issue") is True


def test_escalation_intent_telugu():
    """Telugu escalation phrases trigger intent detection."""
    assert _detect_escalation_intent("", "వ్యవసాయ అధికారితో మాట్లాడాలి") is True
    assert _detect_escalation_intent("", "నిపుణుడి సహాయం కావాలి") is True
    assert _detect_escalation_intent("", "అధికారిని సంప్రదించాలి") is True
    assert _detect_escalation_intent("", "ఏఈవో గారికి కాల్ చేయండి") is True


def test_non_escalation_query_skipped():
    """General agronomic or weather questions do not trigger escalation."""
    assert _detect_escalation_intent("how much urea should i use for cotton?", "how much urea should i use for cotton?") is False
    assert _detect_escalation_intent("", "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?") is False
    assert _detect_escalation_intent("what is the weather tomorrow in warangal?", "what is the weather tomorrow in warangal?") is False


# ---------------------------------------------------------------------------
# 2. Specialty & Helper Tests
# ---------------------------------------------------------------------------

def test_specialty_hint_detection():
    """Specialty hint correctly detected from query terms."""
    assert _detect_specialty_hint("pest insect damage on leaves") == "Pest"
    assert _detect_specialty_hint("పురుగు నివారణ ఎలా?") == "Pest"
    assert _detect_specialty_hint("soil fertilizer dosage") == "Soil"
    assert _detect_specialty_hint("cotton boll rot") == "Cotton"
    assert _detect_specialty_hint("drip irrigation schedule") == "Irrigation"
    assert _detect_specialty_hint("general question") is None


def test_ticket_id_generation():
    """Ticket ID generated in expected clean format."""
    ticket_id = _generate_ticket_id()
    assert ticket_id.startswith("ESC-")
    parts = ticket_id.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8  # YYYYMMDD


def test_find_recent_pending_ticket():
    """Detects active pending tickets created today."""
    today = datetime.utcnow().strftime("%Y%m%d")
    history = [
        {"ticket_id": f"ESC-{today}-1234", "status": "Assigned", "expert_name": "Dr. Rao"},
    ]
    found = _find_recent_pending_ticket(history)
    assert found is not None
    assert found["ticket_id"] == f"ESC-{today}-1234"

    # Resolved tickets are ignored
    resolved_history = [
        {"ticket_id": f"ESC-{today}-1234", "status": "Resolved", "expert_name": "Dr. Rao"},
    ]
    assert _find_recent_pending_ticket(resolved_history) is None


# ---------------------------------------------------------------------------
# 3. Pipeline Enrichment Tests (English & Telugu)
# ---------------------------------------------------------------------------

def _make_mock_expert(name="Dr. K. Srinivas Rao", specialty="Pest Control", phone="+91 9848012345"):
    expert = MagicMock()
    expert.id = uuid4()
    expert.name = name
    expert.specialty = specialty
    expert.phone_number = phone
    expert.is_active = True
    return expert


@pytest.mark.asyncio
async def test_enrich_escalation_assigned_expert_english():
    """Escalation enrichment creates a ticket and assigns an active specialist in English."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    mock_expert = _make_mock_expert()
    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[mock_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        res = await enrich_response_with_escalation(
            db, "I want to talk to an agriculture officer", "Base agronomic response.", farmer
        )

    assert "Base agronomic response." in res
    assert "Krishi Officer Escalation Ticket" in res
    assert "Assigned to District Agriculture Officer" in res
    assert mock_expert.name in res
    assert mock_expert.phone_number in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_enrich_escalation_assigned_expert_telugu():
    """Escalation enrichment responds with authentic Telugu labels when farmer language is Telugu."""
    farmer = MagicMock(id=uuid4(), preferred_language="te")
    db = AsyncMock()

    mock_expert = _make_mock_expert(name="డాక్టర్ కె. శ్రీనివాస్ రావు", specialty="పంట రక్షణ నిపుణులు")
    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[mock_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        res = await enrich_response_with_escalation(
            db, "వ్యవసాయ అధికారితో మాట్లాడాలి", "వ్యవసాయ సలహా.", farmer
        )

    assert "వ్యవసాయ సలహా." in res
    assert "వ్యవసాయ అధికారి సంప్రదింపు టికెట్" in res
    assert "జిల్లా వ్యవసాయ అధికారికి కేటాయించబడింది" in res
    assert "డాక్టర్ కె. శ్రీనివాస్ రావు" in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_enrich_escalation_no_expert_fallback():
    """When no active expert is registered, the pipeline falls back gracefully to Kisan Call Centre."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        res = await enrich_response_with_escalation(
            db, "Connect me to an expert", "Base advice.", farmer
        )

    assert "Base advice." in res
    assert "No local officer is currently on duty" in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_enrich_escalation_duplicate_prevention():
    """Duplicate escalation queries on the same day acknowledge the existing active ticket."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()
    today = datetime.utcnow().strftime("%Y%m%d")
    existing_ticket = {
        "ticket_id": f"ESC-{today}-5555",
        "status": "Assigned",
        "expert_name": "Dr. K. Srinivas Rao",
        "expert_phone": "+91 9848012345",
    }

    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[existing_ticket]):

        res = await enrich_response_with_escalation(
            db, "Where is my officer callback?", "Base response.", farmer
        )

    assert "ESC-" in res
    assert f"ESC-{today}-5555" in res
    assert "already have an active escalation ticket" in res


@pytest.mark.asyncio
async def test_enrich_escalation_db_failure_returns_original():
    """Database exception does not break the pipeline and returns original response safely."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("Database connection error"))
    original = "Important crop diagnosis advice."

    res = await enrich_response_with_escalation(db, "Connect me to an expert", original, farmer)
    assert res == original


# ---------------------------------------------------------------------------
# 4. REST Endpoints Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_escalation_service():
    service = AsyncMock(spec=EscalationService)
    app.dependency_overrides[get_escalation_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_list_experts_endpoint(mock_escalation_service):
    exp = ExpertResponse(
        id=uuid4(),
        name="Dr. K. Srinivas Rao",
        phone_number="+91 9848012345",
        specialty="Pest Control",
        is_active=True,
    )
    mock_escalation_service.list_experts.return_value = [exp]

    response = client.get("/escalation/experts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Dr. K. Srinivas Rao"


def test_create_expert_endpoint(mock_escalation_service):
    exp = ExpertResponse(
        id=uuid4(),
        name="Dr. Ananya Sharma",
        phone_number="+91 9876543211",
        specialty="Soil Health",
        is_active=True,
    )
    mock_escalation_service.create_expert.return_value = exp

    payload = {
        "name": "Dr. Ananya Sharma",
        "phone_number": "+91 9876543211",
        "specialty": "Soil Health",
        "is_active": True,
    }
    response = client.post("/escalation/experts", json=payload)
    assert response.status_code == 201
    assert response.json()["name"] == "Dr. Ananya Sharma"


def test_get_farmer_tickets_endpoint(mock_escalation_service):
    farmer_id = uuid4()
    history_resp = FarmerEscalationHistoryResponse(
        farmer_id=farmer_id,
        total_tickets=1,
        tickets=[{"ticket_id": "ESC-20260820-1111", "status": "Assigned"}],
    )
    mock_escalation_service.get_farmer_escalations.return_value = history_resp

    response = client.get(f"/escalation/tickets/farmer/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tickets"] == 1
    assert data["tickets"][0]["ticket_id"] == "ESC-20260820-1111"
