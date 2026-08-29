import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime
from fastapi.testclient import TestClient
from src.main import app
from src.core.models import Farmer, Expert, UserAccount
from src.escalation.service import (
    _detect_escalation_intent,
    _detect_specialty_hint,
    _generate_ticket_id,
    _find_recent_pending_ticket,
    enrich_response_with_escalation,
    EscalationService,
)
from src.escalation.repository import EscalationRepository
from src.escalation.schemas import (
    ExpertResponse,
    FarmerEscalationHistoryResponse,
    TicketQueueResponse,
    TicketQueueItem,
)
from src.escalation.dependencies import get_escalation_service
from src.auth.dependencies import require_admin, require_expert, get_current_user

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Conservative Intent & Hazard Detection Tests (English & Telugu)
# ---------------------------------------------------------------------------

def test_escalation_intent_english_explicit():
    """English explicit escalation phrases trigger intent detection."""
    has_esc, reason = _detect_escalation_intent("i want to talk to an expert", "i want to talk to an expert")
    assert has_esc is True
    assert reason == "explicit"

    has_esc, reason = _detect_escalation_intent("connect me to an agriculture officer", "connect me to an agriculture officer")
    assert has_esc is True
    assert reason == "explicit"

    has_esc, reason = _detect_escalation_intent("can an agronomist call me back?", "can an agronomist call me back?")
    assert has_esc is True
    assert reason == "explicit"


def test_escalation_intent_telugu_explicit():
    """Telugu explicit escalation phrases trigger intent detection."""
    has_esc, reason = _detect_escalation_intent("", "వ్యవసాయ అధికారితో మాట్లాడాలి")
    assert has_esc is True
    assert reason == "explicit"

    has_esc, reason = _detect_escalation_intent("", "నిపుణుడి సహాయం కావాలి")
    assert has_esc is True
    assert reason == "explicit"

    has_esc, reason = _detect_escalation_intent("", "ఏఈవో గారికి కాల్ చేయండి")
    assert has_esc is True
    assert reason == "explicit"


def test_escalation_intent_hazardous_chemicals():
    """Hazardous and banned chemical queries trigger immediate safety escalation."""
    has_esc, reason = _detect_escalation_intent("is paraquat or monocrotophos safe?", "is paraquat or monocrotophos safe?")
    assert has_esc is True
    assert reason == "hazard"

    has_esc, reason = _detect_escalation_intent("swallowed pesticide poison emergency", "swallowed pesticide poison emergency")
    assert has_esc is True
    assert reason == "hazard"

    has_esc, reason = _detect_escalation_intent("", "పురుగుల మందు తాగడం జరిగింది విషం")
    assert has_esc is True
    assert reason == "hazard"


def test_escalation_intent_physical_inspection():
    """Physical farm visit and widespread catastrophic crop collapse trigger inspection escalation."""
    has_esc, reason = _detect_escalation_intent("need field inspection for crop dying completely", "need field inspection for crop dying completely")
    assert has_esc is True
    assert reason == "inspection"

    has_esc, reason = _detect_escalation_intent("", "తోట పరిశీలన కోసం అధికారి రావాలి మొక్కలు అన్నీ ఎండిపోతున్నాయి")
    assert has_esc is True
    assert reason == "inspection"


def test_non_escalation_query_skipped():
    """General agronomic questions and standard cautionary phrasing do not trigger escalation."""
    has_esc, reason = _detect_escalation_intent("how much urea should i use for cotton?", "how much urea should i use for cotton?")
    assert has_esc is False
    assert reason is None

    has_esc, reason = _detect_escalation_intent("", "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?")
    assert has_esc is False
    assert reason is None

    has_esc, reason = _detect_escalation_intent("what is the weather tomorrow in warangal?", "what is the weather tomorrow in warangal?")
    assert has_esc is False
    assert reason is None


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
async def test_enrich_escalation_hazard_warning():
    """Hazardous chemical triggers attach prominent safety warning and ticket."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    mock_expert = _make_mock_expert()
    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[mock_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        res = await enrich_response_with_escalation(
            db, "Can I spray Paraquat directly on cotton?", "Chemical advisory.", farmer
        )

    assert "URGENT SAFETY CAUTION" in res
    assert "Krishi Officer Escalation Ticket" in res
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
# 4. REST Endpoints Tests & RBAC
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_escalation_service():
    service = AsyncMock(spec=EscalationService)
    app.dependency_overrides[get_escalation_service] = lambda: service
    mock_admin = UserAccount(id=uuid4(), email="admin@bhoomimitra.in", role="admin", is_active=True)
    app.dependency_overrides[require_admin] = lambda: mock_admin
    app.dependency_overrides[require_expert] = lambda: mock_admin
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


def test_list_tickets_queue_endpoint(mock_escalation_service):
    queue_resp = TicketQueueResponse(
        total=2,
        pending=1,
        assigned=1,
        resolved=0,
        items=[
            TicketQueueItem(
                ticket_id="ESC-20260824-1001",
                farmer_name="Ramesh",
                farmer_phone="+91 9848000001",
                status="Assigned",
                topic="Cotton bollworm infestation",
                crop="Cotton",
                language="te",
                created_at="2026-08-24T10:00:00",
            ),
            TicketQueueItem(
                ticket_id="ESC-20260824-1002",
                farmer_name="Suresh",
                farmer_phone="+91 9848000002",
                status="Pending",
                topic="Chilli leaf curl virus",
                crop="Chilli",
                language="te",
                created_at="2026-08-24T11:00:00",
            ),
        ],
    )
    mock_escalation_service.list_tickets.return_value = queue_resp

    response = client.get("/escalation/tickets")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["pending"] == 1
    assert len(data["items"]) == 2
    assert data["items"][0]["ticket_id"] == "ESC-20260824-1001"


def test_update_ticket_status_endpoint(mock_escalation_service):
    updated_item = TicketQueueItem(
        ticket_id="ESC-20260824-1001",
        farmer_name="Ramesh",
        status="Resolved",
        topic="Cotton bollworm infestation",
        crop="Cotton",
        language="te",
        created_at="2026-08-24T10:00:00",
        notes="Prescribed Emamectin Benzoate 5% SG @ 4g/10L.",
    )
    mock_escalation_service.update_ticket_status.return_value = updated_item

    payload = {
        "status": "Resolved",
        "notes": "Prescribed Emamectin Benzoate 5% SG @ 4g/10L.",
    }
    response = client.patch("/escalation/tickets/ESC-20260824-1001/status", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Resolved"
    assert "Emamectin Benzoate" in data["notes"]


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


# ---------------------------------------------------------------------------
# Pilot Readiness Regression Tests (Task B: Dummy Expert Contact Safety)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_production_demo_expert_only_no_demo_phone_shown():
    """Regression Test 1: In production, demo expert dummy numbers are NEVER leaked to farmers."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    demo_expert = _make_mock_expert(name="Dr. K. Srinivas Rao", phone="+91 9848012345")

    with patch("src.config.get_settings") as mock_settings, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[demo_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        mock_settings.return_value.app_env = "production"

        res = await enrich_response_with_escalation(
            db, "I need to talk to an agricultural officer immediately", "Base advice.", farmer
        )

    # Demo expert contact and name must NEVER appear in farmer response in production
    assert "+91 9848012345" not in res
    assert "Dr. K. Srinivas Rao" not in res
    # Ticket is still logged, and official national Kisan Call Centre is provided
    assert "ESC-" in res
    assert "1800-180-1551" in res
    assert "No local officer is currently on duty" in res


@pytest.mark.asyncio
async def test_production_verified_active_expert_shown():
    """Regression Test 2: In production, a real verified pilot expert contact is properly displayed."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    # Real pilot expert with authentic non-demo phone number
    real_expert = _make_mock_expert(name="Dr. M. Suresh Kumar", specialty="Pest Control", phone="+91 9440123456")

    with patch("src.config.get_settings") as mock_settings, \
         patch("src.escalation.notifications.notify_expert_escalation_ticket", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[real_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        mock_settings.return_value.app_env = "production"

        res = await enrich_response_with_escalation(
            db, "Connect me to an officer", "Base advice.", farmer
        )

    assert "Dr. M. Suresh Kumar" in res
    assert "+91 9440123456" in res
    assert "Assigned to District Agriculture Officer" in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_no_active_expert_kisan_call_centre_fallback_regression():
    """Regression Test 3: When no expert is available at all, Kisan Call Centre is safely provided."""
    farmer = MagicMock(id=uuid4(), preferred_language="te")
    db = AsyncMock()

    with patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        res = await enrich_response_with_escalation(
            db, "వ్యవసాయ అధికారితో మాట్లాడాలి", "సలహా.", farmer
        )

    assert "ప్రస్తుతం స్థానిక అధికారి అందుబాటులో లేరు" in res
    assert "1800-180-1551" in res
    assert "ESC-" in res


@pytest.mark.asyncio
async def test_local_development_expert_behavior_regression():
    """Regression Test 4: In development/test mode, mock active experts render properly for testing."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()

    mock_expert = _make_mock_expert(name="Test Expert", phone="+91 9999988888")

    with patch("src.config.get_settings") as mock_settings, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[mock_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        mock_settings.return_value.app_env = "development"

        res = await enrich_response_with_escalation(
            db, "I need an expert", "Base response.", farmer
        )

    assert "Test Expert" in res
    assert "+91 9999988888" in res


# ---------------------------------------------------------------------------
# Pilot Readiness Tests (Expert Alert Dispatch & Privacy Safeguards)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verified_expert_receives_alert_for_new_ticket():
    """Verified active expert receives exactly one WhatsApp notification for a new escalation ticket."""
    farmer = MagicMock(id=uuid4(), phone_number="+91 9876543210", preferred_language="en")
    db = AsyncMock()

    real_expert = _make_mock_expert(name="Dr. M. Suresh Kumar", specialty="Pest Control", phone="+91 9440123456")

    mock_settings_obj = MagicMock()
    mock_settings_obj.app_env = "production"
    mock_settings_obj.expert_whatsapp_group_id = ""

    with patch("src.config.get_settings", return_value=mock_settings_obj), \
         patch("src.escalation.notifications.get_settings", return_value=mock_settings_obj), \
         patch("src.escalation.notifications.send_text_message", new_callable=AsyncMock) as mock_send_wa, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[real_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        mock_send_wa.return_value = "wamid.HBgLMTIzNDU2Nzg5MA=="

        res = await enrich_response_with_escalation(
            db, "I need to talk to an agriculture officer for cotton pest attack", "Advisory note.", farmer
        )

    assert "ESC-" in res
    # Exactly one direct notification sent to the verified expert
    assert mock_send_wa.call_count == 1
    call_args = mock_send_wa.call_args
    assert call_args.kwargs["to_phone"] == "919440123456"
    assert "BhoomiMitra Expert Escalation Alert" in call_args.kwargs["message_text"]
    assert "9876543210" in call_args.kwargs["message_text"]


@pytest.mark.asyncio
async def test_demo_expert_receives_zero_alerts():
    """Demo / placeholder experts receive ZERO outbound WhatsApp alerts."""
    farmer = MagicMock(id=uuid4(), phone_number="+91 9876543210", preferred_language="en")
    db = AsyncMock()

    demo_expert = _make_mock_expert(name="Dr. K. Srinivas Rao", phone="+91 9848012345")

    with patch("src.config.get_settings") as mock_settings, \
         patch("src.escalation.notifications.send_text_message", new_callable=AsyncMock) as mock_send_wa, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[demo_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True):

        mock_settings.return_value.app_env = "production"
        mock_settings.return_value.expert_whatsapp_group_id = ""

        res = await enrich_response_with_escalation(
            db, "Connect me to an officer", "Advisory note.", farmer
        )

    # Zero outbound WhatsApp calls should be made to demo expert
    assert mock_send_wa.call_count == 0
    # Safe fallback presented to farmer
    assert "+91 9848012345" not in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_expert_alert_failure_is_non_blocking():
    """If Meta WhatsApp API delivery fails, ticket creation and farmer response remain successful."""
    farmer = MagicMock(id=uuid4(), phone_number="+91 9876543210", preferred_language="en")
    db = AsyncMock()

    real_expert = _make_mock_expert(name="Dr. M. Suresh Kumar", specialty="Pest Control", phone="+91 9440123456")

    with patch("src.config.get_settings") as mock_settings, \
         patch("src.escalation.notifications.send_text_message", side_effect=Exception("Meta API Connection Timeout")) as mock_send_wa, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[]), \
         patch.object(EscalationRepository, "get_active_experts", new_callable=AsyncMock, return_value=[real_expert]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock, return_value=True) as mock_record:

        mock_settings.return_value.app_env = "production"
        mock_settings.return_value.expert_whatsapp_group_id = ""

        res = await enrich_response_with_escalation(
            db, "Connect me to an expert officer immediately", "Base advisory.", farmer
        )

    # Ticket was still persisted
    assert mock_record.call_count == 1
    # Response was still cleanly returned to farmer with contact info
    assert "Base advisory." in res
    assert "Dr. M. Suresh Kumar" in res
    assert "1800-180-1551" in res


@pytest.mark.asyncio
async def test_duplicate_ticket_does_not_send_another_alert():
    """Duplicate escalation queries on the same day do not re-send notifications."""
    farmer = MagicMock(id=uuid4(), phone_number="+91 9876543210", preferred_language="en")
    db = AsyncMock()

    from datetime import datetime
    today_str = datetime.utcnow().strftime("%Y%m%d")
    existing_ticket = {
        "ticket_id": f"ESC-{today_str}-5555",
        "status": "Assigned",
        "expert_name": "Dr. M. Suresh Kumar",
        "expert_phone": "+91 9440123456",
    }

    with patch("src.config.get_settings") as mock_settings, \
         patch("src.escalation.notifications.send_text_message", new_callable=AsyncMock) as mock_send_wa, \
         patch.object(EscalationRepository, "seed_default_experts_if_empty", new_callable=AsyncMock), \
         patch.object(EscalationRepository, "get_farmer_consultation_history", new_callable=AsyncMock, return_value=[existing_ticket]), \
         patch.object(EscalationRepository, "record_escalation_ticket", new_callable=AsyncMock) as mock_record:

        mock_settings.return_value.app_env = "production"

        res = await enrich_response_with_escalation(
            db, "I already requested an officer earlier", "Advisory.", farmer
        )

    # No duplicate ticket created in DB
    assert mock_record.call_count == 0
    # No duplicate WhatsApp alert dispatched
    assert mock_send_wa.call_count == 0
    # Existing ticket info returned to farmer
    assert f"ESC-{today_str}-5555" in res


@pytest.mark.asyncio
async def test_group_notification_privacy_safeguard_no_farmer_pii():
    """Group alerts strictly exclude farmer phone numbers, GPS coordinates, and raw conversation history."""
    from src.escalation.notifications import build_expert_alert_message, notify_expert_escalation_ticket

    # 1. Direct message test -> includes farmer phone for 1:1 triage
    direct_msg = build_expert_alert_message(
        ticket_id="ESC-20260825-9999",
        topic="Cotton pink bollworm pest infestation",
        region="Warangal, Telangana",
        reason="hazard",
        farmer_phone="+91 9876543210",
        is_group=False,
    )
    assert "9876543210" in direct_msg
    assert "Urgent Chemical Hazard" in direct_msg

    # 2. Group message test -> strictly OMITS farmer phone and PII
    group_msg = build_expert_alert_message(
        ticket_id="ESC-20260825-9999",
        topic="Cotton pink bollworm pest infestation",
        region="Warangal, Telangana",
        reason="hazard",
        farmer_phone="+91 9876543210",
        is_group=True,
    )
    assert "9876543210" not in group_msg
    assert "Farmer Contact" not in group_msg
    assert "ESC-20260825-9999" in group_msg
    assert "Warangal, Telangana" in group_msg

    # 3. Async group dispatch test with group_id configured
    with patch("src.escalation.notifications.send_text_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "wa_msg_group_123"
        res = await notify_expert_escalation_ticket(
            ticket_id="ESC-20260825-9999",
            topic="Cotton pest",
            region="Warangal",
            reason="inspection",
            expert_phone=None,
            farmer_phone="+91 9876543210",
            group_id="1203630248572910@g.us",
        )
        assert res == "wa_msg_group_123"
        call_msg = mock_send.call_args.kwargs["message_text"]
        assert "9876543210" not in call_msg
        assert "Farmer Contact" not in call_msg
