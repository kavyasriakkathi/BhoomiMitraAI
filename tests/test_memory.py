import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

from src.main import app
from src.memory.schemas import (
    FarmerMemoryResponse,
    FarmerMemoryUpdate,
    FarmerMemorySummaryResponse,
    VoiceSettingsResponse,
)
from src.memory.service import FarmerMemoryService
from src.memory.dependencies import get_memory_service
from src.memory.prompts import build_memory_context_prompt

client = TestClient(app)

@pytest.fixture
def mock_memory_service():
    service = AsyncMock(spec=FarmerMemoryService)
    app.dependency_overrides[get_memory_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_get_farmer_memory_success(mock_memory_service):
    farmer_id = uuid4()
    memory_id = uuid4()
    now = datetime.utcnow()

    mock_resp = FarmerMemoryResponse(
        id=memory_id,
        farmer_id=farmer_id,
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.0,
        voice_gender="FEMALE",
        farm_size=5.0,
        village="Karimnagar",
        district="Karimnagar",
        state="Telangana",
        gps_coordinates={"latitude": 18.43, "longitude": 79.12},
        soil_type="Black",
        water_source="Borewell",
        irrigation_method="Drip",
        primary_crops=["Cotton", "Paddy"],
        secondary_crops=["Chilli"],
        crop_history=[],
        disease_history=[],
        pesticide_history=[],
        fertilizer_history=[],
        yield_history=[],
        weather_region="North Telangana",
        favorite_shops=["Ramesh Fertilizers"],
        purchase_history=[],
        preferred_brands=["Bayer"],
        budget_range="Medium",
        government_schemes_used=["PM-KISAN"],
        expert_consultation_history=[],
        conversation_summary="Farmer grows cotton on 5 acres in Karimnagar.",
        frequently_asked_questions=[],
        ai_learned_preferences={"prefers_organic": True},
        risk_factors=["Pest vulnerability"],
        confidence_scores={"farm_size": 0.9, "village": 0.9},
        created_at=now,
        last_updated=now
    )

    mock_memory_service.get_memory_response.return_value = mock_resp

    response = client.get(f"/memory/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["farmer_id"] == str(farmer_id)
    assert data["farm_size"] == 5.0
    assert data["village"] == "Karimnagar"
    assert "Cotton" in data["primary_crops"]
    assert "Ramesh Fertilizers" in data["favorite_shops"]


def test_update_farmer_memory(mock_memory_service):
    farmer_id = uuid4()
    memory_id = uuid4()
    now = datetime.utcnow()

    mock_resp = FarmerMemoryResponse(
        id=memory_id,
        farmer_id=farmer_id,
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.2,
        voice_gender="FEMALE",
        farm_size=10.0,
        village="Warangal",
        district="Warangal",
        state="Telangana",
        gps_coordinates={},
        soil_type="Red",
        water_source="Canal",
        irrigation_method="Flood",
        primary_crops=["Paddy"],
        secondary_crops=[],
        crop_history=[],
        disease_history=[],
        pesticide_history=[],
        fertilizer_history=[],
        yield_history=[],
        weather_region=None,
        favorite_shops=["Sri Krishna Seeds"],
        purchase_history=[],
        preferred_brands=[],
        budget_range=None,
        government_schemes_used=[],
        expert_consultation_history=[],
        conversation_summary=None,
        frequently_asked_questions=[],
        ai_learned_preferences={},
        risk_factors=[],
        confidence_scores={},
        created_at=now,
        last_updated=now
    )

    mock_memory_service.update_memory.return_value = mock_resp

    payload = {
        "farm_size": 10.0,
        "village": "Warangal",
        "voice_speed": 1.2
    }

    response = client.put(f"/memory/{farmer_id}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["farm_size"] == 10.0
    assert data["village"] == "Warangal"
    assert data["voice_speed"] == 1.2


def test_refresh_farmer_memory(mock_memory_service):
    farmer_id = uuid4()
    memory_id = uuid4()
    now = datetime.utcnow()

    mock_resp = FarmerMemoryResponse(
        id=memory_id,
        farmer_id=farmer_id,
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.0,
        voice_gender="FEMALE",
        farm_size=5.0,
        village="Nalgonda",
        district="Nalgonda",
        state="Telangana",
        gps_coordinates={},
        soil_type="Alluvial",
        water_source="Canal",
        irrigation_method="Drip",
        primary_crops=["Cotton"],
        secondary_crops=[],
        crop_history=[],
        disease_history=[],
        pesticide_history=[],
        fertilizer_history=[],
        yield_history=[],
        weather_region=None,
        favorite_shops=[],
        purchase_history=[],
        preferred_brands=[],
        budget_range=None,
        government_schemes_used=[],
        expert_consultation_history=[],
        conversation_summary=None,
        frequently_asked_questions=[],
        ai_learned_preferences={},
        risk_factors=[],
        confidence_scores={},
        created_at=now,
        last_updated=now
    )

    mock_memory_service.refresh_farmer_memory.return_value = mock_resp

    response = client.post("/memory/refresh", json={"farmer_id": str(farmer_id)})
    assert response.status_code == 200
    data = response.json()
    assert data["farmer_id"] == str(farmer_id)
    assert data["village"] == "Nalgonda"


def test_get_farmer_memory_summary(mock_memory_service):
    farmer_id = uuid4()
    memory_id = uuid4()
    now = datetime.utcnow()

    from src.memory.models import FarmerMemory
    mock_memory = FarmerMemory(
        id=memory_id,
        farmer_id=farmer_id,
        conversation_summary="Farmer reported pink bollworm on cotton. Recommended neem oil.",
        primary_crops=["Cotton"],
        district="Warangal",
        risk_factors=["Pink Bollworm"],
        last_updated=now
    )
    mock_memory_service.get_memory.return_value = mock_memory

    response = client.get(f"/memory/summary/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["farmer_id"] == str(farmer_id)
    assert "pink bollworm" in data["summary"].lower()
    assert data["primary_crops"] == ["Cotton"]


def test_get_farmer_voice_settings(mock_memory_service):
    farmer_id = uuid4()
    mock_voice = VoiceSettingsResponse(
        farmer_id=farmer_id,
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.0,
        voice_gender="FEMALE"
    )
    mock_memory_service.get_voice_settings.return_value = mock_voice

    response = client.get(f"/memory/voice/{farmer_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["preferred_language"] == "te"
    assert data["voice_gender"] == "FEMALE"


def test_build_memory_context_prompt():
    from src.memory.models import FarmerMemory
    farmer_id = uuid4()

    mem = FarmerMemory(
        farmer_id=farmer_id,
        village="Karimnagar",
        district="Karimnagar",
        farm_size=5.0,
        soil_type="Black",
        water_source="Borewell",
        primary_crops=["Cotton", "Paddy"],
        favorite_shops=["Ramesh Fertilizers"],
        conversation_summary="High yield expected this season."
    )

    prompt = build_memory_context_prompt(mem)
    assert "Karimnagar" in prompt
    assert "5.0 acres" in prompt
    assert "Black" in prompt
    assert "Cotton" in prompt
    assert "Ramesh Fertilizers" in prompt
    assert "High yield expected" in prompt


@pytest.mark.asyncio
async def test_extract_and_update_memory_heuristics():
    """Unit test heuristic regex memory extraction in service."""
    from src.memory.service import FarmerMemoryService
    from src.memory.repository import FarmerMemoryRepository
    from unittest.mock import MagicMock

    mock_session = AsyncMock()
    mock_repo = MagicMock(spec=FarmerMemoryRepository)
    mock_repo.session = mock_session

    from src.memory.models import FarmerMemory
    farmer_id = uuid4()
    mem = FarmerMemory(
        farmer_id=farmer_id,
        primary_crops=[],
        favorite_shops=[],
        confidence_scores={}
    )
    mock_repo.get_or_create.return_value = mem
    mock_repo.save.side_effect = lambda m: m

    service = FarmerMemoryService(mock_repo)

    with patch("src.memory.service.generate_response", return_value=None):
        updated_mem = await service.extract_and_update_memory(
            farmer_id=farmer_id,
            user_message="My farm is 5 acres and my village is Karimnagar. I grow cotton and buy from Ramesh Fertilizers.",
            ai_response="Great! Cotton is suitable for Karimnagar."
        )

        assert updated_mem.farm_size == 5.0
        assert updated_mem.village == "Karimnagar"
        assert "Cotton" in updated_mem.primary_crops
        assert "Ramesh Fertilizers" in updated_mem.favorite_shops
        assert updated_mem.confidence_scores.get("farm_size") == 0.9
        assert updated_mem.confidence_scores.get("village") == 0.9
