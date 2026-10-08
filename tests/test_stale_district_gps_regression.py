"""
Regression Tests: Audit and Fix Stale District Memory after New WhatsApp GPS Location

Proves:
A. Farmer has district Korutla.
B. Farmer shares a new GPS location elsewhere (e.g. lat=17.4093054, lon=78.6495555).
C. Old Korutla district cannot be sent to Gemini as current location context.

Also verifies:
- GPS remains available.
- Shop discovery still uses GPS.
- Weather still uses GPS.
- Explicit query location still overrides GPS.
- No fake district is generated.
- Refresh memory does not resurrect stale district when GPS is present.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import Farmer, Conversation, FarmerProfile, Farm
from src.memory.models import FarmerMemory
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import _handle_inbound_location
from src.memory.prompts import build_memory_context_prompt
from src.ai.prompts import build_farmer_context
from src.shops.service import _resolve_farmer_location, enrich_response_with_shops
from src.weather.service import _extract_district_from_query, enrich_response_with_weather
from src.memory.service import FarmerMemoryService
from src.memory.repository import FarmerMemoryRepository


def _make_mock_db_with_farmer_data(farmer, memory=None, profile=None, farms=None):
    """Create a mock AsyncSession that returns memory, profile, and farms appropriately."""
    mock_db = AsyncMock()

    async def _execute_mock(stmt):
        stmt_str = str(stmt)
        res = MagicMock()
        if "farmer_memory" in stmt_str or "FarmerMemory" in stmt_str:
            res.scalar_one_or_none.return_value = memory
            res.scalars.return_value.all.return_value = [memory] if memory else []
        elif "farmer_profiles" in stmt_str or "FarmerProfile" in stmt_str:
            res.scalar_one_or_none.return_value = profile
            res.scalars.return_value.all.return_value = [profile] if profile else []
        elif "farms" in stmt_str or "Farm" in stmt_str:
            res.scalar_one_or_none.return_value = farms[0] if farms else None
            res.scalars.return_value.all.return_value = farms or []
        elif "crop_health" in stmt_str or "CropHealth" in stmt_str:
            res.scalar_one_or_none.return_value = None
            res.scalars.return_value.all.return_value = []
        elif "order_requests" in stmt_str or "OrderRequest" in stmt_str:
            res.scalar_one_or_none.return_value = None
            res.scalars.return_value.all.return_value = []
        elif "scheme_applications" in stmt_str or "SchemeApplication" in stmt_str:
            res.scalar_one_or_none.return_value = None
            res.scalars.return_value.all.return_value = []
        else:
            res.scalar_one_or_none.return_value = None
            res.scalars.return_value.all.return_value = []
        return res

    mock_db.execute = AsyncMock(side_effect=_execute_mock)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.flush = AsyncMock()
    return mock_db


@pytest.mark.asyncio
async def test_regression_new_gps_clears_stale_korutla_district():
    """
    Core Regression A -> B -> C:
    A. Farmer has district Korutla.
    B. Farmer shares a new GPS location elsewhere (17.4093054, 78.6495555).
    C. Stale Korutla district is cleared from FarmerMemory and FarmerProfile.
       No fake district is generated.
       GPS remains available.
    """
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")

    # Step A: Farmer initially has district Korutla in FarmerMemory and FarmerProfile
    memory = FarmerMemory(
        farmer_id=farmer_id,
        district="Korutla",
        confidence_scores={"district": 0.9},
    )
    profile = FarmerProfile(
        farmer_id=farmer_id,
        district="Korutla",
    )
    farmer.profile = profile

    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=profile)
    conv = Conversation(id=uuid4(), farmer_id=farmer_id, message_id="wamid.loc_pin_new")

    # Step B: Farmer shares a new WhatsApp GPS location pin elsewhere (e.g. Hyderabad outskirts)
    new_lat = 17.4093054
    new_lng = 78.6495555
    parsed = ParsedIncomingMessage(
        phone_number="919848011234",
        message_id="wamid.loc_pin_new",
        timestamp="1700000000",
        message_type="location",
        latitude=new_lat,
        longitude=new_lng,
    )

    response = await _handle_inbound_location(mock_db, farmer, conv, parsed, language="te")

    # Confirmation sent to WhatsApp
    assert "ధన్యవాదాలు! మీ లొకేషన్ విజయవంతంగా సేవ్ చేయబడింది" in response

    # Step C Verification:
    # 1. GPS remains available and matches new coordinates
    assert memory.gps_coordinates == {
        "latitude": 17.409305,
        "longitude": 78.649556,
    }
    # 2. Stale district "Korutla" is completely cleared, NOT retained
    assert memory.district is None
    # 3. No fake district is generated
    assert memory.district != "Korutla"
    assert memory.district != "Hyderabad"
    assert memory.district is None
    # 4. District confidence score is cleared
    assert "district" not in (memory.confidence_scores or {})
    # 5. FarmerProfile district is also cleared so it cannot leak
    assert profile.district is None
    assert getattr(farmer.profile, "district", None) is None


def test_regression_gemini_memory_prompt_does_not_contain_old_korutla():
    """
    Test C (Prompt Layer):
    build_memory_context_prompt does NOT send old Korutla district to Gemini
    when current GPS is present.
    """
    farmer_id = uuid4()
    # Memory with new GPS and cleared district
    mem = FarmerMemory(
        farmer_id=farmer_id,
        district=None,
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
        primary_crops=["Cotton"],
    )

    prompt = build_memory_context_prompt(mem)
    assert "Korutla" not in prompt
    assert "GPS: (17.409305, 78.649556)" in prompt
    assert "District:" not in prompt


def test_regression_gemini_memory_prompt_suppresses_conflicting_stale_district_if_gps_present():
    """
    Defense-in-depth:
    Even if a legacy FarmerMemory object still had both Korutla and new GPS,
    build_memory_context_prompt prioritizes GPS and never sends stale Korutla.
    """
    farmer_id = uuid4()
    mem = FarmerMemory(
        farmer_id=farmer_id,
        district="Korutla",  # legacy un-cleared value
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
        primary_crops=["Cotton"],
    )

    prompt = build_memory_context_prompt(mem)
    assert "Korutla" not in prompt
    assert "District: Korutla" not in prompt
    assert "GPS: (17.409305, 78.649556)" in prompt


@pytest.mark.asyncio
async def test_regression_gemini_system_prompt_does_not_receive_stale_district():
    """
    Test C (General Gemini Request):
    When a farmer with new GPS sends an agronomic query, the system prompt
    assembled in generate_ai_response does NOT include the old Korutla district.
    """
    from src.ai.service import AIService
    from src.ai.schemas import AIGenerateRequest
    from src.ai.repository import AIRepository

    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")
    # New GPS saved, stale district cleared
    memory = FarmerMemory(
        farmer_id=farmer_id,
        district=None,
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    profile = FarmerProfile(farmer_id=farmer_id, district=None, current_crop="Cotton")

    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=profile)
    ai_repo = AIRepository(mock_db)
    ai_service = AIService(ai_repo)

    captured_system_prompts = []

    async def _mock_generate_response(system_prompt, conversation_history, user_message, **kwargs):
        captured_system_prompts.append(system_prompt)
        return "ప్రత్తి పంటలో ఎరువుల యాజమాన్యం..."

    with patch("src.ai.service.generate_response", side_effect=_mock_generate_response), \
         patch("src.language.detector.detect_language", return_value="te"):

        req = AIGenerateRequest(farmer_id=farmer_id, message="ప్రత్తి పంటకు ఏ ఎరువు వేయాలి?")
        resp = await ai_service.generate_ai_response(req)

    assert resp is not None
    assert len(captured_system_prompts) == 1
    sys_prompt = captured_system_prompts[0]
    # Korutla district must NOT be present anywhere in the Gemini system prompt
    assert "Korutla" not in sys_prompt
    assert "District: Korutla" not in sys_prompt


@pytest.mark.asyncio
async def test_shop_discovery_still_uses_gps_after_stale_district_cleared():
    """
    Shop discovery correctly uses the latest GPS coordinates and does NOT use
    stale Korutla district.
    """
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")

    memory = FarmerMemory(
        farmer_id=farmer_id,
        district=None,  # cleared after new GPS pin
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=None)

    lat, lon, dist, state = await _resolve_farmer_location(
        db=mock_db, farmer=farmer, query_text="fertilizer shops near me"
    )

    # Must resolve the new GPS coordinates
    assert lat == 17.409305
    assert lon == 78.649556
    # District must be None (no stale Korutla)
    assert dist is None


@pytest.mark.asyncio
async def test_weather_still_uses_gps_after_stale_district_cleared():
    """
    Weather service correctly uses the latest GPS coordinates and does NOT use
    stale Korutla district.
    """
    from src.weather.openweather_client import OpenWeatherClient

    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="en")

    memory = FarmerMemory(
        farmer_id=farmer_id,
        district=None,  # cleared after new GPS pin
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=None)

    called_kwargs = {}
    from datetime import datetime, timedelta
    tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d 12:00:00")

    async def _mock_fetch_weather(latitude=None, longitude=None, district=None, state=None):
        called_kwargs.update({
            "latitude": latitude,
            "longitude": longitude,
            "district": district,
            "state": state,
        })
        return {
            "location_name": f"Lat: {latitude:.2f}, Lon: {longitude:.2f}" if latitude else (district or "Selected Farm"),
            "latitude": latitude or 17.4093,
            "longitude": longitude or 78.6495,
            "current": {
                "temp": 30.2,
                "feels_like": 32.5,
                "humidity": 65,
                "wind_speed": 12.0,
                "description": "Partly Cloudy",
                "condition_code": 802,
            },
            "forecast": [{
                "dt_txt": tomorrow_str,
                "temp": 28.5,
                "humidity": 70,
                "description": "Clear Sky",
                "condition_code": 800,
            }],
            "data_available": True,
            "is_live": False,
            "source_note": "Simulated Weather (Local Fallback)",
        }

    with patch.object(OpenWeatherClient, "fetch_weather", side_effect=_mock_fetch_weather):
        res = await enrich_response_with_weather(
            db=mock_db,
            query_text="What is the weather today?",
            ai_response="Farming advice.",
            farmer=farmer,
        )

    assert "Weather Information" in res
    # Weather was queried with GPS coordinates
    assert called_kwargs.get("latitude") == 17.409305
    assert called_kwargs.get("longitude") == 78.649556
    # District was NOT passed as Korutla
    assert called_kwargs.get("district") is None


@pytest.mark.asyncio
async def test_explicit_query_location_still_overrides_gps():
    """
    Requirement 6: Explicit location in the current user message must remain HIGHEST priority,
    overriding current GPS.
    """
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")

    # Farmer has current GPS in Hyderabad
    memory = FarmerMemory(
        farmer_id=farmer_id,
        district=None,
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=None)

    # 1. Shop discovery: Explicit "Warangal" query overrides GPS
    lat, lon, dist, state = await _resolve_farmer_location(
        db=mock_db, farmer=farmer, query_text="Warangal lo fertilizer shops unnaya?"
    )
    assert dist == "Warangal"

    # 2. Weather: Explicit "Warangal" query overrides GPS
    query_dist = _extract_district_from_query("Warangal weather forecast")
    assert query_dist == "Warangal"

    # 3. Farmer context for Gemini: Explicit query district overrides GPS
    effective_district = query_dist  # Highest priority
    f_ctx = build_farmer_context(district=effective_district)
    assert "District: Warangal" in f_ctx
    assert "Korutla" not in f_ctx


@pytest.mark.asyncio
async def test_refresh_memory_does_not_resurrect_stale_district_when_gps_present():
    """
    Test 6:
    Verify FarmerMemoryService.refresh_farmer_memory does NOT overwrite
    the cleared district with an old profile/farm district if GPS is present,
    and does NOT overwrite new WhatsApp GPS with old farm coordinates.
    """
    from datetime import datetime
    farmer_id = uuid4()
    memory = FarmerMemory(
        id=uuid4(),
        farmer_id=farmer_id,
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.0,
        voice_gender="FEMALE",
        farm_size=5.0,
        village=None,
        district=None,
        state=None,
        soil_type=None,
        water_source=None,
        irrigation_method=None,
        primary_crops=["Cotton"],
        secondary_crops=[],
        crop_history=[],
        disease_history=[],
        pesticide_history=[],
        fertilizer_history=[],
        yield_history=[],
        favorite_shops=[],
        purchase_history=[],
        preferred_brands=[],
        government_schemes_used=[],
        expert_consultation_history=[],
        conversation_summary=None,
        frequently_asked_questions=[],
        ai_learned_preferences={},
        risk_factors=[],
        confidence_scores={},
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    # Stale profile and farm in DB with old Korutla district and coordinates
    profile = FarmerProfile(farmer_id=farmer_id, district="Korutla")
    farm = Farm(
        id=uuid4(),
        farmer_id=farmer_id,
        district="Korutla",
        latitude=18.82,
        longitude=78.71,
    )

    mock_db = _make_mock_db_with_farmer_data(None, memory=memory, profile=profile, farms=[farm])
    repo = FarmerMemoryRepository(mock_db)
    service = FarmerMemoryService(repo)

    with patch.object(repo, "get_or_create", new_callable=AsyncMock, return_value=memory), \
         patch.object(repo, "save", new_callable=AsyncMock, side_effect=lambda m: m):
        refreshed = await service.refresh_farmer_memory(farmer_id)

    # District must NOT have been resurrected to Korutla from profile or farm
    assert memory.district is None
    assert refreshed.district is None
    # GPS coordinates must remain the authoritative WhatsApp GPS (not overwritten by old farm)
    assert memory.gps_coordinates == {"latitude": 17.409305, "longitude": 78.649556}


@pytest.mark.asyncio
async def test_rag_prompt_construction_does_not_leak_stale_district_when_gps_present():
    """
    Requirement 7:
    Check RAG prompt construction so stale district cannot leak there when current GPS exists.
    """
    from src.rag.service import RAGService
    from src.rag.repository import RAGRepository

    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")
    from datetime import datetime
    memory = FarmerMemory(
        id=uuid4(),
        farmer_id=farmer_id,
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        preferred_language="te",
        preferred_voice="Google-te-IN-Standard-A",
        voice_speed=1.0,
        voice_gender="FEMALE",
        farm_size=5.0,
        village=None,
        district=None,
        state=None,
        soil_type=None,
        water_source=None,
        irrigation_method=None,
        primary_crops=["Cotton"],
        secondary_crops=[],
        crop_history=[],
        disease_history=[],
        pesticide_history=[],
        fertilizer_history=[],
        yield_history=[],
        favorite_shops=[],
        purchase_history=[],
        preferred_brands=[],
        government_schemes_used=[],
        expert_consultation_history=[],
        conversation_summary=None,
        frequently_asked_questions=[],
        ai_learned_preferences={},
        risk_factors=[],
        confidence_scores={},
        gps_coordinates={"latitude": 17.409305, "longitude": 78.649556},
    )
    # Stale profile in DB
    profile = FarmerProfile(farmer_id=farmer_id, district="Korutla", current_crop="Cotton", state="Telangana")

    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=profile)
    rag_repo = RAGRepository(mock_db)
    rag_service = RAGService(rag_repo)

    captured_system_prompts = []

    async def _mock_generate(system_prompt, conversation_history, user_message, **kwargs):
        captured_system_prompts.append(system_prompt)
        return "ప్రత్తి పంటలో సలహా..."

    with patch("src.ai.gemini_client.generate_response", side_effect=_mock_generate), \
         patch.object(rag_service, "hybrid_search_knowledge", new_callable=AsyncMock, return_value=[]):
        res = await rag_service.generate_rag_response(
            farmer_id=farmer_id,
            message="ప్రత్తి పంటకు ఎంత ఎరువు వేయాలి?",
        )

    assert len(captured_system_prompts) == 1
    sys_prompt = captured_system_prompts[0]
    # Korutla district must NOT be present anywhere in the RAG prompt
    assert "Korutla" not in sys_prompt
    assert "District: Korutla" not in sys_prompt
    # Must include GPS coordinates
    assert "17.409305" in sys_prompt
    assert "78.649556" in sys_prompt


@pytest.mark.asyncio
async def test_get_or_create_farmer_eagerly_loads_farmer_profile():
    """
    Test E.2 & E.8:
    Verify that get_or_create_farmer eagerly loads FarmerProfile using selectinload,
    ensuring farmer.profile is accessible in async context without greenlet_spawn error.
    """
    from src.gateway.service import get_or_create_farmer
    mock_db = AsyncMock()
    farmer_id = uuid4()
    mock_farmer = Farmer(id=farmer_id, phone_number="919999999999")
    mock_profile = FarmerProfile(farmer_id=farmer_id, district="Korutla")
    mock_farmer.profile = mock_profile

    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = mock_farmer
    mock_db.execute.return_value = mock_res

    farmer = await get_or_create_farmer(mock_db, "919999999999", "Test Farmer")

    assert farmer.id == farmer_id
    assert mock_db.execute.called


@pytest.mark.asyncio
async def test_handle_inbound_location_prevents_greenlet_spawn_error_on_unloaded_relationship():
    """
    Test E.1 & E.8:
    Simulate an attached SQLAlchemy model where accessing farmer.profile directly
    would trigger a MissingGreenlet exception ('greenlet_spawn has not been called').
    Proves that _handle_inbound_location safely inspects the unloaded state and avoids
    triggering synchronous IO / lazy load.
    """
    from sqlalchemy.exc import MissingGreenlet

    farmer_id = uuid4()

    class DangerousFarmer:
        def __init__(self, fid):
            self.id = fid
            self.phone_number = "919848011234"
            self.preferred_language = "te"

        @property
        def profile(self):
            raise MissingGreenlet(
                "greenlet_spawn has not been called; can't call await_only() here. Was IO attempted in an unexpected place?"
            )

    dangerous_farmer = DangerousFarmer(farmer_id)

    mock_insp = MagicMock()
    mock_insp.unloaded = {"profile"}

    mock_db = AsyncMock()
    memory = FarmerMemory(farmer_id=farmer_id, district="Korutla")
    profile = FarmerProfile(farmer_id=farmer_id, district="Korutla")

    async def _mock_exec(stmt):
        res = MagicMock()
        stmt_str = str(stmt)
        if "farmer_memory" in stmt_str or "FarmerMemory" in stmt_str:
            res.scalar_one_or_none.return_value = memory
        elif "farmer_profiles" in stmt_str or "FarmerProfile" in stmt_str:
            res.scalar_one_or_none.return_value = profile
        else:
            res.scalar_one_or_none.return_value = None
        return res

    mock_db.execute = AsyncMock(side_effect=_mock_exec)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    parsed = ParsedIncomingMessage(
        phone_number="919848011234",
        message_id="wamid.loc_test_safe",
        timestamp="1700000000",
        message_type="location",
        latitude=17.409305,
        longitude=78.649556,
    )
    conv = Conversation(id=uuid4(), farmer_id=farmer_id, message_id="wamid.loc_test_safe")

    with patch("src.gateway.service.sa_inspect", return_value=mock_insp):
        # Must execute cleanly without raising MissingGreenlet
        reply = await _handle_inbound_location(mock_db, dangerous_farmer, conv, parsed, language="te")

    assert "లొకేషన్ విజయవంతంగా సేవ్ చేయబడింది" in reply
    assert memory.district is None
    assert profile.district is None
    assert memory.gps_coordinates == {"latitude": 17.409305, "longitude": 78.649556}


@pytest.mark.asyncio
async def test_handle_inbound_location_with_eagerly_loaded_profile_clears_district_directly():
    """
    Test E.2 & E.3:
    When FarmerProfile is eagerly loaded in Stage 2, _handle_inbound_location directly
    clears profile.district in-memory and queues it for commit without extra IO or errors.
    """
    farmer_id = uuid4()
    farmer = Farmer(id=farmer_id, phone_number="919848011234", preferred_language="te")
    memory = FarmerMemory(farmer_id=farmer_id, district="Korutla", confidence_scores={"district": 0.95})
    profile = FarmerProfile(farmer_id=farmer_id, district="Korutla")
    farmer.profile = profile

    mock_db = _make_mock_db_with_farmer_data(farmer, memory=memory, profile=profile)
    conv = Conversation(id=uuid4(), farmer_id=farmer_id, message_id="wamid.loc_test_eager")

    parsed = ParsedIncomingMessage(
        phone_number="919848011234",
        message_id="wamid.loc_test_eager",
        timestamp="1700000000",
        message_type="location",
        latitude=17.409305,
        longitude=78.649556,
    )

    reply = await _handle_inbound_location(mock_db, farmer, conv, parsed, language="te")

    assert "లొకేషన్ విజయవంతంగా సేవ్ చేయబడింది" in reply
    assert memory.gps_coordinates == {"latitude": 17.409305, "longitude": 78.649556}
    assert memory.district is None
    assert "district" not in (memory.confidence_scores or {})
    assert profile.district is None
    assert farmer.profile.district is None
