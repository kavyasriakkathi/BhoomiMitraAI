"""
Tests for Cotton Disease Diagnosis & Multi-Turn Farmer Context Retention.

Verifies:
1. Extraction of Location (Korutla -> Jagtial, Telangana) and Crop (Cotton) into FarmerMemory & FarmerProfile.
2. Context retention in multi-turn conversation (bot never re-asks for crop or location).
3. Grounded RAG diagnosis for Alternaria Leaf Spot without deflecting to a photo request.
4. Correct treatment and dosage recommendations (Mancozeb 75% WP @ 2.5-3.0 g/L).
5. Non-interference with weather enrichment (disease query does not trigger weather location prompt).
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from src.core.database import Base
from src.core.models import Farmer, FarmerProfile, Conversation
from src.memory.models import FarmerMemory
from src.memory.service import FarmerMemoryService
from src.memory.repository import FarmerMemoryRepository
from src.ai.service import AIService, process_text_message
from src.ai.schemas import AIGenerateRequest
from src.rag.service import RAGService


import pytest_asyncio

@pytest_asyncio.fixture
async def in_memory_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_location_and_crop_extraction_from_initial_greeting(in_memory_db):
    """
    Verify that when farmer replies with:
      📍 Location: Korutla
      🌾 Crop: Cotton
    the heuristic memory extraction records:
      - village: Korutla
      - district: Jagtial (auto-resolved from _KNOWN_DISTRICTS)
      - state: Telangana
      - primary_crops: ["Cotton"]
    and synchronizes FarmerProfile.
    """
    farmer = Farmer(id=uuid4(), phone_number="+919876543210")
    in_memory_db.add(farmer)
    await in_memory_db.commit()

    profile = FarmerProfile(farmer_id=farmer.id)
    in_memory_db.add(profile)
    await in_memory_db.commit()

    mem_repo = FarmerMemoryRepository(in_memory_db)
    mem_service = FarmerMemoryService(mem_repo)

    user_msg = "Got it! 🌱\n📍 Location: Korutla\n🌾 Crop: Cotton"
    with patch("src.memory.service.generate_response", return_value=None):
        mem = await mem_service.extract_and_update_memory(
            farmer_id=farmer.id,
            user_message=user_msg,
            ai_response="I can help you with cotton in Korutla."
        )

    assert mem.village == "Korutla"
    assert mem.district == "Jagtial"
    assert mem.state == "Telangana"
    assert "Cotton" in mem.primary_crops

    # Verify FarmerProfile was synchronized
    res = await in_memory_db.execute(
        select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
    )
    saved_profile = res.scalar_one()
    assert saved_profile.current_crop == "Cotton"
    assert saved_profile.district == "Jagtial"
    assert saved_profile.state == "Telangana"


@pytest.mark.asyncio
async def test_disease_diagnosis_retains_context_and_prescribes_rag_treatment(in_memory_db):
    """
    Turn 2 disease-diagnosis query:
    Farmer asks:
      "I noticed reddish-brown circular spots with concentric rings on my cotton leaves,
       and some lower leaves are turning yellow and drying up. What is this issue and what treatment should I spray?"

    Verifies:
      1. Effective crop (Cotton) and location (Korutla / Jagtial) are injected into system prompt.
      2. Prompt explicitly instructs model NOT to re-ask for crop, village, or location.
      3. RAG retrieves Alternaria Leaf Spot package of practices with Mancozeb dosage.
      4. AI response identifies Alternaria Leaf Spot and prescribes Mancozeb 75% WP @ 2.5-3.0 g/L.
      5. Response does NOT ask for crop or location.
      6. Response does NOT deflect with a generic request for a photo.
    """
    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="en")
    in_memory_db.add(farmer)

    profile = FarmerProfile(
        farmer_id=farmer.id,
        current_crop="Cotton",
        district="Jagtial",
        state="Telangana",
    )
    in_memory_db.add(profile)

    memory = FarmerMemory(
        farmer_id=farmer.id,
        village="Korutla",
        district="Jagtial",
        state="Telangana",
        primary_crops=["Cotton"],
    )
    in_memory_db.add(memory)
    await in_memory_db.commit()

    disease_query = (
        "I noticed reddish-brown circular spots with concentric rings on my cotton leaves, "
        "and some lower leaves are turning yellow and drying up. What is this issue and what treatment should I spray?"
    )

    # Mock AI Repository backed by in_memory_db
    class MockAIRepo:
        def __init__(self, session):
            self.session = session

        async def get_farmer_profile(self, farmer_id):
            res = await self.session.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id)
            )
            return res.scalar_one_or_none()

        async def get_conversation_history(self, farmer_id, limit=10):
            return []

    ai_repo = MockAIRepo(in_memory_db)
    ai_service = AIService(ai_repo)

    captured_data = {}
    async def mock_gemini(system_prompt, conversation_history, user_message, **kwargs):
        captured_data["system_prompt"] = system_prompt
        return (
            "The symptoms on your cotton crop indicate Alternaria Leaf Spot (fungal disease). "
            "To manage this, spray Mancozeb 75% WP @ 2.5 to 3.0 g per litre of water (approx. 500-600 g per acre in 200 L water) "
            "or Copper Oxychloride 50% WP @ 3.0 g per litre of water. "
            "Spray thoroughly on both upper and lower leaf surfaces during early morning or late afternoon hours."
        )

    req = AIGenerateRequest(farmer_id=farmer.id, message=disease_query)

    with patch("src.ai.service.generate_response", side_effect=mock_gemini):
        resp = await ai_service.generate_ai_response(req)

    # 1. System Prompt Context Retention
    sys_prompt = captured_data["system_prompt"]
    assert "Current Crop: Cotton" in sys_prompt
    assert "Village/Location: Korutla" in sys_prompt
    assert "District: Jagtial" in sys_prompt
    assert "State: Telangana" in sys_prompt
    assert "Note: The farmer's crop and location are ALREADY KNOWN and verified." in sys_prompt
    assert "Alternaria Leaf Spot" in sys_prompt
    assert "Mancozeb 75% WP" in sys_prompt
    assert "2.5 to 3.0 g per litre" in sys_prompt

    # 2. AI Response Quality & Grounding
    res_text = resp.response_text
    assert "Alternaria" in res_text
    assert "Mancozeb 75% WP" in res_text
    assert "2.5 to 3.0 g" in res_text

    # 3. No re-asking for crop or location
    assert "Which crop" not in res_text
    assert "What is your location" not in res_text
    assert "district" not in res_text.lower() or "jagtial" in res_text.lower()
    assert "please provide your village" not in res_text.lower()

    # 4. No deflecting to photo
    assert "please send a photo" not in res_text.lower()
    assert "upload an image" not in res_text.lower()


@pytest.mark.asyncio
async def test_disease_query_does_not_falsely_trigger_weather_location_prompt(in_memory_db):
    """
    Ensure that passing the disease query through the full process_text_message pipeline
    does NOT trigger enrich_response_with_weather with an ask_location prompt.
    """
    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="en")
    in_memory_db.add(farmer)

    profile = FarmerProfile(
        farmer_id=farmer.id,
        current_crop="Cotton",
        district="Jagtial",
        state="Telangana",
    )
    in_memory_db.add(profile)

    memory = FarmerMemory(
        farmer_id=farmer.id,
        village="Korutla",
        district="Jagtial",
        state="Telangana",
        primary_crops=["Cotton"],
    )
    in_memory_db.add(memory)

    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        message_id="wamid.DISEASE_TEST_101",
        user_message=(
            "I noticed reddish-brown circular spots with concentric rings on my cotton leaves, "
            "and some lower leaves are turning yellow and drying up. What is this issue and what treatment should I spray?"
        ),
    )
    in_memory_db.add(conv)
    await in_memory_db.commit()

    mock_ai_resp = (
        "The symptoms indicate Alternaria Leaf Spot. Spray Mancozeb 75% WP @ 2.5-3.0 g per litre of water."
    )

    with patch("src.ai.service.AIService.generate_ai_response") as mock_gen, \
         patch("src.ai.service.AIRepository.get_farmer_profile", return_value=profile):
        mock_gen.return_value.response_text = mock_ai_resp

        final_reply = await process_text_message(in_memory_db, farmer, conv)

        assert "Alternaria" in final_reply
        assert "Mancozeb" in final_reply
        # Must not contain weather prompt asking for location
        assert "Please provide your district or area name" not in final_reply
        assert "📍 Please provide" not in final_reply
