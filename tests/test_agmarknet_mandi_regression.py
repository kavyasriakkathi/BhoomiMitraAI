"""
Regression tests for BhoomiMitra AI Agmarknet Mandi and Location Processing.

Tests:
1. Agmarknet connection failure handling, bounded retries, and secret redaction.
2. Location normalization (Ghatkesar -> Medchal-Malkajgiri, Telangana) while preserving explicit location.
3. Missing market records without fabricated prices.
4. Valid nearby-mandi results with distance ranking and clear labeling.
5. Strict geographic filtering preventing cross-state leakage.
6. WhatsApp response content assembly without unrelated weather hallucinations.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import httpx

from src.market.schemas import MarketPriceQueryResponse, MarketPriceResponse
from src.market.agmarknet_client import AgmarknetClient, _redact_secrets
from src.market.service import (
    MarketService,
    extract_market_explicit_location,
    normalize_district_name,
    infer_state_from_district,
    get_distance_between_locations,
    _is_pure_price_query,
    _clean_ai_response_for_market_enrichment,
)
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.ai.formatting import decompose_assembled_response, format_multi_intent_response


# ==============================================================================
# 1. Agmarknet Connection Failure & Secrets Redaction Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_agmarknet_connection_failure_bounded_retries_no_crash():
    """Verify Agmarknet client handles transport connect errors safely with bounded retries."""
    client = AgmarknetClient(
        api_key="TEST_SECRET_API_KEY_12345",
        api_url="https://api.data.gov.in/resource/test",
        timeout_seconds=0.5,
        max_retries=1,
        retry_delay_seconds=0.01,
    )

    # Simulate transport-level ConnectError ("All connection attempts failed")
    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("All connection attempts failed.")

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = await client.fetch_prices(
            commodity="Cotton",
            district="Ghatkesar",
            state=None,
        )

    # Client must never raise and must return empty list on connection failure
    assert result == []
    # 1 initial attempt + 1 bounded retry = 2 calls total
    assert call_count == 2


def test_agmarknet_secret_redaction_in_logs():
    """Verify that api keys are masked and never exposed in logs or exception representations."""
    raw_error = "ConnectError: Failed to connect to https://api.data.gov.in/resource?api-key=SUPER_SECRET_KEY_999&limit=10"
    cleaned = _redact_secrets(raw_error)
    assert "SUPER_SECRET_KEY_999" not in cleaned
    assert "***REDACTED***" in cleaned


# ==============================================================================
# 2. District Normalization & Location Extraction Tests
# ==============================================================================

def test_ghatkesar_extraction_and_normalization():
    """Verify Ghatkesar is correctly extracted and mapped to Medchal-Malkajgiri, Telangana."""
    query = "Ghatkesar lo cotton mandi price entha?"
    extracted = extract_market_explicit_location(query)
    assert extracted == "Ghatkesar"

    normalized_district = normalize_district_name(extracted)
    assert normalized_district == "Medchal-Malkajgiri"

    inferred_state = infer_state_from_district(normalized_district)
    assert inferred_state == "Telangana"

    # Telugu script variants
    extracted_te = extract_market_explicit_location("ఘట్‌కేసర్‌లో పత్తి ధర ఎంత?")
    assert extracted_te in ("Ghatkesar", "ఘట్‌కేసర్", "ఘట్కేసర్")
    norm_te = normalize_district_name(extracted_te)
    assert norm_te == "Medchal-Malkajgiri"


# ==============================================================================
# 3. Missing Market Records & Non-Fabrication Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_missing_market_records_does_not_fabricate_prices():
    """When a crop has zero records anywhere in DB or API, ensure no prices are invented."""
    repo = MagicMock()
    repo.get_prices_by_commodity = AsyncMock(return_value=[])
    client = MagicMock()
    client.fetch_prices = AsyncMock(return_value=[])

    svc = MarketService(repository=repo, client=client)
    res = await svc.get_prices_for_query(
        commodity="Dragonfruit",
        district="Medchal-Malkajgiri",
        state="Telangana",
        explicit_location="Ghatkesar",
    )

    assert res.data_available is False
    assert res.results == []
    assert res.is_nearby is False

    reply_te = svc.format_whatsapp_reply(res, language="te")
    # Must inform clearly that data is not available without inventing numbers
    assert "అందుబాటులో లేదు" in reply_te
    assert "₹" not in reply_te


# ==============================================================================
# 4. Nearby Mandi Fallback & Distance Ranking Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_nearby_mandi_fallback_for_ghatkesar_cotton():
    """
    Ghatkesar / Medchal-Malkajgiri has no direct cotton mandi.
    The service must find the closest valid Telangana cotton mandi (Warangal Mandi)
    and clearly label it as a nearby market price.
    """
    from uuid import uuid4
    now = datetime(2026, 10, 16, 10, 0, 0)
    # Mandis in dataset: Warangal (~115km), Khammam (~160km), Adilabad (~250km)
    warangal_record = MagicMock(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        min_price=7100.0,
        max_price=7650.0,
        modal_price=7450.0,
        unit="Quintal",
        price_date=now,
        source="manual_seed",
        created_at=now,
        updated_at=now,
    )
    khammam_record = MagicMock(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Khammam Mandi",
        district="Khammam",
        state="Telangana",
        min_price=7050.0,
        max_price=7600.0,
        modal_price=7400.0,
        unit="Quintal",
        price_date=now,
        source="manual_seed",
        created_at=now,
        updated_at=now,
    )

    repo = MagicMock()
    # When queried with district="Medchal-Malkajgiri", returns state records including Warangal & Khammam
    repo.get_prices_by_commodity = AsyncMock(return_value=[khammam_record, warangal_record])
    client = MagicMock()
    client.fetch_prices = AsyncMock(return_value=[])

    svc = MarketService(repository=repo, client=client)
    res = await svc.get_prices_for_query(
        commodity="Cotton",
        district="Medchal-Malkajgiri",
        state="Telangana",
        explicit_location="Ghatkesar",
    )

    assert res.data_available is True
    assert res.is_nearby is True
    assert res.explicit_location == "Ghatkesar"
    assert len(res.results) > 0
    # Warangal is closer to Ghatkesar than Khammam
    assert res.results[0].market_name == "Warangal Mandi"
    assert res.results[0].modal_price == 7450.0

    # Test Telugu formatting
    reply_te = svc.format_whatsapp_reply(res, language="te")
    assert "📍 Ghatkesar వద్ద నేరుగా" in reply_te or "📍 ఘట్‌కేసర్ వద్ద నేరుగా" in reply_te
    assert "సమీప మార్కెట్లలో ధరలు" in reply_te
    assert "Warangal Mandi" in reply_te
    assert "7,450" in reply_te

    # Test English formatting
    reply_en = svc.format_whatsapp_reply(res, language="en")
    assert "📍 No direct Cotton mandi at Ghatkesar." in reply_en
    assert "Nearby market prices in Telangana:" in reply_en
    assert "Warangal Mandi" in reply_en


# ==============================================================================
# 5. Strict Geographic Filtering (No Cross-State Leakage)
# ==============================================================================

@pytest.mark.asyncio
async def test_strict_geographic_filtering_rejects_cross_state():
    """Out-of-state records (e.g. Andhra Pradesh, Maharashtra) must NEVER be returned as Telangana nearby prices."""
    now = datetime(2026, 10, 16, 10, 0, 0)
    guntur_record = MagicMock(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Guntur Mandi",
        district="Guntur",
        state="Andhra Pradesh",
        min_price=7200.0,
        max_price=7800.0,
        modal_price=7500.0,
        unit="Quintal",
        price_date=now,
        source="manual_seed",
    )

    repo = MagicMock()
    # Repository only has Andhra Pradesh records
    repo.get_prices_by_commodity = AsyncMock(return_value=[guntur_record])
    client = MagicMock()
    client.fetch_prices = AsyncMock(return_value=[])

    svc = MarketService(repository=repo, client=client)
    res = await svc.get_prices_for_query(
        commodity="Cotton",
        district="Medchal-Malkajgiri",
        state="Telangana",
        explicit_location="Ghatkesar",
    )

    # Must NOT substitute Andhra Pradesh record for Telangana Ghatkesar query
    assert res.data_available is False
    assert res.results == []


# ==============================================================================
# 6. Response Assembly & Prevention of Unrelated Weather Text
# ==============================================================================

def test_pure_mandi_query_detection():
    """Verify pure mandi queries bypass general LLM generation."""
    query = "Ghatkesar lo cotton mandi price entha?"
    assert _is_pure_price_query(query) is True

    # Decision engine intent detection
    intents = AIDecisionEngine.detect_all_intents(query)
    assert intents == [FarmerIntent.MARKET_PRICE]


def test_clean_ai_response_removes_speculative_weather():
    """Verify _clean_ai_response_for_market_enrichment strips weather text from AI preamble."""
    ai_text = "Ghatkesar weather is 32C and sunny with high humidity. Cotton is grown in Telangana."
    cleaned = _clean_ai_response_for_market_enrichment(ai_text)
    assert "weather" not in cleaned.lower()
    assert "32C" not in cleaned


def test_decompose_response_handles_location_pin():
    """Verify decompose_assembled_response categorizes chunks starting with 📍 as market section."""
    assembled = "📍 Ghatkesar కోసం ప్రస్తుతం Cotton market-price data అందుబాటులో లేదు."
    sections = decompose_assembled_response(assembled)
    assert sections["market"] == assembled
    assert sections["crop_advice"] == ""


# ==============================================================================
# 7. Webhook Deduplication, Redis Atomic Locks, and DB Uniqueness Safeguards
# ==============================================================================

@pytest.mark.asyncio
async def test_redis_backed_deduplication_multi_worker():
    """Verify atomic Redis SET NX blocks duplicate message IDs across workers."""
    from src.gateway.router import _is_duplicate_or_record_message
    from unittest.mock import AsyncMock

    mock_redis = AsyncMock()
    # First call acquires lock (returns True)
    # Second call fails to acquire lock (returns None/False -> duplicate detected)
    mock_redis.set = AsyncMock(side_effect=[True, None])
    mock_redis.aclose = AsyncMock()

    with patch("src.gateway.router.get_settings") as mock_settings, \
         patch("redis.asyncio.from_url", return_value=mock_redis):
        mock_settings.return_value.redis_url = "redis://localhost:6379/0"

        # First worker receives message
        is_dup1 = await _is_duplicate_or_record_message("wamid.CONCURRENT_REDIS_1")
        assert is_dup1 is False  # Not a duplicate, queued successfully

        # Second worker receives identical message concurrently
        is_dup2 = await _is_duplicate_or_record_message("wamid.CONCURRENT_REDIS_1")
        assert is_dup2 is True  # Duplicate detected across workers!


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_in_memory_deduplication():
    """Verify router gracefully falls back to process in-memory cache if Redis is down."""
    from src.gateway.router import _is_duplicate_or_record_message, _RECENT_QUEUED_MESSAGE_IDS

    _RECENT_QUEUED_MESSAGE_IDS.clear()

    with patch("src.gateway.router.get_settings") as mock_settings, \
         patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis unreachable")):
        mock_settings.return_value.redis_url = "redis://broken-host:6379/0"

        # First request should record in-memory
        is_dup1 = await _is_duplicate_or_record_message("wamid.REDIS_FALLBACK_1")
        assert is_dup1 is False

        # Second request with same ID should be blocked by in-memory cache
        is_dup2 = await _is_duplicate_or_record_message("wamid.REDIS_FALLBACK_1")
        assert is_dup2 is True


@pytest.mark.asyncio
async def test_concurrent_duplicate_webhook_router_queues_once():
    """Verify concurrent webhook POSTs with identical message IDs queue exactly one background task."""
    from src.gateway.router import receive_message, _RECENT_QUEUED_MESSAGE_IDS
    from fastapi import BackgroundTasks
    from unittest.mock import MagicMock
    import json

    _RECENT_QUEUED_MESSAGE_IDS.clear()

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.CONCURRENT_POST_TEST",
                                    "timestamp": "1700000000",
                                    "text": {"body": "Ghatkesar cotton price?"},
                                    "type": "text",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    raw_bytes = json.dumps(payload).encode("utf-8")

    class MockRequest:
        def __init__(self, body):
            self._body = body
            self.client = MagicMock(host="127.0.0.1")
            self.headers = {"content-type": "application/json"}
        async def body(self):
            return self._body

    bg1 = BackgroundTasks()
    bg2 = BackgroundTasks()

    with patch("src.gateway.router.verify_webhook_signature", new_callable=AsyncMock, return_value=True), \
         patch("src.gateway.router.get_settings") as mock_settings:
        mock_settings.return_value.redis_url = None

        res1 = await receive_message(MockRequest(raw_bytes), bg1)
        res2 = await receive_message(MockRequest(raw_bytes), bg2)

    assert res1["status"] == "ok"
    assert res1["messages_queued"] == 1
    assert len(bg1.tasks) == 1

    assert res2["status"] == "ok"
    assert res2["messages_queued"] == 0
    assert len(bg2.tasks) == 0


@pytest.mark.asyncio
async def test_service_pipeline_db_unique_constraint_safeguard():
    """
    Verify that if two workers race into process_message_pipeline concurrently,
    the DB unique constraint in Stage 3 catches the duplicate and cleanly rolls back
    without generating an AI response or sending a duplicate outbound WhatsApp message.
    """
    import uuid
    from src.gateway.service import process_message_pipeline
    from src.core.models import Farmer
    from src.gateway.schemas import ParsedIncomingMessage

    parsed = ParsedIncomingMessage(
        phone_number="919876543210",
        message_id="wamid.DB_RACE_SAFEGUARD_TEST",
        timestamp="1700000000",
        message_type="text",
        text_content="Cotton mandi price",
    )
    mock_farmer = Farmer(id=uuid.uuid4(), phone_number="919876543210")

    with patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=None) as mock_store, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock) as mock_ai, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock) as mock_send:

        # Worker 2 attempts pipeline but store_incoming_message returns None due to IntegrityError
        await process_message_pipeline(parsed)

        mock_store.assert_awaited_once()
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


# ==============================================================================
# 8. Database Schema Compatibility, Migration Safety & Non-Destructive Dedup
# ==============================================================================

def test_conversation_model_has_database_unique_constraint():
    """Verify Conversation.message_id defines database-level unique constraint."""
    from src.core.models import Conversation
    from sqlalchemy import UniqueConstraint

    # Column-level unique attribute
    col = Conversation.__table__.c.message_id
    assert col.unique is True, "Conversation.message_id must have unique=True"

    # Table constraints check
    has_uq = any(
        isinstance(c, UniqueConstraint) and "message_id" in [col.name for col in c.columns]
        for c in Conversation.__table__.constraints
    ) or col.unique is True
    assert has_uq is True


@pytest.mark.asyncio
async def test_database_engine_enforces_message_id_unique_integrity_error():
    """Verify database raises IntegrityError when duplicate message_id is committed."""
    import uuid
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from src.core.models import Base, Farmer, Conversation

    # Create isolated in-memory sqlite engine
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        farmer = Farmer(id=uuid.uuid4(), phone_number="919999999999")
        session.add(farmer)
        await session.commit()

        # Insert first message
        conv1 = Conversation(
            farmer_id=farmer.id,
            message_id="wamid.DB_INTEGRITY_CHECK",
            user_message="First message",
        )
        session.add(conv1)
        await session.commit()

        # Insert duplicate message with same message_id
        conv2 = Conversation(
            farmer_id=farmer.id,
            message_id="wamid.DB_INTEGRITY_CHECK",
            user_message="Duplicate message",
        )
        session.add(conv2)

        with pytest.raises(IntegrityError):
            await session.commit()

        await session.rollback()

    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_017_disambiguates_duplicates_without_deleting_rows():
    """
    Verify migration 017 safely disambiguates duplicate message IDs
    without deleting any conversation or message records.
    """
    from sqlalchemy import create_engine, text
    import importlib
    import uuid

    # Synchronous in-memory sqlite engine to simulate existing legacy database
    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.begin() as conn:
        # Create legacy table without unique constraint on message_id
        conn.execute(text("""
            CREATE TABLE conversations (
                id VARCHAR(36) PRIMARY KEY,
                farmer_id VARCHAR(36),
                message_id VARCHAR(100),
                user_message TEXT,
                created_at TIMESTAMP
            )
        """))
        # Insert duplicate legacy records
        id1, id2, id3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        conn.execute(text("""
            INSERT INTO conversations (id, farmer_id, message_id, user_message, created_at)
            VALUES 
                (:id1, 'farmer1', 'wamid.LEGACY_DUP', 'Message 1', '2026-01-01 10:00:00'),
                (:id2, 'farmer1', 'wamid.LEGACY_DUP', 'Message 2', '2026-01-01 10:05:00'),
                (:id3, 'farmer1', 'wamid.LEGACY_DUP', 'Message 3', '2026-01-01 10:10:00')
        """), {"id1": id1, "id2": id2, "id3": id3})

        # Verify 3 rows initially exist
        initial_count = conn.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
        assert initial_count == 3

    # Import and run migration upgrade logic with mocked alembic op
    mig = importlib.import_module("migrations.versions.017_ensure_conversations_message_id_unique")

    with patch("alembic.op.get_bind", return_value=test_engine), \
         patch("alembic.op.create_index") as mock_create_index:
        mig.upgrade()
        mock_create_index.assert_called_once_with(
            'uq_conversations_message_id',
            'conversations',
            ['message_id'],
            unique=True,
        )

    # Verify results in database:
    # 1. Total row count is still 3 (NO rows were deleted)
    with test_engine.connect() as conn:
        final_count = conn.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
        assert final_count == 3, "Migration must never delete any conversation rows"

        rows = conn.execute(text("SELECT id, message_id FROM conversations ORDER BY created_at ASC")).fetchall()
        # Earliest row preserves original canonical message_id
        assert rows[0][1] == "wamid.LEGACY_DUP"
        # Duplicate rows were safely disambiguated with suffix
        assert "__dup_1_" in rows[1][1]
        assert "__dup_2_" in rows[2][1]

        # Verify all message_id values are now unique
        unique_ids = {r[1] for r in rows}
        assert len(unique_ids) == 3


@pytest.mark.asyncio
async def test_migration_017_idempotent_when_constraint_already_exists():
    """Verify migration 017 is a clean no-op if unique constraint/index already exists."""
    from sqlalchemy import create_engine, text
    import importlib

    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE conversations (
                id VARCHAR(36) PRIMARY KEY,
                message_id VARCHAR(100)
            )
        """))
        conn.execute(text("""
            CREATE UNIQUE INDEX uq_conversations_message_id ON conversations (message_id)
        """))

    mig = importlib.import_module("migrations.versions.017_ensure_conversations_message_id_unique")

    with patch("alembic.op.get_bind", return_value=test_engine), \
         patch("alembic.op.create_index") as mock_create_index:
        mig.upgrade()
        # Index creation must NOT be called since unique constraint already exists
        mock_create_index.assert_not_called()


def test_alembic_revision_identifiers_conform_to_varchar_32_and_single_head():
    """
    Regression test for Render deployment:
    Ensure all Alembic revision identifiers do not exceed the 32-character limit
    of the standard alembic_version.version_num column (VARCHAR(32)), avoiding
    StringDataRightTruncationError on PostgreSQL / asyncpg.
    Also verifies single head and correct down_revision hierarchy.
    """
    import os
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cfg = Config(os.path.join(root_dir, "alembic.ini"))
    alembic_cfg.set_main_option("script_location", os.path.join(root_dir, "migrations"))
    script = ScriptDirectory.from_config(alembic_cfg)

    # 1. Single head
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected exactly 1 migration head, got: {heads}"
    assert heads[0] == "017_msg_id_unique"

    # 2. All revision identifiers <= 32 chars
    revisions = list(script.walk_revisions())
    assert len(revisions) > 0

    seen_revisions = set()
    for rev in revisions:
        assert len(rev.revision) <= 32, (
            f"Revision ID '{rev.revision}' is {len(rev.revision)} characters; "
            f"must be <= 32 characters for alembic_version VARCHAR(32) column."
        )
        assert rev.revision not in seen_revisions, f"Duplicate revision ID found: {rev.revision}"
        seen_revisions.add(rev.revision)

    # 3. Migration 017 properties
    rev_017 = script.get_revision("017_msg_id_unique")
    assert rev_017 is not None
    assert rev_017.down_revision == "016_maps_url_text"



# ==============================================================================
# 9. Pipeline E2E Market Availability, Multi-Intent & Purity Regression Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_market_price_unavailable_response_wording_telugu_and_multilingual():
    """
    Verify market price unavailable notice produces exact localized messages:
    - Telugu: 'ఈ పంటకు మార్కెట్ ధరలు అందుబాటులో లేవు'
    - English: 'Market price information is currently unavailable'
    - Hindi: 'मंडी भाव अभी उपलब्ध नहीं हैं'
    """
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import process_text_message
    from src.ai.schemas import AIGenerateResponse

    # 1. Telugu Multi-Intent with unlisted market crop
    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer_te = Farmer(id=uuid4(), phone_number="919876543210", preferred_language="te")
    conv_te = Conversation(
        id=uuid4(),
        farmer_id=farmer_te.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి. అరటి మార్కెట్ ధర ఎంత?"
    )
    mock_gemini_te = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు తగిన పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini_te), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=[]):

        reply_te = await process_text_message(db_mock, farmer_te, conv_te)
        assert "🌱 *పంట సలహా*" in reply_te
        assert "📊 *మార్కెట్ ధరలు*" in reply_te
        assert "ఈ పంటకు మార్కెట్ ధరలు అందుబాటులో లేవు" in reply_te

    # 2. English Multi-Intent with unlisted market crop
    farmer_en = Farmer(id=uuid4(), phone_number="919876543211", preferred_language="en")
    conv_en = Conversation(
        id=uuid4(),
        farmer_id=farmer_en.id,
        user_message="My cotton crop has pests what spray to use? What is the market price of dragonfruit?"
    )
    mock_gemini_en = AIGenerateResponse(
        response_text="Spray Emamectin Benzoate 5% SG for cotton pest control.",
        intent="crop_advisory",
        confidence=0.9,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini_en), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=[]), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=[]):

        reply_en = await process_text_message(db_mock, farmer_en, conv_en)
        assert "🌱 *Crop Advice*" in reply_en
        assert "📊 *Market Prices*" in reply_en
        assert "Market price information is currently unavailable" in reply_en


@pytest.mark.asyncio
async def test_pure_market_query_contains_no_unrelated_crop_advice_or_weather():
    """
    Verify pure market queries:
    1. Bypasses general LLM generation entirely.
    2. Contain only the structured market price block.
    3. Never append AI-generated crop advice (e.g. chemical dosages, pesticide recommendations).
    4. Never append weather forecast blocks or text.
    """
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import process_text_message
    from tests.test_farmer_pipeline_e2e import make_sample_mandi

    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer_te = Farmer(id=uuid4(), phone_number="919876543210", preferred_language="te")
    conv_te = Conversation(
        id=uuid4(),
        farmer_id=farmer_te.id,
        user_message="వరంగల్లో పత్తి మార్కెట్ ధర ఎంత?"
    )

    # Even if AIService mock exists, it must NOT be invoked for pure price queries
    mock_ai = AsyncMock()

    with patch("src.ai.service.AIService.generate_ai_response", mock_ai), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=make_sample_mandi()):

        reply = await process_text_message(db_mock, farmer_te, conv_te)

        # 1. Market block is present with real data
        assert "పత్తి మార్కెట్ ధరలు" in reply
        assert "Warangal Mandi" in reply
        assert "7,450" in reply

        # 2. General AI generation was bypassed
        mock_ai.assert_not_called()

        # 3. No unrelated crop advisory text or headers
        assert "🌱 *పంట సలహా*" not in reply
        assert "పిచికారీ" not in reply
        assert "పురుగుల" not in reply
        assert "ఎరువు" not in reply

        # 4. No unrelated weather text or headers
        assert "🌡️ *వాతావరణ సమాచారం*" not in reply
        assert "వర్షం" not in reply
        assert "ఉష్ణోగ్రత" not in reply


@pytest.mark.asyncio
async def test_genuine_combined_market_and_advisory_query():
    """
    Verify genuine multi-intent queries explicitly requesting both market prices and crop guidance:
    1. Generates genuine crop advice.
    2. Includes authoritative mandi price information.
    3. Neither block is discarded or corrupted.
    """
    from uuid import uuid4
    from src.core.models import Farmer, Conversation
    from src.ai.service import process_text_message
    from src.ai.schemas import AIGenerateResponse
    from tests.test_farmer_pipeline_e2e import make_sample_mandi

    db_mock = AsyncMock()
    mock_prof = MagicMock()
    mock_prof.scalar_one_or_none.return_value = None
    db_mock.execute = AsyncMock(return_value=mock_prof)

    farmer = Farmer(id=uuid4(), phone_number="919876543210", preferred_language="te")
    conv = Conversation(
        id=uuid4(),
        farmer_id=farmer.id,
        user_message="పత్తి పంటలో పురుగులు వస్తున్నాయి ఏం చేయాలి? మరియు వరంగల్లో పత్తి మార్కెట్ ధర ఎంత?"
    )

    mock_gemini = AIGenerateResponse(
        response_text="పత్తిలో పురుగుల నివారణకు ఎమామెక్టిన్ బెంజోయేట్ 5% SG 0.4 గ్రాములు పిచికారీ చేయండి.",
        intent="crop_advisory",
        confidence=0.95,
        provider_used="gemini",
    )

    with patch("src.ai.service.AIService.generate_ai_response", return_value=mock_gemini), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new_callable=AsyncMock, return_value=None), \
         patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new_callable=AsyncMock), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new_callable=AsyncMock, return_value=make_sample_mandi()):

        reply = await process_text_message(db_mock, farmer, conv)

        # Both sections must be present
        assert "🌱 *పంట సలహా*" in reply
        assert "ఎమామెక్టిన్ బెంజోయేట్" in reply

        assert "📊" in reply
        assert "Warangal Mandi" in reply
        assert "7,450" in reply


def test_response_assembly_preserves_market_sections_accurately():
    """
    Verify response decomposition and multi-intent assembly preserves:
    - Market price table blocks
    - Nearby market indicator blocks (📍)
    - Unavailable warning blocks (⚠️)
    - Unavailable labels (ℹ️)
    Without leaking text into crop advice or dropping sections.
    """
    from src.ai.formatting import decompose_assembled_response, format_multi_intent_response

    # 1. Nearby market block
    raw_assembled = (
        "🌱 *పంట సలహా*\n"
        "పత్తిలో తెగులు నివారణకు సరైన పిచికారీ చేయండి.\n\n"
        "📍 Ghatkesar వద్ద నేరుగా పత్తి మార్కెట్ అందుబాటులో లేదు.\n"
        "సమీప మార్కెట్లలో ధరలు (Telangana):\n"
        "📊 *పత్తి మార్కెట్ ధరలు*\n"
        "• Warangal Mandi: ₹7,450"
    )
    sections = decompose_assembled_response(raw_assembled)
    assert "పత్తిలో తెగులు" in sections["crop_advice"]
    assert "Ghatkesar" in sections["market"]
    assert "Warangal Mandi" in sections["market"]

    # 2. Multi-intent formatting preserves both sections
    user_msg = "పత్తిలో తెగులు వచ్చింది. ఘట్‌కేసర్‌లో పత్తి ధర ఎంత?"
    formatted = format_multi_intent_response(raw_assembled, user_message=user_msg, language="te")
    assert "🌱 *పంట సలహా*" in formatted
    assert "📊" in formatted and "మార్కెట్ ధరలు" in formatted
    assert "Ghatkesar" in formatted
    assert "Warangal Mandi" in formatted



