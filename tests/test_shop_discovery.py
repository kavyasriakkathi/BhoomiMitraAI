"""
BhoomiMitra AI — Automatic Location-Based Shop Discovery Tests

Comprehensive test suite covering all 20 requirements:
1. nearby fertilizer query
2. explicit town query
3. Telugu query
4. English query
5. Romanized Telugu query
6. voice-transcribed query
7. WhatsApp location pin inbound handling
8. stored farmer GPS used for discovery
9. no GPS asking farmer for location
10. external provider failure handling (fail-soft)
11. no external results (fail-soft)
12. discovered shop (Tier C formatting)
13. discovered shop has NO inventory
14. verified RAM FERTILIZER remains priority
15. no hallucinated stock
16. no fake phone
17. no fake owner
18. deduplication of discovered shops against verified shops
19. distance ranking
20. TTS receives exact final response
"""
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import Farmer, Conversation, Shop, Inventory, DiscoveredShop, FarmerProfile
from src.memory.models import FarmerMemory
from src.config import Settings
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import _handle_inbound_location
from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.shops.service import enrich_response_with_shops, _is_explicit_nearby_query
from src.shops.discovery import (
    DiscoveredShopItem,
    MockShopDiscoveryProvider,
    AgriculturalShopFilter,
    ShopDiscoveryOrchestrator,
)


def _make_clean_mock_db():
    mock_db = AsyncMock()
    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = None
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.flush = AsyncMock()
    return mock_db


# ---------------------------------------------------------------------------
# Requirement 1: Nearby Fertilizer Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nearby_fertilizer_query_triggers_discovery():
    """1. Nearby fertilizer query: 'fertilizer shops near me' triggers discovery with coordinates."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_1",
        shop_name="Sri Balaji Fertilizers",
        address="Bus Stand Road, Korutla",
        latitude=18.822,
        longitude=78.712,
        phone_number="+91 9440123456",
        distance_km=1.2,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Sri Balaji Fertilizers" in res
    assert "1.2 km" in res or "కి.మీ" in res
    assert "BhoomiMitra" in res
    assert "నిర్ధారించలేదు" in res or "not verified" in res


# ---------------------------------------------------------------------------
# Requirement 2: Explicit Town Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explicit_town_query_discovers_shops_in_town():
    """2. Explicit town query: 'Korutla lo fertilizer shops unnaya?' resolves town coordinates."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_korutla_1",
        shop_name="Kisan Agro Center",
        address="Main Road, Korutla",
        latitude=18.825,
        longitude=78.714,
        phone_number="+91 9848112233",
        distance_km=0.8,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Korutla lo fertilizer shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Kisan Agro Center" in res
    assert "Main Road, Korutla" in res


# ---------------------------------------------------------------------------
# Requirement 3: Telugu Script Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telugu_query_returns_telugu_labels():
    """3. Telugu query: 'నా దగ్గర ఎరువుల షాపులు ఎక్కడ ఉన్నాయి?' formatted in Telugu."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_te_1",
        shop_name="రైతు సేవా కేంద్రం",
        address="గాంధీ రోడ్డు, కోరుట్ల",
        latitude=18.820,
        longitude=78.710,
        phone_number="+91 9988776655",
        distance_km=0.5,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="నా దగ్గర ఎరువుల షాపులు ఎక్కడ ఉన్నాయి?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "రైతు సేవా కేంద్రం" in res
    assert "గుర్తించబడిన దుకాణం" in res or "సమీపంలో" in res
    assert "నిర్ధారించలేదు" in res


# ---------------------------------------------------------------------------
# Requirement 4: English Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_english_query_returns_english_labels():
    """4. English query: 'seed shops near me' formatted with English labels."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_en_1",
        shop_name="Green Agro Seeds",
        address="Market Yard, Jagtial",
        latitude=18.795,
        longitude=78.910,
        phone_number="+91 9876543210",
        distance_km=1.5,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="seed shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Green Agro Seeds" in res
    assert "Nearby discovered shop" in res
    assert "Current stock is not verified by BhoomiMitra" in res


# ---------------------------------------------------------------------------
# Requirement 5: Romanized Telugu Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_romanized_telugu_query():
    """5. Romanized Telugu: 'Korutla lo shops unnaya?' correctly discovers shops."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_rom_1",
        shop_name="Rythu Bandhu Agencies",
        address="Bus Stand, Korutla",
        latitude=18.821,
        longitude=78.711,
        distance_km=0.3,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Korutla lo shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Rythu Bandhu Agencies" in res


# ---------------------------------------------------------------------------
# Requirement 6: Voice-Transcribed Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_voice_transcribed_query_flow():
    """6. Voice query transcribed from audio produces deterministic text output."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    # Simulates STT output from Telugu farmer voice note
    stt_transcript = "నా దగ్గర యూరియా షాపులు ఎక్కడ దొరుకుతాయి"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_voice_1",
        shop_name="Bhavani Agro Traders",
        address="Jagtial Road, Korutla",
        latitude=18.823,
        longitude=78.713,
        phone_number="+91 9848011222",
        distance_km=0.9,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text=stt_transcript,
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Bhavani Agro Traders" in res
    assert "0.9" in res


# ---------------------------------------------------------------------------
# Requirement 7: WhatsApp Location Pin Inbound Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_whatsapp_location_pin_inbound_valid():
    """7a. Inbound WhatsApp location pin extracts and saves coordinates to FarmerMemory."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, message_id="wamid.loc1")

    parsed = ParsedIncomingMessage(
        phone_number="919848011234",
        message_id="wamid.loc1",
        timestamp="1600000000",
        message_type="location",
        latitude=18.8245,
        longitude=78.7150,
        location_name="Korutla Bus Stand",
    )

    reply = await _handle_inbound_location(mock_db, farmer, conv, parsed, language="te")

    assert "ధన్యవాదాలు! మీ లొకేషన్ విజయవంతంగా సేవ్ చేయబడింది" in reply
    # Verify FarmerMemory.gps_coordinates updated in DB
    added_objs = [c[0][0] for c in mock_db.add.call_args_list]
    memory_obj = next((o for o in added_objs if isinstance(o, FarmerMemory)), None)
    assert memory_obj is not None
    assert memory_obj.gps_coordinates["latitude"] == 18.8245
    assert memory_obj.gps_coordinates["longitude"] == 78.7150


@pytest.mark.asyncio
async def test_whatsapp_location_pin_inbound_malformed():
    """7b. Malformed coordinates prompt farmer to resend pin."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, message_id="wamid.loc2")

    parsed = ParsedIncomingMessage(
        phone_number="919848011234",
        message_id="wamid.loc2",
        timestamp="1600000000",
        message_type="location",
        latitude=0.0,
        longitude=0.0,
    )

    reply = await _handle_inbound_location(mock_db, farmer, conv, parsed, language="te")
    assert "లొకేషన్ వివరాలు స్పష్టంగా అందలేదు" in reply


# ---------------------------------------------------------------------------
# Requirement 8: Stored Farmer GPS Used for Discovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stored_farmer_gps_used_for_discovery():
    """8. Subsequent 'fertilizer shops near me' uses stored GPS from FarmerMemory."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # Mock FarmerMemory returning coordinates for farmer
    mock_memory = FarmerMemory(farmer_id=farmer.id)
    mock_memory.gps_coordinates = {"latitude": 18.825, "longitude": 78.715}

    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = mock_memory
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_gps_1",
        shop_name="Annapurna Agro Agency",
        address="Korutla",
        latitude=18.826,
        longitude=78.716,
        phone_number="+91 9000011111",
        distance_km=0.15,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Annapurna Agro Agency" in res
    assert "0.15" in res or "0.2" in res


# ---------------------------------------------------------------------------
# Requirement 9: No GPS / No Location Asks Farmer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_location_asks_farmer_for_location():
    """9. Explicit 'near me' query with no GPS and no district asks farmer for location."""
    mock_db = _make_clean_mock_db()
    # Farmer with no profile district and no memory GPS
    farmer = Farmer(id=uuid4(), preferred_language="te")

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="నా దగ్గర ఎరువుల షాపులు ఎక్కడ ఉన్నాయి?",
        ai_response="",
        farmer=farmer,
    )

    assert "లొకేషన్ పిన్‌ను షేర్ చేయండి" in res or "గ్రామం/మండలం పేరును టైప్ చేయండి" in res


# ---------------------------------------------------------------------------
# Requirement 10: External Provider Failure (Fail-Soft)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_external_provider_failure_fail_soft():
    """10. If external discovery provider raises or fails, fail-soft and do not crash."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    # Provider configured to fail
    failing_provider = MockShopDiscoveryProvider(should_fail=True)
    orchestrator = ShopDiscoveryOrchestrator(provider=failing_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="నా దగ్గర ఎరువుల షాపులు ఎక్కడ ఉన్నాయి?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    # Returns standard dealer notice without crashing
    assert "నమోదిత డీలర్లు అందుబాటులో లేరు" in res or "అందుబాటులో లేరు" in res


# ---------------------------------------------------------------------------
# Requirement 11: No External Results (Fail-Soft)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_external_results():
    """11. When provider returns zero results, return graceful fallback."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    empty_provider = MockShopDiscoveryProvider(injected_shops=[])
    orchestrator = ShopDiscoveryOrchestrator(provider=empty_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "No licensed dealer is currently registered" in res


# ---------------------------------------------------------------------------
# Requirement 12 & 13: Discovered Shop Formatting & NO Inventory
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_discovered_shop_has_no_inventory():
    """12 & 13. Discovered shop displayed with Tier C badge, address, and ZERO inventory/price."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_disc_1",
        shop_name="Shiva Shakti Agro Center",
        address="Near RTC Bus Stand, Korutla",
        latitude=18.821,
        longitude=78.712,
        phone_number="+91 9440556677",
        distance_km=1.1,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Shiva Shakti Agro Center" in res
    assert "1.1 km" in res
    assert "Near RTC Bus Stand, Korutla" in res
    assert "+91 9440556677" in res

    # STRICT INVENTORY ABSENCE CHECK
    for forbidden_marker in ["₹", "Bag", "బస్తా", "స్టాక్ లభ్యత", "కేజీ", "kg", "ధర:", "Price:"]:
        assert forbidden_marker not in res


# ---------------------------------------------------------------------------
# Requirement 14: Verified RAM FERTILIZER Remains Priority
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verified_ram_fertilizer_remains_priority():
    """14. Verified shop (RAM FERTILIZER) is ranked Tier B above Discovered Shop (Tier C)."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    ram_shop = Shop(
        id=uuid4(),
        shop_name="RAM FERTILIZER",
        owner_name="RAM FERTILIZER",
        phone_number="7989271932",
        address="Beside Balaji Book Seller, Srinivasa Road, Korutla",
        district="Jagtial",
        latitude=18.820,
        longitude=78.710,
        status="active",
        delivery_available=False,
    )

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_tier_c",
        shop_name="New Rural Agri Store",
        address="Main Road, Korutla",
        latitude=18.825,
        longitude=78.715,
        distance_km=0.8,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    with patch("src.shops.repository.ShopRepository.search_by_location", new_callable=AsyncMock, return_value=[ram_shop]), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="Korutla lo fertilizer shops unnaya?",
            ai_response="",
            farmer=farmer,
            discovery_orchestrator=orchestrator,
        )

        assert "RAM FERTILIZER" in res
        assert "New Rural Agri Store" in res
        # Verified shop appears BEFORE discovered shop in the formatted output
        ram_pos = res.find("RAM FERTILIZER")
        disc_pos = res.find("New Rural Agri Store")
        assert ram_pos < disc_pos


# ---------------------------------------------------------------------------
# Requirement 15: No Hallucinated Stock for Pure Stock Query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_hallucinated_stock_pure_stock_query():
    """15. Pure stock query: 'Korutla lo urea undha?' NEVER assigns stock to discovered shop."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_stock_disc",
        shop_name="Sri Balaji Fertilizers",
        address="Korutla",
        latitude=18.821,
        longitude=78.712,
        distance_km=0.5,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    with patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):
        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="Korutla lo urea undha?",
            ai_response="",
            farmer=farmer,
            discovery_orchestrator=orchestrator,
        )

        # Must return the verified stock unavailable notice, NOT claim discovered shop has urea
        assert "లైవ్ స్టాక్ సమాచారం అందుబాటులో లేదు" in res or "అందుబాటులో లేరు" in res
        assert "యూరియా స్టాక్ ఉంది" not in res
        assert "₹295" not in res


# ---------------------------------------------------------------------------
# Requirement 16: No Fake Phone
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_fake_phone_when_provider_phone_is_null():
    """16. When discovery provider has no phone number, phone is omitted (NEVER fabricated)."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_no_phone",
        shop_name="Unlisted Contact Agro Center",
        address="Jagtial",
        latitude=18.795,
        longitude=78.910,
        phone_number=None,  # Null phone
        distance_km=1.0,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Unlisted Contact Agro Center" in res
    assert "Contact:" not in res and "📞" not in res


# ---------------------------------------------------------------------------
# Requirement 17: No Fake Owner
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_fake_owner_for_discovered_shops():
    """17. Owner name is NEVER shown or fabricated for discovered shops."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_no_owner",
        shop_name="Public Agro Outlet",
        address="Main Road, Jagtial",
        latitude=18.795,
        longitude=78.910,
        distance_km=1.0,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[discovered_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert "Public Agro Outlet" in res
    assert "Owner:" not in res
    assert "owner" not in res.lower()


# ---------------------------------------------------------------------------
# Requirement 18: Deduplication Against Verified Shops
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deduplication_of_discovered_shops():
    """18. Discovered shop with same phone or name as verified shop is deduplicated."""
    mock_db = _make_clean_mock_db()

    verified_shop = Shop(
        id=uuid4(),
        shop_name="RAM FERTILIZER",
        owner_name="RAM FERTILIZER",
        phone_number="7989271932",
        address="Beside Balaji, Korutla",
        district="Jagtial",
        latitude=18.820,
        longitude=78.710,
        status="active",
    )

    # Discovered shop matching the verified shop phone
    duplicate_disc_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_dup_1",
        shop_name="Ram Fertilizer Store",
        address="Balaji Road, Korutla",
        latitude=18.8205,
        longitude=78.7102,
        phone_number="+91 7989271932",  # Same phone
    )

    mock_provider = MockShopDiscoveryProvider(injected_shops=[duplicate_disc_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    candidates = await orchestrator.discover_nearby_agricultural_shops(
        db=mock_db,
        latitude=18.820,
        longitude=78.710,
        verified_shops=[verified_shop],
    )

    # Duplicate was dropped
    assert len(candidates) == 0


# ---------------------------------------------------------------------------
# Requirement 19: Distance Ranking
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_distance_ranking():
    """19. Discovered shops are sorted in ascending order of distance."""
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    far_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_far",
        shop_name="Far Fertilizer Store",
        address="Outer Ring Road",
        latitude=18.850,
        longitude=78.750,
        distance_km=8.5,
    )
    near_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="mock_near",
        shop_name="Near Fertilizer Store",
        address="Town Center",
        latitude=18.821,
        longitude=78.711,
        distance_km=0.4,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[far_shop, near_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer shops near me",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    near_pos = res.find("Near Fertilizer Store")
    far_pos = res.find("Far Fertilizer Store")
    assert near_pos != -1
    assert far_pos != -1
    assert near_pos < far_pos


# ---------------------------------------------------------------------------
# Requirement 20: TTS Receives Exact Same Final Response
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_receives_exact_final_response():
    """20. The final deterministic ai_response is passed unchanged to TTS service."""
    from src.gateway.service import process_message_pipeline

    mock_db = _make_clean_mock_db()
    mock_db_cm = MagicMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None

    farmer = Farmer(id=uuid4(), phone_number="919848011239", preferred_language="te")
    conv = Conversation(id=uuid4(), farmer_id=farmer.id, message_id="wamid.tts_test")

    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = farmer
    mock_db.execute = AsyncMock(return_value=exec_res)

    parsed = ParsedIncomingMessage(
        phone_number="919848011239",
        message_id="wamid.tts_test",
        timestamp="1600000000",
        message_type="text",
        text_content="Korutla lo shops unnaya?",
    )

    expected_shop_response = (
        "🏬 సమీప వ్యవసాయ దుకాణాలు & లభ్యత:\n\n"
        "• *RAM FERTILIZER* (1.2 km away)\n"
        "  📦 ఉత్పత్తి: Urea (KRIBHCO)\n"
        "  💰 ధర: ₹266.5/bag | స్టాక్ అందుబాటులో ఉంది (50 bags)\n"
        "  🕒 ధృవీకరించిన సమయం: 04-10-2026\n"
        "  📞 సంప్రదించండి: 9440123456 | खुला है\n"
        "  🚚 డెలివరీ: అందుబాటులో ఉంది"
    )

    mock_tts = AsyncMock(return_value=[b"OggOpusMockAudioBytes"])
    mock_send_text = AsyncMock(return_value="outbound_msg_123")

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=conv), \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_shop_response), \
         patch("src.gateway.service.send_text_message", mock_send_text), \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="media_id_123"), \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="audio_msg_123"), \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc_getter:

        lang_svc = MagicMock()
        lang_svc.synthesize_speech = mock_tts
        mock_lang_svc_getter.return_value = lang_svc

        await process_message_pipeline(parsed)

        # 1. Verify text was sent with exact deterministic shop response
        mock_send_text.assert_awaited_once_with(
            to_phone="919848011239",
            message_text=expected_shop_response,
        )

        # 2. Verify TTS was called with the EXACT SAME text string and active language
        mock_tts.assert_awaited_once_with(
            expected_shop_response,
            "te",
        )


# ===========================================================================
# Configuration & Provider Selection Safety Tests (Fail-Closed)
# ===========================================================================

def test_config_discovery_disabled_when_enabled_is_false():
    """8a. enabled=false -> discovery disabled; cannot instantiate active provider."""
    from src.shops.discovery.factory import get_discovery_provider
    from src.shops.discovery.service import ShopDiscoveryOrchestrator

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = False
    mock_settings.shop_discovery_provider = ""

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings), \
         patch("src.shops.discovery.service.get_settings", return_value=mock_settings):
        # Direct factory call fails closed when discovery is disabled
        with pytest.raises(RuntimeError) as exc_info:
            get_discovery_provider()
        assert "SHOP_DISCOVERY_ENABLED=false" in str(exc_info.value)

        # Orchestrator safely initializes with provider=None and does not make external/mock calls
        orchestrator = ShopDiscoveryOrchestrator()
        assert orchestrator.provider is None


def test_config_google_places_selected_when_explicitly_configured():
    """8b. enabled=true + provider=google_places -> Google provider selected."""
    from src.shops.discovery.factory import get_discovery_provider
    from src.shops.discovery.google_places_provider import GooglePlacesDiscoveryProvider

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = True
    mock_settings.shop_discovery_provider = "google_places"
    mock_settings.google_places_api_key = "AIzaFakeTestKeyForUnitTests"
    mock_settings.shop_discovery_api_timeout_seconds = 5.0
    mock_settings.is_production = False

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings):
        provider = get_discovery_provider()
        assert isinstance(provider, GooglePlacesDiscoveryProvider)
        assert provider.api_key == "AIzaFakeTestKeyForUnitTests"


def test_config_mock_selected_only_when_explicitly_configured():
    """8c. enabled=true + provider=mock -> mock provider selected in non-production."""
    from src.shops.discovery.factory import get_discovery_provider
    from src.shops.discovery.mock_provider import MockShopDiscoveryProvider

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = True
    mock_settings.shop_discovery_provider = "mock"
    mock_settings.is_production = False

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings):
        provider = get_discovery_provider()
        assert isinstance(provider, MockShopDiscoveryProvider)


def test_config_missing_provider_fails_closed_never_selects_mock():
    """8d. enabled=true + provider missing -> mock MUST NOT be selected; fails closed."""
    from src.shops.discovery.factory import get_discovery_provider
    from src.shops.discovery.mock_provider import MockShopDiscoveryProvider

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = True
    mock_settings.shop_discovery_provider = ""
    mock_settings.is_production = False

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings):
        with pytest.raises(ValueError) as exc_info:
            provider = get_discovery_provider()
            # If execution reaches here, it failed the security audit
            assert not isinstance(provider, MockShopDiscoveryProvider)

        assert "SHOP_DISCOVERY_PROVIDER is missing or not configured" in str(exc_info.value)
        assert "Cannot fall back to mock" in str(exc_info.value)


def test_config_invalid_provider_fails_closed_never_selects_mock():
    """8e. enabled=true + invalid provider -> mock MUST NOT be selected; fails closed."""
    from src.shops.discovery.factory import get_discovery_provider
    from src.shops.discovery.mock_provider import MockShopDiscoveryProvider

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = True
    mock_settings.shop_discovery_provider = "random_unsupported_provider"
    mock_settings.is_production = False

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings):
        with pytest.raises(ValueError) as exc_info:
            provider = get_discovery_provider()
            assert not isinstance(provider, MockShopDiscoveryProvider)

        assert "Invalid SHOP_DISCOVERY_PROVIDER" in str(exc_info.value)
        assert "Supported providers are 'google_places' and 'mock'" in str(exc_info.value)


def test_config_mock_strictly_forbidden_in_production():
    """8f. enabled=true + provider=mock in production -> strictly rejected."""
    from src.shops.discovery.factory import get_discovery_provider

    mock_settings = MagicMock()
    mock_settings.shop_discovery_enabled = True
    mock_settings.shop_discovery_provider = "mock"
    mock_settings.is_production = True

    with patch("src.shops.discovery.factory.get_settings", return_value=mock_settings):
        with pytest.raises(ValueError) as exc_info:
            get_discovery_provider()

        assert "strictly forbidden in production" in str(exc_info.value)


# ====================================================================

# LOCATION PRIORITY REGRESSION TESTS (Requirement G)
# ===========================================================================

@pytest.mark.asyncio
async def test_regression_explicit_narapally_resolves_and_google_places_receives_coordinates():
    """
    Regression 1: Query 'Narapally lo fertilizer shops unnaya?'
    Expected:
    - explicit location = Narapally
    - Google Places receives Narapally coordinates (~17.4059, ~78.6180)
    - stored Korutla coordinates are NOT used
    - 25 km radius preserved
    - results calculated from Narapally location.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    narapally_shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_narapally_1",
        shop_name="Sri Balaji Agro Agencies - Narapally",
        address="Warangal Highway, Narapally, Hyderabad",
        latitude=17.4065,
        longitude=78.6185,
        distance_km=0.1,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[narapally_shop])
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Narapally lo fertilizer shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]
    called_radius = called_kwargs.get("radius_meters") if "radius_meters" in called_kwargs else called_args[2]

    # Coordinates match Narapally, NOT Korutla
    assert abs(called_lat - 17.4059) < 0.05
    assert abs(called_lon - 78.6180) < 0.05
    assert abs(called_lat - 18.82) > 1.0  # Over 100km away from Korutla (lat: 18.82)
    assert abs(called_lon - 78.71) > 0.05
    # Preserves 25 km radius
    assert called_radius == 25000

    # Results calculated from Narapally location
    assert "Sri Balaji Agro Agencies - Narapally" in res


@pytest.mark.asyncio
async def test_regression_nearby_query_uses_current_gps():
    """
    Regression 2: Query 'నా దగ్గరలో ఎరువుల షాపులు ఉన్నాయా?'
    Expected:
    - no explicit place in query text
    - uses current/recent GPS if available from FarmerMemory.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    gps_lat, gps_lon = 18.825, 78.715
    mock_memory = FarmerMemory(farmer_id=farmer.id)
    mock_memory.gps_coordinates = {"latitude": gps_lat, "longitude": gps_lon}

    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = mock_memory
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)

    shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_gps_2",
        shop_name="Rythu Bandhu Fertilizer Shop",
        address="Near Bus Stand",
        latitude=18.826,
        longitude=78.716,
        distance_km=0.15,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[shop])
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="నా దగ్గరలో ఎరువుల షాపులు ఉన్నాయా?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]

    # Used the farmer's recent GPS
    assert abs(called_lat - gps_lat) < 0.001
    assert abs(called_lon - gps_lon) < 0.001
    assert "Rythu Bandhu Fertilizer Shop" in res


@pytest.mark.asyncio
async def test_regression_explicit_narapally_overrides_stored_korutla_gps():
    """
    Regression 3: Query with explicit Narapally + stored Korutla GPS
    Expected:
    - Narapally wins!
    - Stored Korutla GPS from FarmerMemory is NOT used.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # Stored Korutla GPS in memory
    mock_memory = FarmerMemory(farmer_id=farmer.id)
    mock_memory.gps_coordinates = {"latitude": 18.820, "longitude": 78.710}

    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = mock_memory
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)

    narapally_shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_narapally_3",
        shop_name="Narapally Agro Agency",
        address="Main Road, Narapally",
        latitude=17.406,
        longitude=78.618,
        distance_km=0.1,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[narapally_shop])
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Narapally lo fertilizer shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]

    # Narapally coordinates won!
    assert abs(called_lat - 17.4059) < 0.05
    assert abs(called_lon - 78.6180) < 0.05
    # Stored Korutla coordinates were ignored
    assert abs(called_lat - 18.820) > 0.5
    assert "Narapally Agro Agency" in res


@pytest.mark.asyncio
async def test_regression_no_explicit_location_no_reliable_gps_asks_location():
    """
    Regression 4: Query with no explicit location and no reliable GPS
    Expected:
    - Ask user to share current WhatsApp location rather than searching Korutla.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # DB has stored profile district 'Korutla', but NO GPS
    mock_profile = FarmerProfile(farmer_id=farmer.id, district="Korutla")
    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = mock_profile
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)

    mock_provider = MockShopDiscoveryProvider()
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="నా దగ్గరలో ఎరువుల షాపులు ఉన్నాయా?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    # Provider was NOT called with Korutla coordinates
    assert not spy_search_nearby.called
    # Farmer prompted to share current WhatsApp location
    assert "లొకేషన్ పిన్‌ను షేర్ చేయండి" in res or "గ్రామం/మండలం పేరును టైప్ చేయండి" in res


@pytest.mark.asyncio
async def test_regression_existing_explicit_korutla_continues_working():
    """
    Regression 5: Existing explicit-location queries such as Korutla must continue working.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    korutla_shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_korutla_5",
        shop_name="Korutla Kisan Traders",
        address="Market Yard, Korutla",
        latitude=18.825,
        longitude=78.714,
        distance_km=0.5,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[korutla_shop])
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Korutla lo fertilizer shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]

    # Korutla coordinates used
    assert abs(called_lat - 18.82) < 0.05
    assert abs(called_lon - 78.71) < 0.05
    assert "Korutla Kisan Traders" in res


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Narapally lo fertilizer shops kavali",
        "fertilizer shops near Narapally",
        "Narapally daggara fertilizer shops",
        "Narapally దగ్గర ఎరువుల షాపులు ఉన్నాయా?",
        "నారపల్లిలో ఎరువుల షాపులు ఉన్నాయా?",
        "నారపల్లి లో ఎరువుల షాపులు ఉన్నాయా?",
    ],
)
async def test_regression_telugu_and_romanized_patterns(query):
    """
    Regression E: Support common Telugu/Romanized patterns for Narapally:
    - 'Narapally lo fertilizer shops kavali'
    - 'fertilizer shops near Narapally'
    - 'Narapally daggara fertilizer shops'
    - 'Narapally దగ్గర ఎరువుల షాపులు ఉన్నాయా?'
    - 'నారపల్లిలో ఎరువుల షాపులు ఉన్నాయా?'
    - 'నారపల్లి లో ఎరువుల షాపులు ఉన్నాయా?'
    All must resolve Narapally coordinates (~17.4059, ~78.6180) and override any stored GPS.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # Stored Korutla GPS in memory must be overridden by all patterns
    mock_memory = FarmerMemory(farmer_id=farmer.id)
    mock_memory.gps_coordinates = {"latitude": 18.820, "longitude": 78.710}

    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = mock_memory
    exec_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=exec_res)

    narapally_shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_narapally_patterns",
        shop_name="Narapally Rythu Seva",
        address="Narapally, Hyderabad",
        latitude=17.406,
        longitude=78.618,
        distance_km=0.2,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[narapally_shop])
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text=query,
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called, f"Failed to detect explicit place in query: {query}"
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]

    assert abs(called_lat - 17.4059) < 0.05, f"Incorrect latitude {called_lat} for query: {query}"
    assert abs(called_lon - 78.6180) < 0.05, f"Incorrect longitude {called_lon} for query: {query}"
    assert "Narapally Rythu Seva" in res


@pytest.mark.asyncio
async def test_regression_arbitrary_unlisted_location_dynamic_geocoding():
    """
    Requirement: DO NOT hardcode Narapally. The resolver must work for arbitrary locations.
    Tests dynamic geocoding for an arbitrary unlisted town via discovery provider.
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="te")

    # Arbitrary town coordinates provided dynamically by provider
    arbitrary_coords = {"ghatkesar": (17.452, 78.685)}
    shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_ghatkesar_1",
        shop_name="Ghatkesar Agro Services",
        address="Ghatkesar Mandal",
        latitude=17.453,
        longitude=78.686,
        distance_km=0.3,
    )
    mock_provider = MockShopDiscoveryProvider(
        injected_shops=[shop],
        custom_geocodes=arbitrary_coords,
    )
    spy_search_nearby = AsyncMock(wraps=mock_provider.search_nearby)
    mock_provider.search_nearby = spy_search_nearby
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    res = await enrich_response_with_shops(
        db=mock_db,
        query_text="Ghatkesar lo fertilizer shops unnaya?",
        ai_response="",
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    assert spy_search_nearby.called
    called_args, called_kwargs = spy_search_nearby.call_args
    called_lat = called_kwargs.get("latitude") if "latitude" in called_kwargs else called_args[0]
    called_lon = called_kwargs.get("longitude") if "longitude" in called_kwargs else called_args[1]

    # Dynamically resolved Ghatkesar coordinates
    assert abs(called_lat - 17.452) < 0.01
    assert abs(called_lon - 78.685) < 0.01
    assert "Ghatkesar Agro Services" in res


@pytest.mark.parametrize(
    "phrase,expected_location",
    [
        ("Narapally lo", "Narapally"),
        ("Narapally lo fertilizer shops", "Narapally"),
        ("Narapally daggara", "Narapally"),
        ("Narapally దగ్గర", "Narapally"),
        ("నారపల్లిలో", "నారపల్లి"),
        ("నారపల్లి లో", "నారపల్లి"),
        ("near Narapally", "Narapally"),
    ],
)
def test_explicit_location_phrase_extraction(phrase, expected_location):
    """
    Direct test for requirement E:
    Ensure _extract_explicit_location_from_query correctly extracts the place name
    from all required Telugu and Romanized pattern variants.
    """
    from src.shops.service import _extract_explicit_location_from_query
    extracted = _extract_explicit_location_from_query(phrase)
    assert extracted is not None, f"Failed to extract location from '{phrase}'"
    assert extracted.lower() == expected_location.lower(), f"Expected '{expected_location}', got '{extracted}' for '{phrase}'"


def test_audit_sanitizer_removes_fabricated_shop_data():
    """
    Audit test: Verify that _sanitize_ai_response_for_shops strips:
    - Hallucinated shop names ('Venkata Sai Fertilizers', 'Sri Rama Agri Inputs')
    - Fake phone numbers ('9849012345', '9849054321')
    - Fake stock claims ('Urea, DAP, NPK available')
    While preserving legitimate agronomic advice.
    """
    from src.shops.service import _sanitize_ai_response_for_shops

    fake_text = (
        "Here are fertilizer shops near Narapally:\n"
        "• Venkata Sai Fertilizers - Phone: 9849012345 (Urea, DAP, NPK available)\n"
        "• Sri Rama Agri Inputs - Phone: 9849054321 (Urea, DAP, NPK available)"
    )
    assert _sanitize_ai_response_for_shops(fake_text) == ""

    mixed_text = (
        "వరి పంటకు ఎకరాకు 50 కేజీల యూరియా వేయాలి.\n"
        "• Venkata Sai Fertilizers - Phone: 9849012345 (Urea, DAP, NPK available)"
    )
    cleaned_mixed = _sanitize_ai_response_for_shops(mixed_text)
    assert "వరి పంటకు ఎకరాకు 50 కేజీల యూరియా వేయాలి." in cleaned_mixed
    assert "Venkata Sai" not in cleaned_mixed
    assert "9849012345" not in cleaned_mixed
    assert "Urea, DAP, NPK available" not in cleaned_mixed


def test_audit_decision_engine_detects_fertilizer_near_by_shops_intent():
    """
    Audit test: Verify that 'fertilizer near by shops' and related variants
    are recognized as SHOPS intent and flagged as pure shop queries,
    bypassing general LLM generation entirely.
    """
    from src.ai.decision_engine import AIDecisionEngine, FarmerIntent

    de = AIDecisionEngine()
    test_queries = [
        "fertilizer near by shops",
        "fertilizer nearby shops",
        "fertilizer shops near me",
        "fertilizer shops near Narapally",
        "Narapally lo fertilizer shops unnaya?",
    ]
    for q in test_queries:
        intents = de.detect_all_intents(q)
        assert FarmerIntent.SHOPS in intents, f"Query '{q}' failed to detect SHOPS intent! Detected: {intents}"


@pytest.mark.asyncio
async def test_audit_enrich_shops_cannot_contain_fabricated_gemini_data():
    """
    Audit test: End-to-end verification that enrich_response_with_shops:
    1. Removes any upstream hallucinated shop names and phone numbers
    2. Resolves Narapally coordinates
    3. Strictly uses discovered shop provider data
    4. Never claims Urea/DAP/NPK stock for discovered shops
    5. Disclaims that live stock is not verified
    6. Does not invent phone numbers
    """
    mock_db = _make_clean_mock_db()
    farmer = Farmer(id=uuid4(), preferred_language="en")

    narapally_shop = DiscoveredShopItem(
        provider="google_places",
        provider_place_id="gp_narapally_audit_1",
        shop_name="Authentic Narapally Agro Center",
        address="Near Main Road, Narapally, Hyderabad, Telangana",
        latitude=17.4060,
        longitude=78.6180,
        phone_number=None,  # No phone in Google Places
        distance_km=0.3,
    )
    mock_provider = MockShopDiscoveryProvider(injected_shops=[narapally_shop])
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    hallucinated_upstream_ai = (
        "Here are fertilizer shops near Narapally:\n"
        "• Venkata Sai Fertilizers - Phone: 9849012345 (Urea, DAP, NPK available)\n"
        "• Sri Rama Agri Inputs - Phone: 9849054321 (Urea, DAP, NPK available)"
    )

    result = await enrich_response_with_shops(
        db=mock_db,
        query_text="fertilizer near by shops near Narapally",
        ai_response=hallucinated_upstream_ai,
        farmer=farmer,
        discovery_orchestrator=orchestrator,
    )

    # 1. Hallucinated shops and phone numbers must be completely absent
    assert "Venkata Sai Fertilizers" not in result
    assert "Sri Rama Agri Inputs" not in result
    assert "9849012345" not in result
    assert "9849054321" not in result

    # 2. Authentic provider shop is present
    assert "Authentic Narapally Agro Center" in result

    # 3. Discovered shop stock must be explicitly disclaimed as unverified
    assert "Current stock is not verified by BhoomiMitra" in result or "స్టాక్ ధృవీకరించబడలేదు" in result

    # 4. Zero fabricated inventory claims for discovered shop
    assert "Urea, DAP, NPK available" not in result
    assert "Urea" not in result
    assert "DAP" not in result
    assert "NPK" not in result

    # 5. Never called "Verified" - labeled as Nearby discovered shop
    assert "Nearby discovered shop" in result or "సమీపంలో కనుగొనబడిన దుకాణం" in result
    assert "BhoomiMitra Verified" not in result



@pytest.mark.asyncio
async def test_discovered_shop_includes_google_maps_directions_link():
    """Discovered shops include a Google Maps directions link."""
    mock_db = _make_clean_mock_db()

    farmer = Farmer(id=uuid4(), preferred_language="en")
    farmer.district = "Jagtial"

    discovered_shop = DiscoveredShopItem(
        provider="mock",
        provider_place_id="maps_test_1",
        shop_name="Maps Test Fertilizers",
        address="Main Road, Jagtial",
        latitude=18.7940,
        longitude=78.9160,
        phone_number="+91 9000000000",
        distance_km=1.0,
    )

    mock_provider = MockShopDiscoveryProvider(
        injected_shops=[discovered_shop]
    )
    orchestrator = ShopDiscoveryOrchestrator(provider=mock_provider)

    with patch(
        "src.shops.service._resolve_farmer_location",
        new=AsyncMock(
            return_value=(18.8000, 78.9200, "Jagtial", "Telangana")
        ),
    ):
        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="fertilizer shops near me",
            ai_response="",
            farmer=farmer,
            discovery_orchestrator=orchestrator,
        )

    assert "Maps Test Fertilizers" in res
    assert "https://www.google.com/maps/dir/?" in res
    assert "destination=18.794%2C78.916" in res
    assert "origin=18.8%2C78.92" in res
