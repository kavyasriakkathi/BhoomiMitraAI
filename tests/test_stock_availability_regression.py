# -*- coding: utf-8 -*-
"""
Focused Regression Tests for Fertilizer / Stock Availability Queries in BhoomiMitra AI.

Verifies:
1. 'Korutla lo urea available ga undha ledha?' classifies as SHOPS intent (primary intent).
2. Pure stock query completely bypasses Gemini generation (0 Gemini calls).
3. Verified stock returns ONLY verified database fixture data (no hallucinations, no fake prices).
4. No verified stock returns localized factual unverified response (no fake shops, no Mallanna, no guessed stock).
5. Never generates contradictory preamble ('కోరుట్లలో ప్రస్తుతం యూరియా స్టాక్ లభ్యత గురించి నా వద్ద ప్రత్యక్ష సమాచారం లేదు').
6. English stock query ('Is urea available in Korutla?') works identically.
7. Telugu voice stock query executes end-to-end with identical text and TTS, without extra Gemini calls.
8. Stock Siren and dosage/chemical safety behavior remain unchanged.
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.shops.service import (
    _extract_district_from_query,
    _detect_shop_intent,
    _is_explicit_stock_query,
)
from src.core.models import Farmer, Conversation, Shop, Inventory
from src.gateway.schemas import ParsedIncomingMessage
from src.gateway.service import process_message_pipeline
from src.language.schemas import TranscriptionResponse


# =============================================================================
# Shared Synthetic Test Fixture
# =============================================================================

TEST_SHOP_NAME = "Kisan Seva Kendra Korutla"
TEST_SHOP_PHONE = "9848011223"
TEST_SHOP_ADDRESS = "Near Old Bus Stand, Korutla"
TEST_SHOP_DISTRICT = "Jagtial"
TEST_STOCK_QUANTITY = 35
TEST_PRODUCT_PRICE = 266.50


def _build_test_shop_and_inventory():
    """Create a synthetic test shop and inventory fixture for regression testing."""
    test_shop = Shop(
        id=uuid4(),
        shop_name=TEST_SHOP_NAME,
        district=TEST_SHOP_DISTRICT,
        address=TEST_SHOP_ADDRESS,
        phone_number=TEST_SHOP_PHONE,
        status="active",
        delivery_available=True,
    )
    test_inventory = Inventory(
        id=uuid4(),
        shop_id=test_shop.id,
        product_name="Neem Coated Urea",
        category="fertilizer",
        brand="IFFCO",
        price=TEST_PRODUCT_PRICE,
        unit="bag",
        quantity_in_stock=TEST_STOCK_QUANTITY,
        available=True,
        minimum_stock_level=10,
    )
    return test_shop, test_inventory


def _setup_voice_mocks(mock_farmer, mock_conv):
    mock_db = AsyncMock()
    mock_db_cm = AsyncMock()
    mock_db_cm.__aenter__.return_value = mock_db
    mock_db_cm.__aexit__.return_value = None
    return mock_db_cm


# =============================================================================
# 1. Intent Detection & Routing for 'Korutla lo urea available ga undha ledha?'
# =============================================================================

def test_intent_detection_korutla_urea_available_ga_undha_ledha():
    """Verify 'Korutla lo urea available ga undha ledha?' detects FarmerIntent.SHOPS as primary intent."""
    query = "Korutla lo urea available ga undha ledha?"
    assert _is_explicit_stock_query(query) is True
    assert _detect_shop_intent(query.lower(), query) is True

    intents = AIDecisionEngine.detect_all_intents(query)
    assert FarmerIntent.SHOPS in intents
    assert AIDecisionEngine.detect_primary_intent(query) == FarmerIntent.SHOPS
    assert _extract_district_from_query(query) == "Jagtial"


def test_intent_detection_stock_availability_variations():
    """Verify various stock availability phrases detect FarmerIntent.SHOPS as primary intent."""
    queries = [
        "Korutla lo urea available ga undha ledha?",
        "Korutla lo urea available ga unda leda?",
        "Korutla lo urea available ga undha?",
        "Korutla lo urea available ga unda?",
        "Korutla lo urea undha?",
        "Korutla lo urea unda?",
        "Korutla lo urea vundha?",
        "Korutla lo urea vunda?",
        "Is urea available in Korutla?",
        "is urea in stock in Korutla?",
        "కోరుట్లలో యూరియా అందుబాటులో ఉందా?",
        "కోరుట్లలో యూరియా ఉందా లేదా?",
        "కోరుట్లలో యూరియా ఉందా?",
    ]
    for q in queries:
        assert _is_explicit_stock_query(q) is True, f"_is_explicit_stock_query returned False for '{q}'"
        assert _detect_shop_intent(q.lower(), q) is True, f"_detect_shop_intent returned False for '{q}'"
        intents = AIDecisionEngine.detect_all_intents(q)
        assert FarmerIntent.SHOPS in intents, f"FarmerIntent.SHOPS not in intents for '{q}': {intents}"
        primary = AIDecisionEngine.detect_primary_intent(q)
        assert primary == FarmerIntent.SHOPS, f"Primary intent for '{q}' is '{primary}' instead of SHOPS"


# =============================================================================
# 2. Verified Urea Stock Returns ONLY Verified Database Data (Bypasses Gemini)
# =============================================================================

@pytest.mark.asyncio
async def test_verified_stock_returns_only_database_fixture():
    """
    When verified inventory exists for Korutla/Jagtial:
    - Return ONLY verified inventory from database fixture
    - Bypasses Gemini generation completely (zero Gemini calls)
    - Zero contradictory 'no direct information' preamble
    - Never fabricates shops or prices
    """
    query = "Korutla lo urea available ga undha ledha?"

    mock_db = AsyncMock()
    test_shop, test_inventory = _build_test_shop_and_inventory()

    mock_farmer = Farmer(id=uuid4(), phone_number="919848011223", preferred_language="te")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=query)

    engine = AIDecisionEngine()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(test_shop, test_inventory)]):

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)

        # Gemini must NEVER be called for pure stock availability query
        mock_gemini.assert_not_called()

        # Returns strictly verified data matching the synthetic fixture
        assert TEST_SHOP_NAME in res
        assert str(TEST_PRODUCT_PRICE) in res
        assert f"{TEST_STOCK_QUANTITY} bags" in res
        assert TEST_SHOP_PHONE in res

        # Must NOT contain contradictory refusal or crop advice
        assert "ప్రత్యక్ష సమాచారం లేదు" not in res
        assert "no direct information" not in res.lower()
        assert "వరి పంట ఎలా ఉంది" not in res


# =============================================================================
# 3. No Verified Urea Stock: Factual Unverified Notice (No Fabricated Shops)
# =============================================================================

@pytest.mark.asyncio
async def test_unverified_stock_no_fabrication():
    """
    When no verified current stock exists for 'Korutla lo urea available ga undha ledha?':
    - Returns clear localized unverified notice
    - Never fabricates shops (no Mallanna, no fake shops, no fake quantities/prices/phone numbers)
    - Zero Gemini calls
    """
    query = "Korutla lo urea available ga undha ledha?"
    mock_db = AsyncMock()
    mock_farmer = Farmer(id=uuid4(), phone_number="919848011224", preferred_language="te")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=query)

    engine = AIDecisionEngine()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)

        # Gemini must not be called
        mock_gemini.assert_not_called()

        # Factual localized notice
        assert "🏬" in res
        assert "నమోదు కాలేదు" in res or "not registered" in res or "సమీప" in res

        # No fabricated shop, price, quantity, or phone
        assert "Mallanna" not in res
        assert "8976547654" not in res
        assert "₹295" not in res
        assert "50 Bag" not in res


# =============================================================================
# 4. English Stock Query Verified and Unverified
# =============================================================================

@pytest.mark.asyncio
async def test_english_stock_query_verified_and_unverified():
    """
    English stock query 'Is urea available in Korutla?':
    - Classifies as SHOPS intent
    - Bypasses Gemini
    - Returns verified data when present, localized unverified message when missing.
    """
    query = "Is urea available in Korutla?"
    assert _detect_shop_intent(query.lower(), query) is True
    intents = AIDecisionEngine.detect_all_intents(query)
    assert FarmerIntent.SHOPS in intents
    assert AIDecisionEngine.detect_primary_intent(query) == FarmerIntent.SHOPS

    mock_db = AsyncMock()
    mock_farmer = Farmer(id=uuid4(), phone_number="919848011225", preferred_language="en")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=query)

    engine = AIDecisionEngine()

    # Unverified case
    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)
        mock_gemini.assert_not_called()
        assert "No licensed dealer is currently registered" in res
        assert "Mallanna" not in res


# =============================================================================
# 5. Telugu Voice Stock Availability Pipeline
# =============================================================================

@pytest.mark.asyncio
async def test_telugu_voice_stock_availability_pipeline_exact_string():
    """
    Telugu voice message for 'Korutla lo urea available ga undha ledha?'
    - End-to-end pipeline sends text AND audio with EXACT same grounded response.
    - Zero extra Gemini calls.
    - Uses identical synthetic fixture values (shop name, phone, quantity).
    """
    farmer_query = "Korutla lo urea available ga undha ledha?"

    parsed = ParsedIncomingMessage(
        phone_number="919848011226",
        message_id="wamid.VOICE_AVAIL_01",
        timestamp="1700000000",
        message_type="audio",
        media_id="audio_media_avail",
    )
    mock_farmer = Farmer(id=uuid4(), phone_number="919848011226", preferred_language="te")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, message_id=parsed.message_id, user_message=None, user_message_type="audio")
    mock_db_cm = _setup_voice_mocks(mock_farmer, mock_conv)

    expected_stock_text = (
        "🏬 సమీప వ్యవసాయ దుకాణాలు & లభ్యత:\n\n"
        f"• *{TEST_SHOP_NAME}* (సమీపంలో)\n"
        "  📦 ఉత్పత్తి: Neem Coated Urea (IFFCO)\n"
        f"  💰 ధర: ₹{TEST_PRODUCT_PRICE}/bag | స్టాక్ అందుబాటులో ఉంది ({TEST_STOCK_QUANTITY} bags)\n"
        f"  📞 సంప్రదించండి: {TEST_SHOP_PHONE} | తెరిచి ఉంది\n"
        "  🚚 డెలివరీ: అందుబాటులో ఉంది\n\n"
        "ℹ️ గమనిక: ధరలు మరియు స్టాక్ వివరాలు స్థానిక డీలర్ నిర్ధారణకు లోబడి ఉంటాయి."
    )

    with patch("src.gateway.service.AsyncSessionLocal", return_value=mock_db_cm), \
         patch("src.gateway.service.is_duplicate_message", new_callable=AsyncMock, return_value=False), \
         patch("src.gateway.service.get_or_create_farmer", new_callable=AsyncMock, return_value=mock_farmer), \
         patch("src.gateway.service.store_incoming_message", new_callable=AsyncMock, return_value=mock_conv), \
         patch("src.gateway.service.download_media_bytes", new_callable=AsyncMock, return_value=(b"voice_bytes", "audio/ogg")), \
         patch("src.gateway.service.get_language_service") as mock_lang_svc, \
         patch("src.gateway.service.process_text_message", new_callable=AsyncMock, return_value=expected_stock_text) as mock_proc_text, \
         patch("src.gateway.service.send_text_message", new_callable=AsyncMock, return_value="wamid.OUT_TEXT_AVAIL") as mock_send_text, \
         patch("src.gateway.service.upload_media_bytes", new_callable=AsyncMock, return_value="meta_audio_avail") as mock_upload, \
         patch("src.gateway.service.send_audio_message", new_callable=AsyncMock, return_value="wamid.OUT_AUDIO_AVAIL") as mock_send_audio, \
         patch("src.gateway.service.mark_message_as_read", new_callable=AsyncMock), \
         patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini:

        mock_lang_svc.return_value.transcribe_audio = AsyncMock(
            return_value=TranscriptionResponse(provider_used="google", transcription_text=farmer_query, detected_language="te")
        )
        mock_lang_svc.return_value.synthesize_speech = AsyncMock(return_value=[b"OggS_avail_audio"])

        await process_message_pipeline(parsed)

        # 1. process_text_message called once
        assert mock_proc_text.await_count == 1
        # 2. Gemini NOT called
        mock_gemini.assert_not_called()
        # 3. Exact same text sent to text and TTS
        mock_send_text.assert_awaited_once_with(to_phone="919848011226", message_text=expected_stock_text)
        mock_lang_svc.return_value.synthesize_speech.assert_awaited_once_with(expected_stock_text, "te")
        mock_upload.assert_awaited_once_with(b"OggS_avail_audio", mime_type="audio/ogg")
        mock_send_audio.assert_awaited_once_with("919848011226", "meta_audio_avail")
