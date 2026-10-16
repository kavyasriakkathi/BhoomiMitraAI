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
