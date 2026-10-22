"""
Regression Test Suite for BhoomiMitra AI Production WhatsApp Test Review.

Covers:
- Issue 1: Cotton multi-mandi comparison across Warangal, Khammam, Adilabad in Telangana
          with historical dates and freshness notice; commodity aliases; clear unavailable handling.
- Issue 2: Pure crop diagnosis does not trigger irrelevant weather or market sections;
          genuine multi-intent query preserves both crop health and weather.
- Issue 3: Romanized Telugu query language detection prefers Telugu;
          preserves safe diagnosis and blocks unverified pesticide dosages;
          explicit English request override is honored.
- Safeguards: District/state geographic fencing and fertilizer safety checks remain intact.
"""

import pytest
from uuid import uuid4
from datetime import datetime

from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.language.detector import detect_language
from src.market.service import (
    COMMODITY_MAP,
    extract_market_locations,
    extract_market_explicit_location,
    enrich_response_with_market_prices,
    MarketService,
)
from src.market.schemas import MarketPriceResponse, MarketPriceQueryResponse
from src.weather.service import enrich_response_with_weather


class MockFarmer:
    def __init__(self, farmer_id=None, preferred_language="te", state="Telangana", district="Warangal"):
        self.id = farmer_id or uuid4()
        self.preferred_language = preferred_language
        self.state = state
        self.district = district
        self.gps_coordinates = None


# ==============================================================================
# Issue 1: Cotton market comparison and historical date disclosures
# ==============================================================================

def test_cotton_commodity_aliases():
    """Verify cotton aliases (cotton lint, raw cotton, lint, etc.) resolve to canonical Cotton."""
    aliases = [
        "cotton", "kapas", "patti", "patthi",
        "cotton lint", "cotton (lint)", "cottonlint",
        "raw cotton", "lint", "cotton seed", "raw kapas",
        "పత్తి", "కాపాస్", "పత్తి లింట్", "దూది",
    ]
    for alias in aliases:
        assert alias in COMMODITY_MAP, f"Alias '{alias}' missing from COMMODITY_MAP"
        assert COMMODITY_MAP[alias] == "Cotton", f"Alias '{alias}' mapped to {COMMODITY_MAP[alias]}, expected Cotton"


def test_extract_market_locations_multi_mandi():
    """Verify state and multiple mandis are extracted without misclassifying state as district."""
    q = "Telangana lo cotton ki Warangal, Khammam, Adilabad mandi prices compare cheyyi. Dates kooda cheppu."
    state, locs = extract_market_locations(q)
    assert state == "Telangana"
    assert "Warangal" in locs
    assert "Khammam" in locs
    assert "Adilabad" in locs
    assert "Telangana" not in locs, "State should not be in mandis/locations list"


def test_format_whatsapp_reply_historical_disclosure():
    """Verify format_whatsapp_reply discloses historical dates and the freshness warning when data is older."""
    svc = MarketService(repository=None, client=None)
    records = [
        MarketPriceResponse(
            id=uuid4(),
            commodity="Cotton",
            commodity_telugu="పత్తి",
            market_name="Warangal Mandi",
            district="Warangal",
            state="Telangana",
            min_price=7100.0,
            max_price=7800.0,
            modal_price=7450.0,
            unit="Quintal",
            price_date=datetime(2026, 10, 17, 10, 0),
            source="manual_seed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        MarketPriceResponse(
            id=uuid4(),
            commodity="Cotton",
            commodity_telugu="పత్తి",
            market_name="Khammam Mandi",
            district="Khammam",
            state="Telangana",
            min_price=7000.0,
            max_price=7750.0,
            modal_price=7400.0,
            unit="Quintal",
            price_date=datetime(2026, 10, 17, 10, 0),
            source="manual_seed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        MarketPriceResponse(
            id=uuid4(),
            commodity="Cotton",
            commodity_telugu="పత్తి",
            market_name="Adilabad Mandi",
            district="Adilabad",
            state="Telangana",
            min_price=6950.0,
            max_price=7700.0,
            modal_price=7350.0,
            unit="Quintal",
            price_date=datetime(2026, 10, 17, 10, 0),
            source="manual_seed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
    ]

    query_response = MarketPriceQueryResponse(
        commodity="Cotton",
        district="Warangal",
        state="Telangana",
        results=records,
        data_available=True,
        data_freshness_hours=120.0,
        source_note="Local database",
        is_live=False,
        is_today_requested=False,
    )

    reply = svc.format_whatsapp_reply(query_response, language="te")
    assert "Warangal Mandi" in reply
    assert "Khammam Mandi" in reply
    assert "Adilabad Mandi" in reply
    assert "17 Oct 2026" in reply
    assert "గమనిక: ఇవి మార్కెట్‌లో లభించిన మునుపటి రికార్డు ధరలు" in reply
    assert "నేటి ధర మార్కెట్ యార్డులో మారవచ్చు" in reply


def test_clear_unavailable_message_for_unknown_mandi():
    """Verify unknown mandi returns clear unavailable message without hallucination."""
    svc = MarketService(repository=None, client=None)
    query_response = MarketPriceQueryResponse(
        commodity="Cotton",
        district="NonExistentPlace",
        state="Telangana",
        results=[],
        data_available=False,
        data_freshness_hours=None,
        source_note="No price data available.",
        is_live=False,
        is_today_requested=False,
        explicit_location="NonExistentPlace",
    )
    reply = svc.format_whatsapp_reply(query_response, language="te")
    assert "మార్కెట్ ధరల సమాచారం అందుబాటులో లేదు" in reply
    assert "NonExistentPlace" in reply


# ==============================================================================
# Issue 2: Crop diagnosis vs weather intent isolation & multi-intent preservation
# ==============================================================================

def test_pure_crop_diagnosis_does_not_trigger_weather():
    """Ensure pure crop disease/pest questions never trigger WEATHER intent."""
    queries = [
        "Naa chilli pantalo aakula meeda spots unnayi. Em cheyyali?",
        "Cotton lo purugu undi photo chudu",
        "వరిలో ఆకుమచ్చ తెగులు ఉంది మందు చెప్పు",
        "Tomato leaf spots treatment and photo",
    ]
    for q in queries:
        intents = AIDecisionEngine.detect_all_intents(q)
        assert FarmerIntent.WEATHER not in intents, f"Weather falsely detected in pure crop query: '{q}'"
        assert FarmerIntent.MARKET_PRICE not in intents, f"Market falsely detected in pure crop query: '{q}'"
        assert FarmerIntent.CROP_HEALTH in intents, f"Crop health should be detected in: '{q}'"


def test_genuine_multi_intent_preserves_both_sections():
    """Ensure genuine multi-intent requests (crop disease + rain forecast) retain both intents."""
    q = "Cotton lo purugu undi, repu varsham paduthunda?"
    intents = AIDecisionEngine.detect_all_intents(q)
    assert FarmerIntent.CROP_HEALTH in intents, "Crop health intent missing from multi-intent query"
    assert FarmerIntent.WEATHER in intents, "Weather intent missing from multi-intent query"


@pytest.mark.asyncio
async def test_enrich_weather_skips_pure_crop_query():
    """Verify enrich_response_with_weather does not append weather block to pure crop queries."""
    farmer = MockFarmer()
    crop_query = "Naa chilli pantalo aakula meeda spots unnayi. Em cheyyali?"
    ai_response = "ఆకులపై మచ్చల లక్షణాలు తెగులు కావచ్చు. దయచేసి పంట ఆకుల స్పష్టమైన ఫోటో పంపండి."
    enriched = await enrich_response_with_weather(None, crop_query, ai_response, farmer)
    assert enriched == ai_response, "Weather should not enrich a pure crop question"
    assert "🌤️" not in enriched
    assert "వాతావరణం" not in enriched


# ==============================================================================
# Issue 3: Telugu language consistency and safe diagnosis
# ==============================================================================

def test_romanized_telugu_language_detection():
    """Verify Romanized Telugu queries detect as 'te' instead of English."""
    q = "Naa chilli pantalo aakula meeda spots unnayi. Em cheyyali?"
    lang = detect_language(q)
    assert lang == "te", f"Expected 'te', got '{lang}'"


def test_explicit_english_override():
    """Verify explicit English instruction overrides language to 'en'."""
    queries = [
        "Naa chilli pantalo spots unnayi. English lo cheppu.",
        "Cotton price compare cheyyi in english",
        "Please reply in English about weather",
    ]
    for q in queries:
        lang = detect_language(q)
        assert lang == "en", f"Expected 'en' override for: '{q}', got '{lang}'"


def test_unconfirmed_diagnosis_rule_preservation():
    """Verify safe diagnosis behavior: unconfirmed diagnosis refuses dosage and asks for photo."""
    from src.ai.prompts import BHOOMIMITRA_SYSTEM_PROMPT
    assert "ఖచ్చితమైన నిర్ధారణ లేకుండా మందు మోతాదును సూచించను" in BHOOMIMITRA_SYSTEM_PROMPT
    assert "దయచేసి పంట ఫోటోను పంపండి లేదా వ్యవసాయ అధికారిని సంప్రదించండి" in BHOOMIMITRA_SYSTEM_PROMPT
    assert "NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages" in BHOOMIMITRA_SYSTEM_PROMPT


# ==============================================================================
# Geographic Safeguards & Fertilizer Safety
# ==============================================================================

def test_geographic_fencing_prevents_cross_state_leakage():
    """Verify Telangana query only resolves to Telangana mandis."""
    q = "Telangana lo cotton ki Warangal, Khammam, Adilabad mandi prices compare cheyyi."
    state, locs = extract_market_locations(q)
    assert state == "Telangana"
    for loc in locs:
        assert loc.lower() in [
            "warangal", "khammam", "adilabad"
        ], f"Unexpected location: {loc}"


def test_fertilizer_safety_inquiry_checks_preserved():
    """Verify fertilizer stage checks require basal history before advising top dressing."""
    from src.ai.service import check_fertilizer_history_details
    result = check_fertilizer_history_details("2 acres paddy 30 days urea entha veyali", history_records=[])
    assert result["has_essential_history"] is False, "Essential history should be missing for cold query"
