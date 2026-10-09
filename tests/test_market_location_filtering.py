"""
Tests for Market Price Generic Location Filtering Bug Fix.

Verifies:
1. Extraction of explicit location from English, Tanglish, and Telugu queries:
   - "Korutla lo cotton price entha?" -> Korutla
   - "Metpally lo cotton price entha?" -> Metpally
   - "Nizamabad lo cotton price entha?" -> Nizamabad
   - "కోరుట్లలో పత్తి ధర ఎంత?" -> Korutla
   - "Jagtial lo patti dhara entha?" -> Jagtial
   - "Sircilla lo patti dhara entha?" -> Sircilla
   - "మెట్పల్లిలో పత్తి ధర ఎంత?" -> Metpally
   - "జగిత్యాలలో పత్తి ధర ఎంత?" -> Jagtial
   - "వరంగల్లో పత్తి ధర ఎంత?" -> Warangal
2. No explicit location when asking generally or about local/my village:
   - "Cotton price entha?" -> None
   - "నా ఊరిలో పత్తి ధర ఎంత?" -> None
   - "Na daggara cotton price entha?" -> None
3. Priority ordering:
   - Priority 1: Explicit query location overrides GPS (GPS = Hyderabad, Query = "Jagtial lo...") -> Jagtial
   - Priority 1: Explicit query location overrides saved profile district (Saved = Korutla, Query = "Warangal lo...") -> Warangal
4. Fallback behavior:
   - When explicit location has NO data: clear "data unavailable" message for that requested location,
     never silently returning Warangal or other unrelated markets.
5. Generic location support:
   - Works for ANY town/village/mandal/district (not hard-coded to a few names).
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest

from src.core.models import Farmer, FarmerMemory, FarmerProfile, MarketPrice
from src.market.service import (
    MarketService,
    enrich_response_with_market_prices,
    extract_market_explicit_location,
)


def _mock_price_model(
    commodity: str = "Cotton",
    market_name: str = "Warangal Mandi",
    district: str = "Warangal",
    state: str = "Telangana",
    modal_price: float = 7500.0,
    min_price: float = 7000.0,
    max_price: float = 8000.0,
    price_date: datetime = None,
    source: str = "agmarknet_api",
) -> MarketPrice:
    if price_date is None:
        price_date = datetime.now()
    p = MagicMock(spec=MarketPrice)
    p.id = uuid4()
    p.commodity = commodity
    p.commodity_telugu = "పత్తి" if commodity == "Cotton" else "వరి"
    p.market_name = market_name
    p.district = district
    p.state = state
    p.min_price = min_price
    p.max_price = max_price
    p.modal_price = modal_price
    p.unit = "Quintal"
    p.price_date = price_date
    p.source = source
    p.created_at = datetime.utcnow()
    p.updated_at = datetime.utcnow()
    return p


# ---------------------------------------------------------------------------
# 1. Location Extraction Unit Tests
# ---------------------------------------------------------------------------

def test_extract_location_english_korutla():
    """1. English village/town: 'Korutla lo cotton price entha?'"""
    loc = extract_market_explicit_location("Korutla lo cotton price entha?")
    assert loc == "Korutla"


def test_extract_location_english_metpally():
    """2. Another location: 'Metpally lo cotton price entha?'"""
    loc = extract_market_explicit_location("Metpally lo cotton price entha?")
    assert loc == "Metpally"


def test_extract_location_english_nizamabad():
    """3. Another district/city: 'Nizamabad lo cotton price entha?'"""
    loc = extract_market_explicit_location("Nizamabad lo cotton price entha?")
    assert loc == "Nizamabad"


def test_extract_location_telugu_korutla():
    """4. Telugu: 'కోరుట్లలో పత్తి ధర ఎంత?' -> Korutla"""
    loc = extract_market_explicit_location("కోరుట్లలో పత్తి ధర ఎంత?")
    assert loc == "Korutla"


def test_extract_location_telugu_other_places():
    """Telugu inflection tests for Metpally, Jagtial, Warangal."""
    assert extract_market_explicit_location("మెట్పల్లిలో పత్తి ధర ఎంత?") == "Metpally"
    assert extract_market_explicit_location("జగిత్యాలలో పత్తి ధర ఎంత?") == "Jagtial"
    assert extract_market_explicit_location("వరంగల్లో పత్తి ధర ఎంత?") == "Warangal"


def test_extract_location_tanglish_jagtial():
    """5. Tanglish: 'Jagtial lo patti dhara entha?' -> Jagtial"""
    loc = extract_market_explicit_location("Jagtial lo patti dhara entha?")
    assert loc == "Jagtial"


def test_extract_location_tanglish_sircilla():
    """Tanglish: 'Sircilla lo patti dhara entha?' -> Sircilla"""
    loc = extract_market_explicit_location("Sircilla lo patti dhara entha?")
    assert loc == "Sircilla"


def test_extract_location_variations():
    """Common variations: at, in, market, mandi, price patterns."""
    assert extract_market_explicit_location("at Korutla cotton price") == "Korutla"
    assert extract_market_explicit_location("in Korutla cotton price") == "Korutla"
    assert extract_market_explicit_location("Korutla market cotton rate") == "Korutla"
    assert extract_market_explicit_location("Korutla mandi cotton price") == "Korutla"
    assert extract_market_explicit_location("Korutla lo cotton rate") == "Korutla"
    assert extract_market_explicit_location("Korutla cotton price") == "Korutla"


def test_generic_unknown_village_extraction():
    """System must extract ANY arbitrary village name without hardcoding."""
    assert extract_market_explicit_location("Thimmapur lo cotton price entha?") == "Thimmapur"
    assert extract_market_explicit_location("Raikal lo cotton price entha?") == "Raikal"
    assert extract_market_explicit_location("Ibrahimpatnam lo paddy rate entha?") == "Ibrahimpatnam"


def test_no_explicit_location_general_queries():
    """Queries without a specific place name must NOT extract a place."""
    assert extract_market_explicit_location("Cotton price entha?") is None
    assert extract_market_explicit_location("నా ఊరిలో పత్తి ధర ఎంత?") is None
    assert extract_market_explicit_location("Na daggara cotton price entha?") is None
    assert extract_market_explicit_location("ఈరోజు పత్తి ధర ఎంత?") is None
    assert extract_market_explicit_location("What is the cotton price today?") is None


# ---------------------------------------------------------------------------
# 2. Priority & Fallback Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_location_overrides_gps():
    """
    6. Explicit location must override GPS:
    GPS = Hyderabad
    Query = 'Jagtial lo cotton price entha?'
    Expected location searched = Jagtial (NOT Hyderabad)
    """
    mock_db = AsyncMock()
    farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="te",
    )
    farmer_mem = FarmerMemory(
        id=uuid4(),
        farmer_id=farmer.id,
        gps_coordinates="17.3850,78.4867",  # Hyderabad GPS
    )
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = farmer_mem
    mock_db.execute.return_value = mock_res

    jagtial_price = _mock_price_model(
        commodity="Cotton",
        market_name="Jagtial Mandi",
        district="Jagtial",
        modal_price=7600.0,
    )

    mock_get_prices = AsyncMock(return_value=[jagtial_price])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="Jagtial lo cotton price entha?",
            ai_response="ధర వివరాలు",
            farmer=farmer,
        )

        # Check repository was called with district="Jagtial", NOT Hyderabad
        call_args = mock_get_prices.call_args
        assert call_args is not None
        assert call_args.kwargs.get("district") == "Jagtial"
        assert "Jagtial Mandi" in result
        assert "7,600" in result


@pytest.mark.asyncio
async def test_explicit_location_overrides_saved_farmer_location():
    """
    7. Explicit location must override saved farmer location:
    Saved profile district = Korutla
    Query = 'Warangal lo cotton price entha?'
    Expected location searched = Warangal (NOT Korutla)
    """
    mock_db = AsyncMock()
    profile = FarmerProfile(
        id=uuid4(),
        farmer_id=uuid4(),
        district="Korutla",
        state="Telangana",
    )
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = profile
    mock_db.execute.return_value = mock_res

    farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="en",
    )

    warangal_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal APMC",
        district="Warangal",
        modal_price=7850.0,
    )

    mock_get_prices = AsyncMock(return_value=[warangal_price])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="Warangal lo cotton price entha?",
            ai_response="Price details:",
            farmer=farmer,
        )

        call_args = mock_get_prices.call_args
        assert call_args is not None
        assert call_args.kwargs.get("district") == "Warangal"
        assert "Warangal APMC" in result
        assert "7,850" in result


@pytest.mark.asyncio
async def test_explicit_location_with_no_data_returns_clear_unavailable_message():
    """
    8. Location with no data:
    User asks: 'SomeVillage lo cotton price entha?'
    Expected: clear 'data unavailable for SomeVillage' response,
    NOT silently returning Warangal or any other market!
    """
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="te",
    )

    mock_get_prices = AsyncMock(return_value=[])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="SomeVillage lo cotton price entha?",
            ai_response="ధర సమాచారం",
            farmer=farmer,
        )

        # Must inform user that SomeVillage data is unavailable
        assert "SomeVillage" in result
        assert "అందుబాటులో లేదు" in result
        # Must NEVER return Warangal or fake market data
        assert "Warangal" not in result
        assert "Enumamula" not in result


@pytest.mark.asyncio
async def test_no_explicit_location_uses_existing_gps_or_saved_behavior():
    """
    9. No explicit location:
    'Cotton price entha?'
    Farmer profile has saved district 'Warangal'.
    Expected = uses saved Warangal location.
    """
    mock_db = AsyncMock()
    profile = FarmerProfile(
        id=uuid4(),
        farmer_id=uuid4(),
        district="Warangal",
        state="Telangana",
    )
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = profile
    mock_db.execute.return_value = mock_res

    farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="te",
    )

    warangal_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal Mandi",
        district="Warangal",
        modal_price=7400.0,
    )

    mock_get_prices = AsyncMock(return_value=[warangal_price])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="Cotton price entha?",
            ai_response="ధర సమాచారం",
            farmer=farmer,
        )

        call_args = mock_get_prices.call_args
        assert call_args is not None
        assert call_args.kwargs.get("district") == "Warangal"
        assert "Warangal Mandi" in result
        assert "7,400" in result


@pytest.mark.asyncio
async def test_explicit_nizamabad_paddy_lookup():
    """
    Verify 'Nizamabad lo paddy price entha?' explicitly looks up:
    commodity='Paddy' and district='Nizamabad'.
    """
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(
        id=uuid4(),
        phone_number="+919876543210",
        preferred_language="en",
    )

    nizamabad_price = _mock_price_model(
        commodity="Paddy",
        market_name="Nizamabad Market",
        district="Nizamabad",
        modal_price=2200.0,
    )

    mock_get_prices = AsyncMock(return_value=[nizamabad_price])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="Nizamabad lo paddy price entha?",
            ai_response="Price info:",
            farmer=farmer,
        )

        call_args = mock_get_prices.call_args
        assert call_args is not None
        assert call_args.kwargs.get("commodity") == "Paddy"
        assert call_args.kwargs.get("district") == "Nizamabad"
        assert "Nizamabad Market" in result
        assert "2,200" in result


@pytest.mark.asyncio
async def test_enrich_market_prices_prevents_greenlet_spawn_error_on_unloaded_relationships():
    """
    Regression Test:
    Simulate a farmer where accessing farmer.memory or farmer.profile directly
    would trigger a MissingGreenlet exception ('greenlet_spawn has not been called').
    Proves that enrich_response_with_market_prices safely avoids triggering lazy loading IO
    and queries the database asynchronously.
    """
    from sqlalchemy.exc import MissingGreenlet

    farmer_id = uuid4()

    class DangerousFarmer:
        def __init__(self, fid):
            self.id = fid
            self.phone_number = "919848011234"
            self.preferred_language = "te"

        @property
        def memory(self):
            raise MissingGreenlet(
                "greenlet_spawn has not been called; can't call await_only() here. Was IO attempted in an unexpected place?"
            )

        @property
        def profile(self):
            raise MissingGreenlet(
                "greenlet_spawn has not been called; can't call await_only() here. Was IO attempted in an unexpected place?"
            )

    farmer = DangerousFarmer(farmer_id)

    mock_db = AsyncMock()
    cotton_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal Market",
        district="Warangal",
        modal_price=7500.0,
    )
    mock_get_prices = AsyncMock(return_value=[cotton_price])

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=mock_get_prices):

        # Must execute cleanly without raising MissingGreenlet
        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text="Warangal lo cotton price entha?",
            ai_response="పత్తి ధర వివరాలు:",
            farmer=farmer,
        )

        assert "Warangal Market" in result
        assert "7,500" in result
