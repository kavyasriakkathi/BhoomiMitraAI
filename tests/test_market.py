"""
BhoomiMitra AI — Market Prices Tests

Tests for the market price module.
Uses TestClient + AsyncMock + dependency_overrides.
Pattern mirrors tests/test_schemes.py exactly.

NO real DB connections. NO real API calls. NO real Redis.
"""
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

import pytest

from fastapi.testclient import TestClient  # noqa: E402
from src.main import app  # noqa: E402
from src.market.schemas import (  # noqa: E402
    MarketPriceResponse,
    MarketPriceQueryResponse,
)
from src.market.service import MarketService  # noqa: E402
from src.market.dependencies import get_market_service  # noqa: E402

client = TestClient(app)
NOW = datetime.utcnow()
PRICE_DATE = datetime(2026, 8, 19, 0, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_market_service():
    service = AsyncMock(spec=MarketService)
    app.dependency_overrides[get_market_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_price_response(**kwargs) -> MarketPriceResponse:
    defaults = dict(
        id=uuid4(),
        commodity="Tomato",
        commodity_telugu="టమాటా",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        min_price=900.0,
        max_price=1500.0,
        modal_price=1200.0,
        unit="Quintal",
        price_date=PRICE_DATE,
        source="agmarknet_api",
        created_at=NOW,
    )
    defaults.update(kwargs)
    return MarketPriceResponse(**defaults)


def _mock_query_response(data_available=True, **kwargs) -> MarketPriceQueryResponse:
    results = [_mock_price_response()] if data_available else []
    defaults = dict(
        commodity="Tomato",
        district="Warangal",
        state="Telangana",
        results=results,
        data_available=data_available,
        data_freshness_hours=2.5 if data_available else None,
        source_note="Live data from Agmarknet" if data_available else "No data available.",
        is_live=data_available,
    )
    defaults.update(kwargs)
    return MarketPriceQueryResponse(**defaults)


# ---------------------------------------------------------------------------
# 1. GET /market/prices — data found
# ---------------------------------------------------------------------------

def test_get_market_prices_by_commodity(mock_market_service):
    """GET /market/prices?commodity=Tomato returns 200 with price data."""
    mock_market_service.get_prices_for_query.return_value = _mock_query_response()

    response = client.get("/market/prices?commodity=Tomato")
    assert response.status_code == 200
    data = response.json()
    assert data["commodity"] == "Tomato"
    assert data["data_available"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["modal_price"] == 1200.0
    assert data["results"][0]["market_name"] == "Warangal Mandi"


# ---------------------------------------------------------------------------
# 2. GET /market/prices — no data available
# ---------------------------------------------------------------------------

def test_get_market_prices_not_found(mock_market_service):
    """Returns 200 with data_available=False and empty results (not an error)."""
    mock_market_service.get_prices_for_query.return_value = _mock_query_response(
        data_available=False
    )

    response = client.get("/market/prices?commodity=Tomato")
    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is False
    assert data["results"] == []


# ---------------------------------------------------------------------------
# 3. GET /market/prices/commodities — list
# ---------------------------------------------------------------------------

def test_list_commodities(mock_market_service):
    """Returns a list of distinct commodity names."""
    mock_market_service.list_commodities.return_value = ["Paddy", "Tomato"]

    response = client.get("/market/prices/commodities")
    assert response.status_code == 200
    data = response.json()
    assert "Tomato" in data
    assert "Paddy" in data


# ---------------------------------------------------------------------------
# 4. POST /market/prices — admin seed (authorised)
# ---------------------------------------------------------------------------

def test_create_market_price_authorised(mock_market_service):
    """Admin seed with correct token creates a price record."""
    mock_market_service.create_price.return_value = _mock_price_response()

    payload = {
        "commodity": "Tomato",
        "market_name": "Warangal Mandi",
        "district": "Warangal",
        "state": "Telangana",
        "min_price": 900.0,
        "max_price": 1500.0,
        "modal_price": 1200.0,
        "unit": "Quintal",
        "price_date": PRICE_DATE.isoformat(),
        "source": "manual_seed",
    }

    # Use a non-empty verify token (mocked via settings patch)
    with patch("src.market.router.get_settings") as mock_settings:
        mock_settings.return_value.whatsapp_verify_token = "test-token"
        response = client.post(
            "/market/prices",
            json=payload,
            headers={"X-Admin-Token": "test-token"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["commodity"] == "Tomato"
    assert data["modal_price"] == 1200.0


# ---------------------------------------------------------------------------
# 5. POST /market/prices — unauthorised (missing token)
# ---------------------------------------------------------------------------

def test_create_market_price_unauthorised(mock_market_service):
    """POST without correct X-Admin-Token returns 401."""
    payload = {
        "commodity": "Tomato",
        "market_name": "Warangal Mandi",
        "district": "Warangal",
        "state": "Telangana",
        "min_price": 900.0,
        "max_price": 1500.0,
        "modal_price": 1200.0,
        "unit": "Quintal",
        "price_date": PRICE_DATE.isoformat(),
    }
    with patch("src.market.router.get_settings") as mock_settings:
        mock_settings.return_value.whatsapp_verify_token = "real-token"
        response = client.post(
            "/market/prices",
            json=payload,
            headers={"X-Admin-Token": "wrong-token"},
        )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 6. Intent detection — English
# ---------------------------------------------------------------------------

def test_commodity_intent_detection_english():
    """English price queries correctly identify the commodity."""
    from src.market.service import COMMODITY_MAP, PRICE_INTENT_KEYWORDS_EN

    query = "what is today's tomato price?"
    query_lower = query.lower()

    has_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_EN)
    assert has_intent, "Should detect price intent in English query"

    matched = None
    for kw, canonical in COMMODITY_MAP.items():
        if kw.lower() in query_lower:
            matched = canonical
            break
    assert matched == "Tomato", f"Expected 'Tomato', got '{matched}'"


# ---------------------------------------------------------------------------
# 7. Intent detection — Telugu
# ---------------------------------------------------------------------------

def test_commodity_intent_detection_telugu():
    """Telugu price queries correctly identify the commodity."""
    from src.market.service import COMMODITY_MAP, PRICE_INTENT_KEYWORDS_TE

    query = "టమాటా ధర ఎంత?"
    has_intent = any(kw in query for kw in PRICE_INTENT_KEYWORDS_TE)
    assert has_intent, "Should detect price intent in Telugu query"

    matched = None
    for kw, canonical in COMMODITY_MAP.items():
        if kw in query:
            matched = canonical
            break
    assert matched == "Tomato", f"Expected 'Tomato', got '{matched}'"


# ---------------------------------------------------------------------------
# 8. No intent — enrichment skipped
# ---------------------------------------------------------------------------

def test_no_price_intent_skipped():
    """Non-price query does NOT trigger market enrichment intent detection."""
    from src.market.service import PRICE_INTENT_KEYWORDS_EN, PRICE_INTENT_KEYWORDS_TE

    query = "how to prevent aphids on my crops?"
    query_lower = query.lower()
    has_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_EN)
    has_intent = has_intent or any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_TE)
    assert not has_intent, "Pest query should NOT trigger market price intent"


# ---------------------------------------------------------------------------
# 9. AgmarknetClient — no API key configured
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agmarknet_client_no_key():
    """Client with empty API key returns [] without making any HTTP call."""
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
    )
    with patch("httpx.AsyncClient.get") as mock_get:
        result = await client_obj.fetch_prices("Tomato", state="Telangana")
    assert result == []
    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# 10. AgmarknetClient — API HTTP error returns []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agmarknet_client_api_failure():
    """Client handles API HTTP 500 gracefully — returns [] without raising."""
    import httpx
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="fake-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
    )

    # Patch Redis cache to always miss
    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()):
        # Simulate HTTP 500 response
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch("httpx.AsyncClient.get", return_value=mock_response):
            result = await client_obj.fetch_prices("Tomato", state="Telangana")

    assert result == []


# ---------------------------------------------------------------------------
# 11. format_whatsapp_reply — English, data available
# ---------------------------------------------------------------------------

def test_format_reply_english():
    """English reply contains expected fields."""
    from src.market.service import MarketService as MS

    svc = MS.__new__(MS)  # bypass __init__
    query_resp = _mock_query_response()
    reply = svc.format_whatsapp_reply(query_resp, language="en")

    assert "Tomato" in reply
    assert "1,200" in reply
    assert "Warangal Mandi" in reply
    assert "Modal Price" in reply
    assert "19 Aug 2026" in reply


# ---------------------------------------------------------------------------
# 12. format_whatsapp_reply — Telugu, data available
# ---------------------------------------------------------------------------

def test_format_reply_telugu():
    """Telugu reply contains Telugu labels."""
    from src.market.service import MarketService as MS

    svc = MS.__new__(MS)
    query_resp = _mock_query_response()
    reply = svc.format_whatsapp_reply(query_resp, language="te")

    assert "టమాటా" in reply or "Tomato" in reply   # title uses commodity name
    assert "మోడల్ ధర" in reply
    assert "కనిష్ట" in reply


# ---------------------------------------------------------------------------
# 13. format_whatsapp_reply — data unavailable (English)
# ---------------------------------------------------------------------------

def test_format_reply_unavailable_english():
    """When no data is available, returns an honest unavailable message."""
    from src.market.service import MarketService as MS

    svc = MS.__new__(MS)
    query_resp = _mock_query_response(data_available=False)
    reply = svc.format_whatsapp_reply(query_resp, language="en")

    assert "could not find" in reply.lower() or "unavailable" in reply.lower()
    assert "1800" in reply  # Helpline number


# ---------------------------------------------------------------------------
# 14. format_whatsapp_reply — data unavailable (Telugu)
# ---------------------------------------------------------------------------

def test_format_reply_unavailable_telugu():
    """Telugu unavailable message is returned for Telugu farmers."""
    from src.market.service import MarketService as MS

    svc = MS.__new__(MS)
    query_resp = _mock_query_response(data_available=False)
    reply = svc.format_whatsapp_reply(query_resp, language="te")

    assert "1800" in reply  # Helpline number


# ---------------------------------------------------------------------------
# 15. Paddy intent test
# ---------------------------------------------------------------------------

def test_paddy_intent_detection():
    """'What is the current price of paddy?' detects Paddy correctly."""
    from src.market.service import COMMODITY_MAP, PRICE_INTENT_KEYWORDS_EN

    query = "what is the current price of paddy?"
    query_lower = query.lower()

    has_intent = any(kw in query_lower for kw in PRICE_INTENT_KEYWORDS_EN)
    assert has_intent

    matched = None
    for kw, canonical in COMMODITY_MAP.items():
        if kw.lower() in query_lower:
            matched = canonical
            break
    assert matched == "Paddy"


def _mock_price_model(**kwargs):
    from src.core.models import MarketPrice
    defaults = dict(
        id=uuid4(),
        commodity="Tomato",
        commodity_telugu="టమాటా",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        min_price=900.0,
        max_price=1500.0,
        modal_price=1200.0,
        unit="Quintal",
        price_date=PRICE_DATE,
        source="agmarknet_api",
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(kwargs)
    return MarketPrice(**defaults)


# ---------------------------------------------------------------------------
# 16. Regression Tests: Telugu Market Queries & Unrelated Query Guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_market_enrichment_telugu_query_1():
    """Verify 'ఈరోజు పత్తి ధర ఎంత?' detects intent and enriches response with Cotton prices."""
    from src.market.service import enrich_response_with_market_prices
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 0

    mock_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
    )
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    farmer = MagicMock()
    farmer.id = uuid4()
    farmer.preferred_language = "te"

    base_ai_response = "పత్తి పంట మార్కెట్ వివరాలు క్రింద ఇవ్వబడ్డాయి."
    with patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):
        result = await enrich_response_with_market_prices(
            db_mock, "ఈరోజు పత్తి ధర ఎంత?", base_ai_response, farmer
        )

    assert "పత్తి మార్కెట్ ధర" in result
    assert "Warangal Mandi" in result
    assert "7,450" in result
    assert "మోడల్ ధర" in result


@pytest.mark.asyncio
async def test_market_enrichment_telugu_query_2():
    """Verify 'పత్తి క్వింటాల్ ధర ఎంత?' detects intent and enriches response with Cotton prices."""
    from src.market.service import enrich_response_with_market_prices
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 0

    mock_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Adilabad Mandi",
        district="Adilabad",
        state="Telangana",
        modal_price=7350.0,
        min_price=7000.0,
        max_price=7550.0,
    )
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    farmer = MagicMock()
    farmer.id = uuid4()
    farmer.preferred_language = "te"

    base_ai_response = "పత్తి పంట ధర సమాచారం."
    with patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):
        result = await enrich_response_with_market_prices(
            db_mock, "పత్తి క్వింటాల్ ధర ఎంత?", base_ai_response, farmer
        )

    assert "మార్కెట్ ధరలు" in result
    assert "Adilabad Mandi" in result
    assert "7,350" in result


@pytest.mark.asyncio
async def test_market_enrichment_telugu_query_3():
    """Verify 'నా దగ్గర పత్తి మార్కెట్ ధర ఎంత?' detects intent and enriches response with Cotton prices."""
    from src.market.service import enrich_response_with_market_prices
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 0

    mock_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Khammam Mandi",
        district="Khammam",
        state="Telangana",
        modal_price=7400.0,
        min_price=7050.0,
        max_price=7600.0,
    )
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    farmer = MagicMock()
    farmer.id = uuid4()
    farmer.preferred_language = "te"

    base_ai_response = "పత్తి మార్కెట్ సమాచారం."
    with patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):
        result = await enrich_response_with_market_prices(
            db_mock, "నా దగ్గర పత్తి మార్కెట్ ధర ఎంత?", base_ai_response, farmer
        )

    assert "మార్కెట్ ధరలు" in result
    assert "Khammam Mandi" in result
    assert "7,400" in result


@pytest.mark.asyncio
async def test_market_enrichment_unrelated_query_no_trigger():
    """Verify that unrelated agricultural queries do NOT trigger market price enrichment."""
    from src.market.service import enrich_response_with_market_prices

    db_mock = AsyncMock()
    farmer = MagicMock()
    farmer.id = uuid4()
    farmer.preferred_language = "te"

    unrelated_msg = "నా పత్తి పంటకు ఎరువులు ఎలా వేయాలి?"
    original_ai = "పత్తి పంటకు ఎకరానికి 50 కేజీల యూరియా వేయండి."

    result = await enrich_response_with_market_prices(
        db_mock, unrelated_msg, original_ai, farmer
    )

    # Must be returned completely unmodified
    assert result == original_ai
    assert "మార్కెట్ ధరలు" not in result
    assert "మండి" not in result


@pytest.mark.asyncio
async def test_market_enrichment_refusal_stripped():
    """Verify that when AI generates a generic refusal/disclaimer (e-NAM/local market yard), it is stripped and only the price block is returned."""
    from src.market.service import enrich_response_with_market_prices
    from src.market.agmarknet_client import AgmarknetClient

    db_mock = AsyncMock()
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 0

    mock_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
    )
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_price]
    mock_query_res = MagicMock()
    mock_query_res.scalars.return_value = mock_scalars
    mock_profile_res = MagicMock()
    mock_profile_res.scalar_one_or_none.return_value = None

    db_mock.execute.side_effect = [mock_profile_res, mock_count_res, mock_query_res, mock_query_res, mock_query_res]

    farmer = MagicMock()
    farmer.id = uuid4()
    farmer.preferred_language = "te"

    contradictory_ai_response = (
        "నేను కేవలం వ్యవసాయం మరియు పంటల సాగుకు సంబంధించిన విషయాలపై మాత్రమే సహాయం చేయగలను. "
        "పత్తి ధరల సమాచారం కోసం దయచేసి మీ దగ్గరిలోని మార్కెట్ యార్డ్ లేదా ఈ-నామ్ (e-NAM) పోర్టల్ను సంప్రదించండి."
    )

    with patch.object(AgmarknetClient, "fetch_prices", new_callable=AsyncMock, return_value=[]):
        result = await enrich_response_with_market_prices(
            db_mock, "ఈరోజు పత్తి ధర ఎంత?", contradictory_ai_response, farmer
        )

    # Must contain the clean market price block
    assert "పత్తి మార్కెట్ ధర" in result
    assert "Warangal Mandi" in result
    assert "7,450" in result
    assert "స్థానిక డేటాబేస్" in result

    # Contradictory refusal statements must NOT be present
    assert "e-NAM" not in result
    assert "ఈ-నామ్" not in result
    assert "కేవలం వ్యవసాయం" not in result
    assert "విషయాలపై మాత్రమే" not in result


# ---------------------------------------------------------------------------
# 17. Agmarknet Live API & Local DB Fallback Unit & Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agmarknet_client_successful_live_response():
    """Verify AgmarknetClient correctly parses live API response JSON into normalized records."""
    import httpx
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=5.0,
    )

    mock_payload = {
        "records": [
            {
                "commodity": "Cotton",
                "market": "Warangal Mandi",
                "district": "Warangal",
                "state": "Telangana",
                "min_price": "7100",
                "max_price": "7650",
                "modal_price": "7450",
                "arrival_date": "19/08/2026",
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()), \
         patch("httpx.AsyncClient.get", return_value=mock_response):

        records = await client_obj.fetch_prices("Cotton", state="Telangana", district="Warangal")

    assert len(records) == 1
    assert records[0]["commodity"] == "Cotton"
    assert records[0]["market"] == "Warangal Mandi"
    assert records[0]["modal_price"] == 7450.0
    assert records[0]["arrival_date"] == datetime(2026, 8, 19)


@pytest.mark.asyncio
async def test_agmarknet_client_fetch_prices_accepts_is_today_requested():
    """Verify AgmarknetClient.fetch_prices accepts is_today_requested parameter without TypeError."""
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=5.0,
    )

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=[])), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()):
        records = await client_obj.fetch_prices(
            commodity="Cotton",
            state="Telangana",
            district="Warangal",
            is_today_requested=True,
        )
        assert records == []


@pytest.mark.asyncio
async def test_agmarknet_client_timeout_handling():
    """Verify AgmarknetClient gracefully catches httpx.TimeoutException and returns [] without raising."""
    import httpx
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=5.0,
    )

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()), \
         patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Read timed out")):

        records = await client_obj.fetch_prices("Cotton", state="Telangana")

    assert records == []


@pytest.mark.asyncio
async def test_agmarknet_client_network_error_handling():
    """Verify AgmarknetClient gracefully handles network connection errors and returns []."""
    import httpx
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=5.0,
    )

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()), \
         patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):

        records = await client_obj.fetch_prices("Cotton", state="Telangana")

    assert records == []


@pytest.mark.asyncio
async def test_agmarknet_client_strict_timeout_enforced():
    """Verify AgmarknetClient strictly enforces configured timeout when request hangs."""
    import asyncio
    import time
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=0.1,
    )

    async def slow_get(*args, **kwargs):
        await asyncio.sleep(3.0)
        return MagicMock(status_code=200, json=lambda: {"records": []})

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()), \
         patch("httpx.AsyncClient.get", side_effect=slow_get):

        t0 = time.time()
        records = await client_obj.fetch_prices("Cotton", state="Telangana")
        elapsed = time.time() - t0

    assert records == []
    # Must have timed out via asyncio.wait_for near 0.1s rather than waiting full 3.0s
    assert elapsed < 2.9


@pytest.mark.asyncio
async def test_agmarknet_client_infers_telangana_state_for_warangal():
    """Verify AgmarknetClient automatically resolves state='Telangana' when district is Warangal and state is None."""
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(
        api_key="valid-test-key",
        api_url="https://api.data.gov.in/resource/test",
        cache_ttl_seconds=3600,
        timeout_seconds=5.0,
    )

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"records": []}

    with patch.object(client_obj, "_get_from_cache", new=AsyncMock(return_value=None)), \
         patch.object(client_obj, "_set_in_cache", new=AsyncMock()), \
         patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)) as mock_get:

        await client_obj.fetch_prices("Cotton", state=None, district="Warangal")

        assert mock_get.called
        call_kwargs = mock_get.call_args[1]
        params = call_kwargs["params"]
        assert params["filters[commodity]"] == "Cotton"
        assert params["filters[district]"] == "Warangal"
        assert params["filters[state.keyword]"] == "Telangana"


@pytest.mark.asyncio
async def test_market_service_live_today_warangal_cotton_data_used_directly():
    """Verify that when live API returns today's Warangal cotton records, live data is used with is_live=True."""
    from src.market.service import MarketService, enrich_response_with_market_prices, IST_TZ
    from src.market.agmarknet_client import AgmarknetClient
    from src.core.models import Farmer

    mock_repo = AsyncMock()
    client_obj = AgmarknetClient(api_key="key", api_url="https://api.data.gov.in/test")

    today_dt = datetime.now(IST_TZ).replace(tzinfo=None)
    live_today_records = [
        {
            "commodity": "Cotton",
            "market": "Warangal Mandi",
            "district": "Warangal",
            "state": "Telangana",
            "min_price": 7500.0,
            "max_price": 8100.0,
            "modal_price": 7850.0,
            "arrival_date": today_dt,
        }
    ]

    with patch.object(client_obj, "fetch_prices", new=AsyncMock(return_value=live_today_records)):
        svc = MarketService(repository=mock_repo, client=client_obj)
        res = await svc.get_prices_for_query("Cotton", district="Warangal", state="Telangana", is_today_requested=True)

    assert res.data_available is True
    assert res.is_live is True
    assert res.data_freshness_hours == 0.0
    assert len(res.results) == 1
    assert res.results[0].modal_price == 7850.0
    assert res.results[0].id is not None  # Must have valid UUID, not None
    mock_repo.upsert_prices.assert_called_once()

    # Verify formatting produces the normal title without unavailable warning
    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="te")
    reply = svc.format_whatsapp_reply(res, language="te", is_today_query=True)
    assert "📊 పత్తి మార్కెట్ ధరలు" in reply
    assert "అందుబాటులో లేదు" not in reply
    assert "Warangal Mandi" in reply
    assert "7,850" in reply



@pytest.mark.asyncio
async def test_market_service_live_api_success_path():
    """Verify MarketService upserts live records and marks response as is_live=True."""
    from src.market.service import MarketService
    from src.market.agmarknet_client import AgmarknetClient

    mock_repo = AsyncMock()
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal Mandi",
        modal_price=7450.0,
        source="agmarknet_api",
    )
    mock_repo.get_prices_by_commodity.return_value = [mock_db_price]
    mock_repo.upsert_prices.return_value = 1

    client_obj = AgmarknetClient(api_key="key", api_url="https://api.data.gov.in/test")
    raw_api_records = [
        {
            "commodity": "Cotton",
            "market": "Warangal Mandi",
            "district": "Warangal",
            "state": "Telangana",
            "min_price": 7100.0,
            "max_price": 7650.0,
            "modal_price": 7450.0,
            "arrival_date": datetime(2026, 8, 19),
        }
    ]

    with patch.object(client_obj, "fetch_prices", new=AsyncMock(return_value=raw_api_records)):
        svc = MarketService(repository=mock_repo, client=client_obj)
        response = await svc.get_prices_for_query("Cotton", district="Warangal", state="Telangana")

    assert response.data_available is True
    assert response.is_live is True
    assert "Live data from Agmarknet" in response.source_note
    assert len(response.results) == 1
    assert response.results[0].modal_price == 7450.0
    mock_repo.upsert_prices.assert_called_once()


@pytest.mark.asyncio
async def test_market_service_api_timeout_local_db_fallback():
    """Verify MarketService falls back seamlessly to local DB when live API times out."""
    from src.market.service import MarketService
    from src.market.agmarknet_client import AgmarknetClient

    mock_repo = AsyncMock()
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal Mandi",
        modal_price=7400.0,
        source="local_seed",
    )
    mock_repo.get_prices_by_commodity.return_value = [mock_db_price]

    client_obj = AgmarknetClient(api_key="key", api_url="https://api.data.gov.in/test")

    # fetch_prices returns [] on timeout
    with patch.object(client_obj, "fetch_prices", new=AsyncMock(return_value=[])):
        svc = MarketService(repository=mock_repo, client=client_obj)
        response = await svc.get_prices_for_query("Cotton", district="Warangal", state="Telangana")

    assert response.data_available is True
    assert response.is_live is False
    assert "Local database" in response.source_note
    assert len(response.results) == 1
    assert response.results[0].modal_price == 7400.0
    mock_repo.get_prices_by_commodity.assert_called_once_with(
        commodity="Cotton", district="Warangal", state="Telangana"
    )


@pytest.mark.asyncio
async def test_enrich_response_extracts_district_from_query_and_preserves_db_date():
    """Verify that query mentioning 'వరంగల్లో' extracts Warangal district and outputs exact DB record date."""
    from src.market.service import enrich_response_with_market_prices
    from src.core.models import Farmer

    mock_db = AsyncMock()
    # Mock profile query returning no profile
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="te")
    query_text = "వరంగల్లో ఈరోజు పత్తి ధర ఎంత?"
    ai_response = "పత్తి పంట ధర సమాచారం:"

    past_date = datetime(2026, 8, 23)
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=past_date,
        source="manual_seed",
    )

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=AsyncMock(return_value=[mock_db_price])) as mock_get_prices:
        
        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text=query_text,
            ai_response=ai_response,
            farmer=farmer,
        )

        assert "⚠️ ఈరోజు వరంగల్ పత్తి మార్కెట్ ధర డేటా అందుబాటులో లేదు." in result
        assert "📅 చివరిగా లభించిన ధర: 23 Aug 2026" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result
        assert "23 Aug 2026" in result
        assert "స్థానిక డేటాబేస్" in result
        mock_get_prices.assert_called_once_with(
            commodity="Cotton",
            district="Warangal",
            state=None,
        )


@pytest.mark.asyncio
async def test_today_market_query_telugu_with_today_data_available():
    """Verify that when today's data is available, it returns standard title without warning."""
    from src.market.service import enrich_response_with_market_prices, IST_TZ
    from src.core.models import Farmer

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="te")
    query_text = "వరంగల్లో ఈరోజు పత్తి ధర ఎంత?"
    ai_response = "పత్తి సమాచారం"

    today_dt = datetime.now(IST_TZ).replace(tzinfo=None)
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        modal_price=7800.0,
        min_price=7500.0,
        max_price=8000.0,
        price_date=today_dt,
        source="agmarknet_api",
    )

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=AsyncMock(return_value=[mock_db_price])):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text=query_text,
            ai_response=ai_response,
            farmer=farmer,
        )

        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "అందుబాటులో లేదు" not in result
        assert "Warangal Mandi" in result
        assert "7,800" in result
        assert today_dt.strftime("%d %b %Y") in result


@pytest.mark.asyncio
async def test_today_market_query_english_old_record_shows_warning_and_preserves_date():
    """Verify that an English query asking for today's price when only old data exists shows warning and exact date."""
    from src.market.service import enrich_response_with_market_prices
    from src.core.models import Farmer

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="en")
    query_text = "What is the cotton price in Warangal today?"
    ai_response = "Cotton price info:"

    past_date = datetime(2026, 8, 23)
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        market_name="Warangal Mandi",
        district="Warangal",
        state="Telangana",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=past_date,
        source="local_seed",
    )

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=AsyncMock(return_value=[mock_db_price])):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text=query_text,
            ai_response=ai_response,
            farmer=farmer,
        )

        assert "⚠️ Today's market price data for Warangal Cotton is unavailable." in result
        assert "📅 Last available price: 23 Aug 2026" in result
        assert "Warangal Mandi" in result
        assert "7,450" in result
        assert "23 Aug 2026" in result


@pytest.mark.asyncio
async def test_latest_market_query_without_today_keyword_shows_standard_header_with_true_date():
    """Verify that general queries (not asking for today) return standard header with exact historic record date."""
    from src.market.service import enrich_response_with_market_prices
    from src.core.models import Farmer

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_res

    farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="te")
    query_text = "వరంగల్ పత్తి మార్కెట్ ధర ఎంత?"
    ai_response = "పత్తి సమాచారం"

    past_date = datetime(2026, 8, 23)
    mock_db_price = _mock_price_model(
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",
        modal_price=7450.0,
        min_price=7100.0,
        max_price=7650.0,
        price_date=past_date,
        source="local_seed",
    )

    with patch("src.market.repository.MarketPriceRepository.seed_default_prices_if_empty", new=AsyncMock()), \
         patch("src.market.agmarknet_client.AgmarknetClient.fetch_prices", new=AsyncMock(return_value=[])), \
         patch("src.market.repository.MarketPriceRepository.get_prices_by_commodity", new=AsyncMock(return_value=[mock_db_price])):

        result = await enrich_response_with_market_prices(
            db=mock_db,
            query_text=query_text,
            ai_response=ai_response,
            farmer=farmer,
        )

        assert "📊 పత్తి మార్కెట్ ధరలు" in result
        assert "⚠️ ఈరోజు" not in result
        assert "Warangal Mandi" in result
        assert "7,450" in result
        assert "23 Aug 2026" in result
