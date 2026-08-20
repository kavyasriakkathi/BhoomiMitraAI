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
