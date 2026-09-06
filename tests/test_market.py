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

    assert "మార్కెట్ ధరలు" in result
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
    assert "📊 పత్తి మార్కెట్ ధరలు" in result
    assert "Warangal Mandi" in result
    assert "7,450" in result
    assert "స్థానిక డేటాబేస్" in result

    # Contradictory refusal statements must NOT be present
    assert "e-NAM" not in result
    assert "ఈ-నామ్" not in result
    assert "కేవలం వ్యవసాయం" not in result
    assert "విషయాలపై మాత్రమే" not in result


# ---------------------------------------------------------------------------
# 17. Regression Tests: Local Market Routing Hierarchy & Anti-Hijack
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_warangal_farmer_wins_over_live_out_of_state_records():
    """1. Warangal farmer + live AP/MP records → Telangana/Warangal records win."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.pool import StaticPool
    from src.core.database import Base
    from src.core.models import MarketPrice
    from src.market.repository import MarketPriceRepository
    from datetime import datetime, timedelta

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.utcnow()
    async with factory() as db:
        # Live out-of-state records (e.g. MP, AP with current date)
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Khargone APMC",
            district="Khargone",
            state="Madhya Pradesh",
            modal_price=7100.0,
            min_price=6800.0,
            max_price=7300.0,
            unit="Quintal",
            price_date=now,
            source="agmarknet_api",
        ))
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Cumbum APMC",
            district="Prakasam",
            state="Andhra Pradesh",
            modal_price=7200.0,
            min_price=6900.0,
            max_price=7400.0,
            unit="Quintal",
            price_date=now,
            source="agmarknet_api",
        ))
        # Local Warangal DB record (seeded/older date)
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Warangal Mandi",
            district="Warangal",
            state="Telangana",
            modal_price=7450.0,
            min_price=7100.0,
            max_price=7650.0,
            unit="Quintal",
            price_date=now - timedelta(days=5),
            source="manual_seed",
        ))
        await db.commit()

        repo = MarketPriceRepository(db)
        results = await repo.get_prices_by_commodity(
            commodity="Cotton",
            district="Warangal",
            state="Telangana",
            limit_days=3,
        )

        assert len(results) > 0
        assert results[0].market_name == "Warangal Mandi"
        assert results[0].district == "Warangal"
        assert results[0].state == "Telangana"


@pytest.mark.asyncio
async def test_warangal_farmer_latest_local_district_wins_when_no_recent_district():
    """2. Warangal farmer + no current local rows → latest local district rows win."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.pool import StaticPool
    from src.core.database import Base
    from src.core.models import MarketPrice
    from src.market.repository import MarketPriceRepository
    from datetime import datetime, timedelta

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.utcnow()
    async with factory() as db:
        # Older Warangal record
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Enumamula Mandi",
            district="Warangal",
            state="Telangana",
            modal_price=7500.0,
            min_price=7200.0,
            max_price=7700.0,
            unit="Quintal",
            price_date=now - timedelta(days=10),
            source="manual_seed",
        ))
        # Newer out-of-state record
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Krosuru APMC",
            district="Guntur",
            state="Andhra Pradesh",
            modal_price=7100.0,
            min_price=6800.0,
            max_price=7300.0,
            unit="Quintal",
            price_date=now,
            source="agmarknet_api",
        ))
        await db.commit()

        repo = MarketPriceRepository(db)
        results = await repo.get_prices_by_commodity(
            commodity="Cotton",
            district="Warangal",
            state="Telangana",
            limit_days=3,
        )

        assert len(results) > 0
        assert results[0].market_name == "Enumamula Mandi"
        assert results[0].state == "Telangana"


@pytest.mark.asyncio
async def test_warangal_farmer_telangana_rows_win_when_no_district_rows():
    """3. Warangal farmer + no district rows → Telangana rows win."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.pool import StaticPool
    from src.core.database import Base
    from src.core.models import MarketPrice
    from src.market.repository import MarketPriceRepository
    from datetime import datetime

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.utcnow()
    async with factory() as db:
        # Telangana record from another district (Karimnagar)
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Karimnagar Mandi",
            district="Karimnagar",
            state="Telangana",
            modal_price=7480.0,
            min_price=7150.0,
            max_price=7680.0,
            unit="Quintal",
            price_date=now,
            source="manual_seed",
        ))
        # Out-of-state record
        db.add(MarketPrice(
            commodity="Cotton",
            market_name="Khargone APMC",
            district="Khargone",
            state="Madhya Pradesh",
            modal_price=7100.0,
            min_price=6800.0,
            max_price=7300.0,
            unit="Quintal",
            price_date=now,
            source="agmarknet_api",
        ))
        await db.commit()

        repo = MarketPriceRepository(db)
        results = await repo.get_prices_by_commodity(
            commodity="Cotton",
            district="Warangal",
            state="Telangana",
            limit_days=3,
        )

        assert len(results) > 0
        assert results[0].market_name == "Karimnagar Mandi"
        assert results[0].state == "Telangana"


def test_only_national_rows_clearly_labeled_with_actual_state():
    """4. Only national rows → national fallback is clearly labeled with actual state."""
    from src.market.service import MarketService as MS
    from src.market.schemas import MarketPriceQueryResponse, MarketPriceResponse

    svc = MS.__new__(MS)
    price_resp = MarketPriceResponse(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Khargone APMC",
        district="Khargone",
        state="Madhya Pradesh",
        min_price=6800.0,
        max_price=7300.0,
        modal_price=7100.0,
        unit="Quintal",
        price_date=datetime(2026, 8, 20),
        source="agmarknet_api",
        created_at=NOW,
    )
    query_resp = MarketPriceQueryResponse(
        commodity="Cotton",
        district=None,
        state=None,
        results=[price_resp],
        data_available=True,
        data_freshness_hours=1.0,
        source_note="Live data from Agmarknet",
        is_live=True,
    )

    reply_en = svc.format_whatsapp_reply(query_resp, language="en")
    assert "Khargone APMC, Madhya Pradesh" in reply_en
    assert "Madhya Pradesh" in reply_en

    reply_te = svc.format_whatsapp_reply(query_resp, language="te")
    assert "Khargone APMC, Madhya Pradesh" in reply_te
    assert "Madhya Pradesh" in reply_te


def test_enumamula_alias_resolves_to_warangal_context():
    """5. Enumamula in English and Telugu resolves to Warangal context."""
    from src.weather.service import _extract_district_from_query

    assert _extract_district_from_query("what is cotton price in enumamula?") == "Warangal"
    assert _extract_district_from_query("enamamula market mandi rate") == "Warangal"
    assert _extract_district_from_query("ఎనుమాముల మార్కెట్లో పత్తి ధర ఎంత?") == "Warangal"
    assert _extract_district_from_query("ఏనుమాముల మార్కెట్ ధరలు") == "Warangal"


def test_agmarknet_client_client_side_protection_from_broad_results():
    """6. API returns broader results despite parameters → client-side protection prevents local hijacking."""
    from src.market.agmarknet_client import AgmarknetClient

    client_obj = AgmarknetClient(api_key="", api_url="", cache_ttl_seconds=3600)
    broad_records = [
        {
            "commodity": "Cotton",
            "market": "Khargone APMC",
            "district": "Khargone",
            "state": "Madhya Pradesh",
            "modal_price": 7100.0,
        },
        {
            "commodity": "Cotton",
            "market": "Warangal APMC",
            "district": "Warangal",
            "state": "Telangana",
            "modal_price": 7450.0,
        },
        {
            "commodity": "Cotton",
            "market": "Khammam Mandi",
            "district": "Khammam",
            "state": "Telangana",
            "modal_price": 7400.0,
        },
    ]

    prioritized = client_obj._prioritize_records(
        broad_records, state="Telangana", district="Warangal"
    )
    assert len(prioritized) == 2
    assert prioritized[0]["market"] == "Warangal APMC"
    assert prioritized[0]["state"] == "Telangana"
    assert prioritized[1]["market"] == "Khammam Mandi"


@pytest.mark.asyncio
async def test_upsert_preserves_source_arrival_date():
    """7. Upsert preserves source market date instead of replacing it with now()."""
    from src.market.service import MarketService
    from src.market.repository import MarketPriceRepository
    from src.market.agmarknet_client import AgmarknetClient

    repo_mock = AsyncMock(spec=MarketPriceRepository)
    client_mock = AsyncMock(spec=AgmarknetClient)
    svc = MarketService(repository=repo_mock, client=client_mock)

    exact_source_date = datetime(2026, 8, 15, 10, 30, 0)
    api_records = [
        {
            "commodity": "Cotton",
            "market": "Warangal Mandi",
            "district": "Warangal",
            "state": "Telangana",
            "min_price": 7000.0,
            "max_price": 7600.0,
            "modal_price": 7400.0,
            "arrival_date": exact_source_date,
        },
        {
            "commodity": "Cotton",
            "market": "Invalid Date Mandi",
            "arrival_date": None,  # should be skipped, not replaced with now()
        },
    ]

    await svc._upsert_api_records(api_records, "Cotton")

    repo_mock.upsert_prices.assert_called_once()
    saved_creates = repo_mock.upsert_prices.call_args[0][0]
    assert len(saved_creates) == 1
    assert saved_creates[0].price_date == exact_source_date


def test_existing_market_deduplication_and_formatting_preserved():
    """8. Existing market deduplication/formatting behavior remains unchanged."""
    from src.market.service import MarketService as MS
    from src.market.schemas import MarketPriceQueryResponse, MarketPriceResponse

    svc = MS.__new__(MS)
    price_1 = MarketPriceResponse(
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
        price_date=datetime(2026, 8, 20),
        source="manual_seed",
        created_at=NOW,
    )
    price_2 = MarketPriceResponse(
        id=uuid4(),
        commodity="Cotton",
        commodity_telugu="పత్తి",
        market_name="Warangal Mandi",  # Duplicate market name
        district="Warangal",
        state="Telangana",
        min_price=7000.0,
        max_price=7500.0,
        modal_price=7400.0,
        unit="Quintal",
        price_date=datetime(2026, 8, 19),
        source="manual_seed",
        created_at=NOW,
    )

    query_resp = MarketPriceQueryResponse(
        commodity="Cotton",
        district="Warangal",
        state="Telangana",
        results=[price_1, price_2],
        data_available=True,
        data_freshness_hours=2.0,
        source_note="Local database",
        is_live=False,
    )

    formatted = svc.format_whatsapp_reply(query_resp, language="te")
    # Must only contain Warangal Mandi once
    assert formatted.count("Warangal Mandi") == 1
    assert "7,450" in formatted
