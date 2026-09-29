# -*- coding: utf-8 -*-
"""
Regression tests verifying production stock grounding and that empty shop database
never triggers automatic seeding or fabrication.
"""
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.ai.decision_engine import AIDecisionEngine, FarmerIntent
from src.core.models import Farmer, Conversation, Shop, Inventory
from src.shops.service import enrich_response_with_shops, _detect_shop_intent


def _make_clean_mock_db():
    mock_db = AsyncMock()
    exec_res = MagicMock()
    exec_res.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=exec_res)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    return mock_db


@pytest.mark.asyncio
async def test_enrich_shops_empty_db_never_calls_seed_default_shops():
    """
    Specific regression test proving:
    Calling the stock enrichment path with an empty Shop table does NOT call
    seed_default_shops_if_empty().
    """
    mock_db = _make_clean_mock_db()
    mock_farmer = MagicMock(spec=Farmer)
    mock_farmer.id = uuid4()
    mock_farmer.preferred_language = "te"
    mock_farmer.district = "Jagtial"

    with patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock) as mock_seed, \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await enrich_response_with_shops(
            db=mock_db,
            query_text="Korutla lo urea undha?",
            ai_response="",
            farmer=mock_farmer,
        )

        # MUST NEVER call seed_default_shops_if_empty
        mock_seed.assert_not_called()
        # Must return localized unverified notice
        assert "నమోదిత" in res or "అందుబాటులో లేరు" in res or "సమీప" in res
        # Must never expose hardcoded demo/default shops
        assert "Mallanna" not in res
        assert "Kisan Seva Kendra" not in res
        assert "Rythu Mithra" not in res


@pytest.mark.asyncio
async def test_empty_shop_database_urea_query_never_creates_shop_or_inventory():
    """
    Empty shop database + Urea query must NOT create any shop or inventory:
    - No Shop or Inventory added to db
    - seed_default_shops_if_empty is never called
    - Zero Gemini calls
    """
    query = "Korutla lo urea available ga undha?"
    mock_db = _make_clean_mock_db()

    mock_farmer = Farmer(id=uuid4(), phone_number="919848011230", preferred_language="te")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=query)

    engine = AIDecisionEngine()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock) as mock_seed, \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)

        # 1. seed_default_shops_if_empty was NOT called
        mock_seed.assert_not_called()
        # 2. No shop or inventory was added to DB
        added_objects = [call_args[0][0] for call_args in mock_db.add.call_args_list]
        assert not any(isinstance(obj, (Shop, Inventory)) for obj in added_objects)
        # 3. Gemini was bypassed
        mock_gemini.assert_not_called()
        # 4. Factual unverified notice returned
        assert "🏬" in res
        assert "నమోదిత" in res or "అందుబాటులో లేరు" in res or "సమీప" in res
        # 5. Zero demo fabrication
        assert "Mallanna" not in res
        assert "8976547654" not in res
        assert "₹295" not in res


@pytest.mark.asyncio
async def test_empty_verified_inventory_returns_unavailable_notice():
    """
    When verified inventory in database is empty:
    Urea query returns localized verified current stock unavailable response.
    """
    for query, lang, expected_fragments in [
        ("Korutla lo urea undha?", "te", ["నమోదిత", "అందుబాటులో లేరు"]),
        ("Is urea available in Korutla?", "en", ["No licensed dealer is currently registered"]),
    ]:
        mock_db = _make_clean_mock_db()
        mock_farmer = Farmer(id=uuid4(), phone_number="919848011231", preferred_language=lang)
        mock_farmer.district = "Jagtial"
        mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=query)

        engine = AIDecisionEngine()

        with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
             patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock) as mock_seed, \
             patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

            res = await engine.process_message(mock_db, mock_farmer, mock_conv)

            mock_seed.assert_not_called()
            mock_gemini.assert_not_called()
            assert any(frag in res for frag in expected_fragments)
            assert "Mallanna" not in res
            assert "₹295" not in res


@pytest.mark.asyncio
async def test_existing_verified_inventory_returns_only_actual_database_inventory():
    """
    When database has existing legitimate inventory:
    return ONLY the actual database inventory records.
    """
    real_shop = Shop(
        id=uuid4(),
        shop_name="Telangana Agro Agencies",
        district="Jagtial",
        address="Main Road, Korutla",
        phone_number="+91 9123456780",
        status="active",
        delivery_available=True,
    )
    real_item = Inventory(
        id=uuid4(),
        shop_id=real_shop.id,
        product_name="Neem Coated Urea",
        category="Fertilizers",
        brand="IFFCO",
        price=266.5,
        unit="Bag",
        quantity_in_stock=25,
        available=True,
        minimum_stock_level=5,
    )

    mock_db = _make_clean_mock_db()
    mock_farmer = Farmer(id=uuid4(), phone_number="919848011232", preferred_language="te")
    mock_farmer.district = "Jagtial"
    mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message="Korutla lo urea undha?")

    engine = AIDecisionEngine()

    with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
         patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock) as mock_seed, \
         patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(real_shop, real_item)]):

        res = await engine.process_message(mock_db, mock_farmer, mock_conv)

        mock_seed.assert_not_called()
        mock_gemini.assert_not_called()
        assert "Telangana Agro Agencies" in res
        assert "266.5" in res
        assert "25 Bag" in res
        assert "+91 9123456780" in res
        assert "Mallanna" not in res


@pytest.mark.asyncio
async def test_romanized_telugu_stock_queries_verified_and_unverified():
    """
    Romanized Telugu variations:
    - 'Korutla lo urea undha?'
    - 'Korutla lo urea available ga undha?'
    - 'Korutla lo urea vundha?'
    All properly routed, never calling seed_default_shops_if_empty and never fabricating demo shops.
    """
    test_queries = [
        "Korutla lo urea undha?",
        "Korutla lo urea available ga undha?",
        "Korutla lo urea vundha?",
        "Korutla lo urea unda?",
    ]

    for q in test_queries:
        assert _detect_shop_intent(q.lower(), q) is True
        assert AIDecisionEngine.detect_primary_intent(q) == FarmerIntent.SHOPS

        mock_db = _make_clean_mock_db()
        mock_farmer = Farmer(id=uuid4(), phone_number="919848011233", preferred_language="te")
        mock_farmer.district = "Jagtial"
        mock_conv = Conversation(id=uuid4(), farmer_id=mock_farmer.id, user_message=q)

        engine = AIDecisionEngine()

        with patch("src.ai.service.AIService.generate_ai_response", new_callable=AsyncMock) as mock_gemini, \
             patch("src.shops.repository.ShopRepository.seed_default_shops_if_empty", new_callable=AsyncMock) as mock_seed, \
             patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[]):

            res = await engine.process_message(mock_db, mock_farmer, mock_conv)

            mock_seed.assert_not_called()
            mock_gemini.assert_not_called()
            assert "🏬" in res
            assert "నమోదిత" in res or "అందుబాటులో లేరు" in res or "సమీప" in res
            assert "Mallanna" not in res


@pytest.mark.asyncio
async def test_inventory_timestamp_uses_last_updated_not_verified():
    """
    Requirement 11:
    Verify the 'Verified' label.
    If the system displays a database timestamp, use an appropriate 'Last updated' label
    when it is only a database update timestamp, NOT 'Verified'.
    """
    shop = Shop(
        id=uuid4(),
        shop_name="Guntur Kisan Mart",
        district="Guntur",
        address="Brodipet, Guntur",
        phone_number="9848012345",
        status="active",
        delivery_available=True,
    )
    item = Inventory(
        id=uuid4(),
        shop_id=shop.id,
        product_name="DAP",
        category="Fertilizers",
        brand="IFFCO",
        price=1350.0,
        unit="Bag",
        quantity_in_stock=40,
        available=True,
        minimum_stock_level=5,
        last_updated=datetime(2026, 9, 20, 10, 0, 0),
    )

    mock_db = _make_clean_mock_db()
    mock_farmer_en = Farmer(id=uuid4(), phone_number="919848011234", preferred_language="en")
    mock_farmer_en.district = "Guntur"

    # English check
    with patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(shop, item)]):
        res_en = await enrich_response_with_shops(
            db=mock_db,
            query_text="Where can I buy DAP?",
            ai_response="",
            farmer=mock_farmer_en,
        )
        assert "Last updated: 20-09-2026" in res_en
        assert "Verified: 20-09-2026" not in res_en

    # Telugu check
    mock_farmer_te = Farmer(id=uuid4(), phone_number="919848011235", preferred_language="te")
    mock_farmer_te.district = "Guntur"
    with patch("src.shops.repository.ShopRepository.search_shops_by_product", new_callable=AsyncMock, return_value=[(shop, item)]):
        res_te = await enrich_response_with_shops(
            db=mock_db,
            query_text="DAP ఎక్కడ దొరుకుతుంది?",
            ai_response="",
            farmer=mock_farmer_te,
        )
        assert "చివరిగా అప్‌డేట్ చేయబడింది: 20-09-2026" in res_te
        assert "Verified: 20-09-2026" not in res_te
