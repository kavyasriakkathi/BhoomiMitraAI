import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.shops.schemas import ShopResponse, FarmerShopSearchResponse, FarmerShopSearchResult
from src.shops.service import ShopService, haversine_distance
from src.shops.dependencies import get_shop_service

client = TestClient(app)
TS = "2026-07-28T12:00:00Z"


@pytest.fixture
def mock_shop_service():
    service = AsyncMock(spec=ShopService)
    app.dependency_overrides[get_shop_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_shop_response(**kwargs) -> ShopResponse:
    defaults = dict(
        id=uuid4(),
        shop_name="Sri Lakshmi Agro Centre",
        owner_name="Ramesh Kumar",
        phone_number="+91 9876543210",
        email="contact@srilakshmiagro.com",
        address="Main Road, Guntur",
        village="Guntur Rural",
        mandal="Guntur",
        district="Guntur",
        state="Andhra Pradesh",
        pin_code="522001",
        latitude=16.3067,
        longitude=80.4365,
        opening_time="08:00 AM",
        closing_time="08:00 PM",
        delivery_available=True,
        home_delivery_radius_km=15.0,
        google_maps_link="https://maps.google.com/?q=16.3067,80.4365",
        gst_number="37ABCDE1234F1Z5",
        license_number="AP/GNT/AGRI/2026/089",
        status="active",
        created_at=TS,
        updated_at=TS,
    )
    defaults.update(kwargs)
    return ShopResponse(**defaults)


def test_haversine_distance():
    # Distance between Guntur (16.3067, 80.4365) and Vijayawada (16.5062, 80.6480) ~30 km
    dist = haversine_distance(16.3067, 80.4365, 16.5062, 80.6480)
    assert 20.0 < dist < 40.0


def test_create_shop(mock_shop_service):
    shop_resp = _mock_shop_response()
    mock_shop_service.create_shop.return_value = shop_resp

    payload = {
        "shop_name": "Sri Lakshmi Agro Centre",
        "owner_name": "Ramesh Kumar",
        "phone_number": "+91 9876543210",
        "address": "Main Road, Guntur",
        "district": "Guntur",
        "state": "Andhra Pradesh",
    }
    response = client.post("/shops", json=payload)
    assert response.status_code == 201
    assert response.json()["shop_name"] == "Sri Lakshmi Agro Centre"


def test_get_shop(mock_shop_service):
    shop_resp = _mock_shop_response()
    mock_shop_service.get_shop_by_id.return_value = shop_resp

    response = client.get(f"/shops/{shop_resp.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(shop_resp.id)


def test_update_shop(mock_shop_service):
    shop_resp = _mock_shop_response(shop_name="Updated Agro Centre")
    mock_shop_service.update_shop.return_value = shop_resp

    response = client.put(f"/shops/{shop_resp.id}", json={"shop_name": "Updated Agro Centre"})
    assert response.status_code == 200
    assert response.json()["shop_name"] == "Updated Agro Centre"


def test_delete_shop(mock_shop_service):
    mock_shop_service.delete_shop.return_value = None
    shop_id = uuid4()
    response = client.delete(f"/shops/{shop_id}")
    assert response.status_code == 204


def test_farmer_shop_search(mock_shop_service):
    shop_id = uuid4()
    search_result = FarmerShopSearchResult(
        shop_id=shop_id,
        shop_name="Sri Lakshmi Agro Centre",
        owner_name="Ramesh Kumar",
        distance_km=2.1,
        product_name="Urea",
        brand="IFFCO",
        price=295.0,
        unit="Bag",
        quantity_in_stock=50,
        phone_number="+91 9876543210",
        opening_time="08:00 AM",
        closing_time="08:00 PM",
        status="Open",
        delivery_available=True,
        formatted_display="Shop Name: Sri Lakshmi Agro Centre\nDistance: 2.1 km\nProduct: Urea",
    )
    search_resp = FarmerShopSearchResponse(query="Urea", total_results=1, results=[search_result])
    mock_shop_service.farmer_product_search.return_value = search_resp

    response = client.get("/shops/farmer-search?query=Urea")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["total_results"] == 1
    assert res_data["results"][0]["product_name"] == "Urea"
    assert res_data["results"][0]["price"] == 295.0


# ===========================================================================
# NEW TESTS: Intent Detection, Normalization, Classification, Formatting, Pipeline
# ===========================================================================

from unittest.mock import MagicMock, AsyncMock, patch
from src.shops.service import (
    _detect_shop_intent,
    _detect_product_from_query,
    _format_stock_string,
    _EN_LABELS,
    _TE_LABELS,
    enrich_response_with_shops,
    ShopRepository,
)


def _make_mock_shop(
    name="Kisan Seva Kendra",
    district="Warangal",
    state="Telangana",
    lat=17.9689,
    lon=79.5941,
    phone="+91 9848012345",
    status="active",
    delivery=True,
    opening="08:00",
    closing="20:00",
):
    shop = MagicMock()
    shop.id = uuid4()
    shop.shop_name = name
    shop.owner_name = "Srinivas Rao"
    shop.district = district
    shop.state = state
    shop.latitude = lat
    shop.longitude = lon
    shop.phone_number = phone
    shop.status = status
    shop.delivery_available = delivery
    shop.opening_time = opening
    shop.closing_time = closing
    return shop


def _make_mock_item(
    name="Urea",
    brand="IFFCO",
    price=295.0,
    unit="Bag",
    stock=100,
    min_stock=10,
    available=True,
):
    item = MagicMock()
    item.id = uuid4()
    item.product_name = name
    item.brand = brand
    item.price = price
    item.unit = unit
    item.quantity_in_stock = stock
    item.minimum_stock_level = min_stock
    item.available = available
    return item


def _mock_db_with_farmer_location(
    memory_gps=None,
    memory_district=None,
    profile_district=None,
    profile_state=None,
    memory_state=None,
):
    mock_mem = MagicMock()
    mock_mem.gps_coordinates = memory_gps or {}
    mock_mem.district = memory_district
    mock_mem.state = memory_state

    mock_prof = MagicMock()
    mock_prof.district = profile_district
    mock_prof.state = profile_state

    mem_result = MagicMock()
    mem_result.scalar_one_or_none.return_value = mock_mem
    prof_result = MagicMock()
    prof_result.scalar_one_or_none.return_value = mock_prof

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[mem_result, prof_result])
    return db


def test_shop_intent_detection_english():
    """English shop and purchase keywords trigger shop intent."""
    assert _detect_shop_intent("where can i buy urea?", "where can i buy urea?") is True
    assert _detect_shop_intent("is dap fertilizer available near me", "is dap fertilizer available near me") is True
    assert _detect_shop_intent("find pesticide price", "find pesticide price") is True
    assert _detect_shop_intent("seed dealer in guntur", "seed dealer in guntur") is True


def test_shop_intent_detection_telugu():
    """Telugu shop and purchase keywords trigger shop intent."""
    assert _detect_shop_intent("", "యూరియా ఎక్కడ దొరుకుతుంది?") is True
    assert _detect_shop_intent("", "పురుగుమందుల షాప్ ఎక్కడ ఉంది?") is True
    assert _detect_shop_intent("", "విత్తనాలు కొనాలి ధర ఎంత?") is True
    assert _detect_shop_intent("", "ఎరువుల డీలర్ అందుబాటులో ఉన్నారా?") is True


def test_non_shop_intent_skipped():
    """Agronomic and pest questions without purchase intent do not trigger shop intent."""
    assert _detect_shop_intent("what fertilizer is recommended for cotton", "what fertilizer is recommended for cotton") is False
    assert _detect_shop_intent("", "టమాటా తెగులు నివారణకు ఏ మందు వాడాలి?") is False
    assert _detect_shop_intent("will it rain tomorrow?", "will it rain tomorrow?") is False


def test_product_normalization_english_and_telugu():
    """Product keywords are correctly normalized to standard names."""
    assert _detect_product_from_query("i need urea bag") == "urea"
    assert _detect_product_from_query("యూరియా ధర ఎంత?") == "urea"
    assert _detect_product_from_query("need dap fertilizer") == "dap"
    assert _detect_product_from_query("డిఎపి సంచి కావాలి") == "dap"
    assert _detect_product_from_query("where to buy neem oil") == "neem oil"
    assert _detect_product_from_query("వేప నూనె ఎక్కడ దొరుకుతుంది?") == "neem oil"
    assert _detect_product_from_query("looking for cotton seeds") == "seeds"
    assert _detect_product_from_query("విత్తనాలు కావాలి") == "seeds"
    assert _detect_product_from_query("general question without product") is None


def test_stock_classification():
    """Stock levels are accurately classified without inventing data."""
    # In Stock
    in_stock = _format_stock_string(50, 10, True, "Bag", _EN_LABELS)
    assert in_stock == "In Stock (50 Bags)"

    # Low Stock
    low_stock = _format_stock_string(5, 10, True, "Bag", _EN_LABELS)
    assert low_stock == "Low Stock (5 Bags)"

    # Out of Stock (0 quantity)
    out_stock = _format_stock_string(0, 10, True, "Bag", _EN_LABELS)
    assert out_stock == "Out of Stock"

    # Out of Stock (available=False)
    unavail = _format_stock_string(50, 10, False, "Bag", _EN_LABELS)
    assert unavail == "Out of Stock"


def test_shop_formatting_english():
    """English formatting displays proper icons and labels."""
    from src.shops.service import _format_stock_string

    stock_str = _format_stock_string(100, 10, True, "Bag", _EN_LABELS)
    assert "In Stock (100 Bags)" in stock_str
    assert "Nearby Agricultural Shops & Availability:" in _EN_LABELS["title"]


def test_shop_formatting_telugu():
    """Telugu formatting displays proper Telugu labels."""
    from src.shops.service import _format_stock_string

    stock_str = _format_stock_string(100, 10, True, "సంచి", _TE_LABELS)
    assert "స్టాక్ అందుబాటులో ఉంది (100 సంచిs)" in stock_str
    assert "సమీప వ్యవసాయ దుకాణాలు" in _TE_LABELS["title"]


@pytest.mark.asyncio
async def test_enrich_shops_gps_location_sorting():
    """Shops enrichment prioritizes the geographically closest shop when GPS is present."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    # Farmer in Warangal (17.9689, 79.5941)
    db = _mock_db_with_farmer_location(memory_gps={"latitude": 17.9689, "longitude": 79.5941})

    close_shop = _make_mock_shop(name="Warangal Agri Centre", lat=17.9700, lon=79.5950, district="Warangal")
    farther_local_shop = _make_mock_shop(name="Kazipet Agro Traders", lat=17.9850, lon=79.5300, district="Warangal")
    item = _make_mock_item()

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(farther_local_shop, item), (close_shop, item)]):

        res = await enrich_response_with_shops(db, "Where can I buy urea?", "Here is your answer.", farmer)

    assert "Warangal Agri Centre" in res
    assert "Kazipet Agro Traders" in res
    # Closest shop should appear before the farther local shop
    assert res.find("Warangal Agri Centre") < res.find("Kazipet Agro Traders")
    assert "km away" in res


@pytest.mark.asyncio
async def test_enrich_shops_farmer_profile_district_fallback():
    """Shops enrichment prioritizes matching district from FarmerProfile and excludes distant districts."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = _mock_db_with_farmer_location(profile_district="Guntur", profile_state="Andhra Pradesh")

    guntur_shop = _make_mock_shop(name="Sri Lakshmi Agro Guntur", district="Guntur", lat=None, lon=None)
    khammam_shop = _make_mock_shop(name="Rythu Mithra Khammam", district="Khammam", lat=None, lon=None)
    item = _make_mock_item()

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(khammam_shop, item), (guntur_shop, item)]):

        res = await enrich_response_with_shops(db, "Where can I buy urea?", "Advice.", farmer)

    assert "Sri Lakshmi Agro Guntur" in res
    # Distant Khammam shop is safely excluded from Guntur farmer
    assert "Rythu Mithra Khammam" not in res


@pytest.mark.asyncio
async def test_enrich_shops_farmer_memory_district_fallback():
    """Shops enrichment resolves district from FarmerMemory if FarmerProfile district is blank."""
    farmer = MagicMock(id=uuid4(), preferred_language="te")
    db = _mock_db_with_farmer_location(memory_district="Warangal", memory_state="Telangana")

    warangal_shop = _make_mock_shop(name="కిసాన్ సేవా కేంద్రం", district="Warangal")
    item = _make_mock_item(name="యూరియా")

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(warangal_shop, item)]):

        res = await enrich_response_with_shops(db, "యూరియా ఎక్కడ కొనాలి?", "సలహా.", farmer)

    assert "కిసాన్ సేవా కేంద్రం" in res
    assert "సమీప వ్యవసాయ దుకాణాలు" in res
    assert "ధర: ₹295/Bag" in res


@pytest.mark.asyncio
async def test_enrich_shops_no_location_returns_all_active():
    """When farmer has no location data, enrichment returns available shops with generic location indicator."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = _mock_db_with_farmer_location()  # No GPS, no district

    shop = _make_mock_shop(name="Central Agri Store", lat=None, lon=None)
    item = _make_mock_item()

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(shop, item)]):

        res = await enrich_response_with_shops(db, "Where to buy DAP?", "Advice.", farmer)

    assert "Central Agri Store" in res
    assert "Nearby" in res


@pytest.mark.asyncio
async def test_enrich_shops_no_product_matched_returns_original():
    """When query has buy intent but no product keyword is recognized, return unchanged response."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()
    original = "General agricultural response."

    res = await enrich_response_with_shops(db, "Where is the nearest shop?", original, farmer)
    assert res == original


@pytest.mark.asyncio
async def test_enrich_shops_db_failure_returns_original():
    """When database raises an exception, enrichment safely returns original response without crashing."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("DB connection error"))
    original = "Important farming advice."

    res = await enrich_response_with_shops(db, "Where can I buy urea?", original, farmer)
    assert res == original


def test_extract_district_from_query():
    """Extracts known districts from farmer queries in English and Telugu."""
    from src.shops.service import _extract_district_from_query

    assert _extract_district_from_query("Urea in Karimnagar") == "Karimnagar"
    assert _extract_district_from_query("యూరియా కరీంనగర్ లో ఎక్కడ దొరుకుతుంది?") == "Karimnagar"
    assert _extract_district_from_query("find fertilizer in warangal") == "Warangal"
    assert _extract_district_from_query("గుంటూరు లో వేప నూనె ఎక్కడ ఉంది?") == "Guntur"
    assert _extract_district_from_query("shops near vijayawada") == "Krishna"
    assert _extract_district_from_query("no location mentioned here") is None


def test_expanded_product_synonyms_telugu_and_english():
    """Expanded product trade names and Telugu terms normalize correctly for search."""
    assert _detect_product_from_query("నానో యూరియా ధర ఎంత?") == "urea"
    assert _detect_product_from_query("need nano urea bottle") == "urea"
    assert _detect_product_from_query("కలుపు మందు ఎక్కడ దొరుకుతుంది?") == "herbicide"
    assert _detect_product_from_query("best weedicide for cotton") == "herbicide"
    assert _detect_product_from_query("సాఫ్ శిలీంద్ర సంహారిణి కావాలి") == "mancozeb"
    assert _detect_product_from_query("looking for saaf fungicide") == "mancozeb"
    assert _detect_product_from_query("కోరజెన్ పురుగుల మందు ధర") == "coragen"
    assert _detect_product_from_query("where to buy coragen") == "coragen"
    assert _detect_product_from_query("రౌండప్ కలుపు మందు") == "herbicide"


@pytest.mark.asyncio
async def test_enrich_shops_query_district_override():
    """Query location takes precedence over farmer profile location."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    # Farmer profile is in Guntur, but farmer asks for Karimnagar in the query
    db = _mock_db_with_farmer_location(profile_district="Guntur", profile_state="Andhra Pradesh")

    guntur_shop = _make_mock_shop(name="Sri Lakshmi Agro Guntur", district="Guntur", lat=16.3067, lon=80.4365)
    karimnagar_shop = _make_mock_shop(name="Balaji Fertilizers Karimnagar", district="Karimnagar", lat=18.4386, lon=79.1288)
    item = _make_mock_item(name="Urea", price=295.0)

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(guntur_shop, item), (karimnagar_shop, item)]):

        res = await enrich_response_with_shops(db, "Where can I buy urea in Karimnagar?", "Farming advice.", farmer)

    assert "Balaji Fertilizers Karimnagar" in res
    # Karimnagar shop is returned, and Guntur shop is excluded due to query override
    assert "Sri Lakshmi Agro Guntur" not in res


@pytest.mark.asyncio
async def test_enrich_shops_all_out_of_stock_warning():
    """When all matching shops have zero stock or unavailable, a clear out-of-stock note is attached."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = _mock_db_with_farmer_location(profile_district="Warangal")

    shop = _make_mock_shop(name="Kisan Seva Warangal", district="Warangal")
    out_of_stock_item = _make_mock_item(name="Urea", price=295.0, stock=0, available=False)

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(shop, out_of_stock_item)]):

        res = await enrich_response_with_shops(db, "Where to buy urea in Warangal?", "Farming advice.", farmer)

    assert "Kisan Seva Warangal" in res
    assert "Out of Stock" in res
    assert "currently out of stock across nearby registered shops" in res
    assert "Prices and stock levels are subject to local dealer confirmation" in res


@pytest.mark.asyncio
async def test_enrich_shops_all_out_of_stock_telugu():
    """Telugu out-of-stock formatting displays authentic Telugu caution message."""
    farmer = MagicMock(id=uuid4(), preferred_language="te")
    db = _mock_db_with_farmer_location(profile_district="Warangal")

    shop = _make_mock_shop(name="కిసాన్ సేవా కేంద్రం", district="Warangal")
    out_of_stock_item = _make_mock_item(name="యూరియా", price=295.0, stock=0, available=True)

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(shop, out_of_stock_item)]):

        res = await enrich_response_with_shops(db, "యూరియా ఎక్కడ దొరుకుతుంది?", "సలహా.", farmer)

    assert "కిసాన్ సేవా కేంద్రం" in res
    assert "స్టాక్ లేదు" in res
    assert "ప్రస్తుతం సమీప నమోదిత దుకాణాలలో స్టాక్ అందుబాటులో లేదు" in res
    assert "ధరలు మరియు స్టాక్ వివరాలు స్థానిక డీలర్ నిర్ధారణకు లోబడి ఉంటాయి" in res


# ---------------------------------------------------------------------------
# Pilot Readiness Regression Tests (Task A: Out-of-District Fallback Guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_verified_shop_found_within_radius():
    """Regression Test 1: Local verified shop within safe radius is formatted with distance."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = _mock_db_with_farmer_location(memory_gps={"latitude": 17.9689, "longitude": 79.5941})

    local_shop = _make_mock_shop(name="Warangal Rythu Seva", lat=17.9750, lon=79.6000, district="Warangal")
    item = _make_mock_item(name="DAP", price=1350.0)

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(local_shop, item)]):

        res = await enrich_response_with_shops(db, "Where to buy DAP?", "Advice.", farmer)

    assert "Warangal Rythu Seva" in res
    assert "km away" in res
    assert "DAP" in res


@pytest.mark.asyncio
async def test_no_local_shop_safe_no_dealer_response_english():
    """Regression Test 2: When no dealer exists in farmer district, return safe no-dealer message in English."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    db = _mock_db_with_farmer_location(profile_district="Adilabad")

    guntur_shop = _make_mock_shop(name="Guntur Agro Center", district="Guntur", lat=16.3067, lon=80.4365)
    item = _make_mock_item(name="Urea")

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(guntur_shop, item)]):

        res = await enrich_response_with_shops(db, "Where to buy urea in Adilabad?", "Advice.", farmer)

    assert "No licensed dealer is currently registered in your mandal/district" in res
    assert "Guntur Agro Center" not in res


@pytest.mark.asyncio
async def test_no_local_shop_safe_no_dealer_response_telugu():
    """Regression Test 2 (Telugu): Safe no-dealer message rendered cleanly in Telugu."""
    farmer = MagicMock(id=uuid4(), preferred_language="te")
    db = _mock_db_with_farmer_location(profile_district="Nizamabad")

    guntur_shop = _make_mock_shop(name="Guntur Agro Center", district="Guntur", lat=16.3067, lon=80.4365)
    item = _make_mock_item(name="యూరియా")

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(guntur_shop, item)]):

        res = await enrich_response_with_shops(db, "యూరియా ఎక్కడ దొరుకుతుంది?", "సలహా.", farmer)

    assert "మీ మండలం/జిల్లాలో ఈ ఉత్పత్తికి సంబంధించి ప్రస్తుతం నమోదిత లైసెన్స్ డీలర్లు అందుబాటులో లేరు" in res
    assert "Guntur Agro Center" not in res


@pytest.mark.asyncio
async def test_distant_seeded_shop_not_shown_as_fallback():
    """Regression Test 3: Distant seeded shop (>50km away in another district) is NEVER shown as fallback."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    # Farmer in Warangal (GPS: 17.9689, 79.5941)
    db = _mock_db_with_farmer_location(memory_gps={"latitude": 17.9689, "longitude": 79.5941}, profile_district="Warangal")

    distant_shop = _make_mock_shop(name="Distant Vijayawada Agro", district="Krishna", lat=16.5062, lon=80.6480)
    item = _make_mock_item(name="Mancozeb")

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(distant_shop, item)]):

        res = await enrich_response_with_shops(db, "Where can I get mancozeb?", "Crop protection advice.", farmer)

    assert "Distant Vijayawada Agro" not in res
    assert "No licensed dealer is currently registered in your mandal/district" in res


@pytest.mark.asyncio
async def test_explicit_district_query_overrides_stored_location_regression():
    """Regression Test 4: Explicit district query overrides stored profile and matches the requested district."""
    farmer = MagicMock(id=uuid4(), preferred_language="en")
    # Farmer stored in Warangal
    db = _mock_db_with_farmer_location(profile_district="Warangal")

    khammam_shop = _make_mock_shop(name="Khammam Rythu Seva", district="Khammam", lat=17.2473, lon=80.1514)
    item = _make_mock_item(name="Urea")

    with patch.object(ShopRepository, "seed_default_shops_if_empty", new_callable=AsyncMock), \
         patch.object(ShopRepository, "search_shops_by_product", new_callable=AsyncMock, return_value=[(khammam_shop, item)]):

        # Farmer explicitly asks for Khammam
        res = await enrich_response_with_shops(db, "Find shops in Khammam selling urea", "Advice.", farmer)

    assert "Khammam Rythu Seva" in res
