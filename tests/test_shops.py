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
