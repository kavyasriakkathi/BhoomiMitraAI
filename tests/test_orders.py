import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.orders.schemas import (
    OrderRequestResponse,
    PaginatedOrderRequestResponse,
    SalesAnalyticsResponse,
)
from src.orders.service import OrderService
from src.orders.dependencies import get_order_service

client = TestClient(app)
TS = "2026-07-28T12:30:00Z"


@pytest.fixture
def mock_order_service():
    service = AsyncMock(spec=OrderService)
    app.dependency_overrides[get_order_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_order_response(**kwargs) -> OrderRequestResponse:
    defaults = dict(
        id=uuid4(),
        farmer_id=uuid4(),
        shop_id=uuid4(),
        inventory_id=uuid4(),
        product_name="Urea",
        brand="IFFCO",
        unit="Bag",
        unit_price=295.0,
        quantity=5,
        total_price=1475.0,
        status="Pending",
        notes="Deliver to Guntur Main Road",
        created_at=TS,
        updated_at=TS,
    )
    defaults.update(kwargs)
    return OrderRequestResponse(**defaults)


def test_create_order_request(mock_order_service):
    order_resp = _mock_order_response()
    mock_order_service.create_order_request.return_value = order_resp

    payload = {
        "farmer_id": str(order_resp.farmer_id),
        "shop_id": str(order_resp.shop_id),
        "inventory_id": str(order_resp.inventory_id),
        "quantity": 5,
        "notes": "Deliver to Guntur Main Road",
    }
    response = client.post("/orders", json=payload)
    assert response.status_code == 201
    assert response.json()["product_name"] == "Urea"
    assert response.json()["quantity"] == 5
    assert response.json()["total_price"] == 1475.0


def test_get_order(mock_order_service):
    order_resp = _mock_order_response()
    mock_order_service.get_order_by_id.return_value = order_resp

    response = client.get(f"/orders/{order_resp.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(order_resp.id)


def test_update_order_status(mock_order_service):
    order_resp = _mock_order_response(status="Accepted")
    mock_order_service.update_status.return_value = order_resp

    response = client.patch(f"/orders/{order_resp.id}/status", json={"status": "Accepted"})
    assert response.status_code == 200
    assert response.json()["status"] == "Accepted"


def test_list_farmer_orders(mock_order_service):
    farmer_id = uuid4()
    order_resp = _mock_order_response(farmer_id=farmer_id)
    paginated = PaginatedOrderRequestResponse(items=[order_resp], total=1, page=1, size=20)
    mock_order_service.list_farmer_orders.return_value = paginated

    response = client.get(f"/orders/farmer/{farmer_id}")
    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_sales_analytics(mock_order_service):
    shop_id = uuid4()
    analytics = SalesAnalyticsResponse(
        shop_id=shop_id,
        total_orders=10,
        pending_orders=2,
        accepted_orders=3,
        ready_orders=1,
        completed_orders=4,
        cancelled_orders=0,
        total_revenue_inr=12500.0,
        popular_products=[{"product_name": "Urea", "units_sold": "50"}],
        category_demand=[{"category": "Fertilizers", "demand": "High"}],
    )
    mock_order_service.get_sales_analytics.return_value = analytics

    response = client.get(f"/orders/analytics/{shop_id}")
    assert response.status_code == 200
    assert response.json()["total_orders"] == 10
    assert response.json()["total_revenue_inr"] == 12500.0
