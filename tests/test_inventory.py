import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from uuid import uuid4
from src.main import app
from src.inventory.schemas import (
    InventoryResponse,
    ShopDashboardSummaryResponse,
    ProductCategoryEnum,
)
from src.inventory.service import InventoryService
from src.inventory.dependencies import get_inventory_service

client = TestClient(app)
TS = "2026-07-28T12:00:00Z"


@pytest.fixture
def mock_inventory_service():
    service = AsyncMock(spec=InventoryService)
    app.dependency_overrides[get_inventory_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def _mock_inventory_response(**kwargs) -> InventoryResponse:
    defaults = dict(
        id=uuid4(),
        shop_id=uuid4(),
        product_name="Urea",
        category="Fertilizers",
        brand="IFFCO",
        product_description="Neem Coated Urea 45kg bag",
        unit="Bag",
        price=295.0,
        discount_price=None,
        quantity_in_stock=50,
        minimum_stock_level=10,
        available=True,
        expiry_date=None,
        last_updated=TS,
        created_at=TS,
        updated_at=TS,
    )
    defaults.update(kwargs)
    return InventoryResponse(**defaults)


def test_add_product(mock_inventory_service):
    item_resp = _mock_inventory_response()
    mock_inventory_service.add_product.return_value = item_resp

    payload = {
        "shop_id": str(item_resp.shop_id),
        "product_name": "Urea",
        "category": "Fertilizers",
        "brand": "IFFCO",
        "unit": "Bag",
        "price": 295.0,
        "quantity_in_stock": 50,
        "minimum_stock_level": 10,
    }
    response = client.post("/inventory", json=payload)
    assert response.status_code == 201
    assert response.json()["product_name"] == "Urea"
    assert response.json()["price"] == 295.0


def test_get_product(mock_inventory_service):
    item_resp = _mock_inventory_response()
    mock_inventory_service.get_product_by_id.return_value = item_resp

    response = client.get(f"/inventory/{item_resp.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(item_resp.id)


def test_update_stock(mock_inventory_service):
    item_resp = _mock_inventory_response(quantity_in_stock=100)
    mock_inventory_service.update_stock.return_value = item_resp

    response = client.patch(f"/inventory/{item_resp.id}/stock", json={"quantity_in_stock": 100})
    assert response.status_code == 200
    assert response.json()["quantity_in_stock"] == 100


def test_delete_product(mock_inventory_service):
    mock_inventory_service.delete_product.return_value = None
    item_id = uuid4()
    response = client.delete(f"/inventory/{item_id}")
    assert response.status_code == 204


def test_shop_dashboard(mock_inventory_service):
    shop_id = uuid4()
    low_item = _mock_inventory_response(product_name="DAP", quantity_in_stock=2, minimum_stock_level=5)
    dashboard_resp = ShopDashboardSummaryResponse(
        shop_id=shop_id,
        total_products=4,
        available_products_count=3,
        low_stock_count=1,
        out_of_stock_count=1,
        low_stock_items=[low_item],
        out_of_stock_items=[],
    )
    mock_inventory_service.get_dashboard_summary.return_value = dashboard_resp

    response = client.get(f"/inventory/dashboard/{shop_id}")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["total_products"] == 4
    assert res_data["low_stock_count"] == 1
    assert len(res_data["low_stock_items"]) == 1


def test_product_categories_enum():
    categories = [c.value for c in ProductCategoryEnum]
    assert "Seeds" in categories
    assert "Fertilizers" in categories
    assert "Pesticides" in categories
    assert "Fungicides" in categories
    assert "Herbicides" in categories
    assert "Micronutrients" in categories
    assert "Bio Fertilizers" in categories
    assert "Organic Products" in categories
    assert "Farm Equipment" in categories
    assert "Irrigation" in categories
    assert "Animal Feed" in categories
    assert "Veterinary Medicines" in categories
    assert len(categories) == 12
