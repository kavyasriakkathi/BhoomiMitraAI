import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest.mark.asyncio
async def test_root_route_serves_unified_dashboard():
    """Verify default route '/' serves the unified HTML dashboard template."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "BhoomiMitra AI" in response.text
        assert "Farmer Dashboard" in response.text
        assert "Shop Owner Dashboard" in response.text

@pytest.mark.asyncio
async def test_swagger_docs_route_exists():
    """Verify developer Swagger UI is accessible at /docs."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify health endpoint is working."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["service"] == "bhoomimitra-ai"

@pytest.mark.asyncio
async def test_data_deletion_get_and_head_methods():
    """Verify GET /data-deletion and HEAD /data-deletion both return HTTP 200 for Meta crawler validation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # GET /data-deletion
        res_get = await client.get("/data-deletion")
        assert res_get.status_code == 200
        assert "text/html" in res_get.headers["content-type"]
        assert "BhoomiMitra AI - User Data Deletion Instructions" in res_get.text
        assert "kavyasriakkathi@gmail.com" in res_get.text

        # HEAD /data-deletion (Meta Dashboard crawler check)
        res_head = await client.head("/data-deletion")
        assert res_head.status_code == 200
        assert "text/html" in res_head.headers["content-type"]

        # GET /data-deletion.html
        res_get_html = await client.get("/data-deletion.html")
        assert res_get_html.status_code == 200
        assert "text/html" in res_get_html.headers["content-type"]

        # HEAD /data-deletion.html
        res_head_html = await client.head("/data-deletion.html")
        assert res_head_html.status_code == 200

        # Static fallback
        res_static = await client.get("/static/data-deletion.html")
        assert res_static.status_code == 200



