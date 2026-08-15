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
async def test_data_deletion_page_serves_html():
    """Verify data deletion instructions route serves HTML with required email and instructions."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Test /data-deletion.html
        res1 = await client.get("/data-deletion.html")
        assert res1.status_code == 200
        assert "text/html" in res1.headers["content-type"]
        assert "Data Deletion Instructions" in res1.text
        assert "kavyasriakkathi@gmail.com" in res1.text

        # Test /data-deletion
        res2 = await client.get("/data-deletion")
        assert res2.status_code == 200
        assert "text/html" in res2.headers["content-type"]
        assert "Data Deletion Instructions" in res2.text

        # Test /static/data-deletion.html
        res3 = await client.get("/static/data-deletion.html")
        assert res3.status_code == 200
        assert "Data Deletion Instructions" in res3.text


