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
        assert data["status"] in ["healthy", "degraded"]
        assert data["database"] == "ok"
        assert "redis" in data

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

@pytest.mark.asyncio
async def test_privacy_policy_get_and_head_methods():
    """Verify GET /privacy-policy and HEAD /privacy-policy both return HTTP 200 for Meta crawler validation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # GET /privacy-policy
        res_get = await client.get("/privacy-policy")
        assert res_get.status_code == 200
        assert "text/html" in res_get.headers["content-type"]
        assert "BhoomiMitra AI - Privacy Policy" in res_get.text
        assert "kavyasriakkathi@gmail.com" in res_get.text

        # HEAD /privacy-policy (Meta Dashboard crawler check)
        res_head = await client.head("/privacy-policy")
        assert res_head.status_code == 200
        assert "text/html" in res_head.headers["content-type"]

@pytest.mark.asyncio
async def test_terms_get_and_head_methods():
    """Verify GET /terms and HEAD /terms both return HTTP 200."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # GET /terms — must return full HTML page with 200
        res_get = await client.get("/terms")
        assert res_get.status_code == 200, f"Expected 200, got {res_get.status_code}"
        assert "text/html" in res_get.headers["content-type"]
        assert "BhoomiMitra AI - Terms of Service" in res_get.text
        assert "Terms of Service" in res_get.text
        assert "kavyasriakkathi@gmail.com" in res_get.text

        # HEAD /terms — must return 200 with no body (Meta/WhatsApp crawler check)
        res_head = await client.head("/terms")
        assert res_head.status_code == 200, f"Expected 200 for HEAD, got {res_head.status_code}"
        assert "text/html" in res_head.headers["content-type"]


@pytest.mark.asyncio
async def test_dashboard_renders_auth_and_expert_ui():
    """Verify dashboard HTML contains the authentication modal, auth buttons, and expert views."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.get("/dashboard")
        assert res.status_code == 200
        html = res.text
        # Auth navbar elements
        assert "btn-nav-login" in html
        assert "user-profile-badge" in html
        assert "auth-controls" in html
        # Expert Dashboard view
        assert "view-expert" in html
        assert "Expert Dashboard" in html
        # Auth modal with login & registration forms
        assert "modal-auth" in html
        assert "form-auth-login" in html
        assert "form-auth-register" in html


@pytest.mark.asyncio
async def test_app_js_cookie_handling_security():
    """Verify frontend app.js does not store auth tokens in localStorage or sessionStorage."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.get("/static/app.js")
        assert res.status_code == 200
        js_code = res.text
        assert "localStorage.setItem('token'" not in js_code
        assert "localStorage.setItem('access_token'" not in js_code
        assert "sessionStorage.setItem('token'" not in js_code
        assert "credentials: 'include'" in js_code
        assert "checkAuthStatus" in js_code
        assert "handleAuthLogin" in js_code


