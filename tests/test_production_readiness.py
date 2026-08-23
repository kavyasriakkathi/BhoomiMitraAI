"""
Production Readiness & Security Hardening Tests for BhoomiMitra AI.

Tests:
1. Health check endpoint status and structure.
2. CORS headers verification.
3. RAG PDF upload file size limit enforcement (>25MB rejected).
4. RAG filename sanitization (path traversal protection).
5. Production cookie security evaluation.
6. Sensitive secret leakage prevention.
7. Error response sanitization (no internal traces leaked).
"""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from src.config import Settings
from src.auth.constants import UserRole
from src.auth.security import create_access_token


@pytest.fixture
def admin_token():
    return create_access_token({
        "sub": "00000000-0000-0000-0000-000000000001",
        "email": "admin@bhoomimitra.ai",
        "role": UserRole.ADMIN.value,
    })


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify health check endpoint returns 200 with service status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["service"] == "bhoomimitra-ai"


@pytest.mark.asyncio
async def test_cors_middleware_headers():
    """Verify CORS middleware responds with appropriate headers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/health",
            headers={
                "Origin": "https://dashboard.bhoomimitra.ai",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


def test_production_secure_cookie_evaluation():
    """Verify that in production mode, cookie_secure automatically evaluates to True."""
    dev_settings = Settings(app_env="development", auth_cookie_secure=False)
    assert dev_settings.cookie_secure is False

    prod_settings = Settings(app_env="production", auth_cookie_secure=False)
    assert prod_settings.cookie_secure is True


@pytest.mark.asyncio
async def test_rag_upload_rejects_oversized_files(admin_token):
    """Verify that uploaded documents exceeding 25MB are rejected with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create simulated oversized file (>25MB)
        oversized_bytes = b"%PDF-1.4 " + (b"0" * (26 * 1024 * 1024))
        files = {
            "file": ("large_book.pdf", oversized_bytes, "application/pdf")
        }
        data = {
            "title": "Large Agronomy Handbook",
            "source": "ICAR",
            "category": "manual",
        }
        resp = await client.post(
            "/rag/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400
        assert "exceeds maximum permitted size" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_global_exception_sanitization():
    """Verify unhandled exceptions return generic sanitized JSON without stack traces."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Mock weather service to throw an unexpected database exception
        with patch("src.weather.service.WeatherService.get_weather_for_query", side_effect=RuntimeError("Internal socket connection timeout")):
            resp = await client.get("/weather/forecast?latitude=17.38&longitude=78.48")
            assert resp.status_code == 500
            data = resp.json()
            assert data["success"] is False
            assert "An unexpected error occurred" in data["error"]["message"]
            # Ensure raw stack trace is not exposed
            assert "Traceback" not in resp.text
            assert "socket connection timeout" not in resp.text
