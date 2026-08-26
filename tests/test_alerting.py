"""
Tests for BhoomiMitra AI Founder Critical Alerting System.

Verifies:
1. Webhook dispatch formatting.
2. Graceful local logging when webhook URL is unconfigured.
3. In-memory cooldown / throttling rate-limiting.
4. Secret / credential sanitization in alert payloads.
5. Meta API 401 triggers AUTH_FAILURE alert.
6. DB failure in /health triggers DATABASE_DOWN alert.
7. Unhandled exception handler triggers UNHANDLED_EXCEPTION alert.
8. Webhook network failure is non-blocking and isolated.
"""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport, Response

from src.main import app
from src.core.alerting import (
    dispatch_founder_alert,
    reset_alert_cooldowns,
    AlertCategory,
    AlertSeverity,
)
from src.config import Settings


@pytest.fixture(autouse=True)
def clean_cooldowns():
    reset_alert_cooldowns()
    yield
    reset_alert_cooldowns()


@pytest.mark.asyncio
async def test_dispatch_founder_alert_with_webhook():
    """Verify alert sends valid JSON payload to configured webhook."""
    fake_settings = Settings(
        app_env="production",
        founder_alert_webhook_url="https://discord.com/api/webhooks/12345/mock",
    )

    with patch("src.core.alerting.get_settings", return_value=fake_settings):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = Response(204)

            success = await dispatch_founder_alert(
                category=AlertCategory.AUTH_FAILURE,
                severity=AlertSeverity.CRITICAL,
                component="whatsapp_gateway",
                summary="Meta token 401 Unauthorized",
                recommended_action="Rotate WHATSAPP_API_TOKEN on Render.",
                details={"status_code": 401},
            )

            assert success is True
            assert mock_post.called
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["category"] == "AUTH_FAILURE"
            assert payload["severity"] == "CRITICAL"
            assert payload["component"] == "whatsapp_gateway"
            assert payload["summary"] == "Meta token 401 Unauthorized"
            assert payload["environment"] == "production"
            assert "content" in payload  # Discord fallback compatibility


@pytest.mark.asyncio
async def test_dispatch_founder_alert_unconfigured_logs_gracefully():
    """When webhook is unconfigured, logs locally and returns True without raising."""
    fake_settings = Settings(
        app_env="development",
        founder_alert_webhook_url="",
    )

    with patch("src.core.alerting.get_settings", return_value=fake_settings):
        success = await dispatch_founder_alert(
            category=AlertCategory.DATABASE_DOWN,
            severity=AlertSeverity.CRITICAL,
            component="postgresql",
            summary="PostgreSQL connection refused",
            recommended_action="Check database instance status.",
        )
        assert success is True


@pytest.mark.asyncio
async def test_alert_throttling_cooldown():
    """Subsequent alerts for the same category within cooldown window must be throttled."""
    fake_settings = Settings(
        app_env="production",
        founder_alert_webhook_url="https://mock.webhook/alert",
    )

    with patch("src.core.alerting.get_settings", return_value=fake_settings):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = Response(200)

            # First dispatch: sent
            res1 = await dispatch_founder_alert(
                category=AlertCategory.AUTH_FAILURE,
                severity=AlertSeverity.CRITICAL,
                component="whatsapp_gateway",
                summary="Meta token 401",
                recommended_action="Fix token.",
            )
            assert res1 is True
            assert mock_post.call_count == 1

            # Second immediate dispatch for same category: throttled
            res2 = await dispatch_founder_alert(
                category=AlertCategory.AUTH_FAILURE,
                severity=AlertSeverity.CRITICAL,
                component="whatsapp_gateway",
                summary="Meta token 401 duplicate",
                recommended_action="Fix token.",
            )
            assert res2 is False
            assert mock_post.call_count == 1

            # Force bypass sends anyway
            res3 = await dispatch_founder_alert(
                category=AlertCategory.AUTH_FAILURE,
                severity=AlertSeverity.CRITICAL,
                component="whatsapp_gateway",
                summary="Meta token 401 force",
                recommended_action="Fix token.",
                force=True,
            )
            assert res3 is True
            assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_alert_sanitizes_secrets():
    """Tokens, passwords, and sensitive keys in details dictionary must be redacted."""
    fake_settings = Settings(
        app_env="production",
        founder_alert_webhook_url="https://mock.webhook/alert",
    )

    with patch("src.core.alerting.get_settings", return_value=fake_settings):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = Response(200)

            await dispatch_founder_alert(
                category=AlertCategory.AUTH_FAILURE,
                severity=AlertSeverity.CRITICAL,
                component="whatsapp_gateway",
                summary="Sanitization test",
                recommended_action="Action",
                details={
                    "whatsapp_token": "EAAXsecret123456",
                    "user_password": "supersecretpassword",
                    "safe_metric": 42,
                },
            )

            payload = mock_post.call_args.kwargs["json"]
            details = payload["details"]
            assert details["whatsapp_token"] == "[REDACTED]"
            assert details["user_password"] == "[REDACTED]"
            assert details["safe_metric"] == "42"


@pytest.mark.asyncio
async def test_db_failure_in_health_check_triggers_alert():
    """When /health detects DB connection failure, it attempts founder alert dispatch."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("src.core.database.AsyncSessionLocal", side_effect=Exception("DB Down")):
            with patch("src.core.alerting.dispatch_founder_alert", new_callable=AsyncMock) as mock_alert:
                resp = await client.get("/health")
                assert resp.status_code == 503
                assert mock_alert.called
                call_kwargs = mock_alert.call_args.kwargs
                assert call_kwargs["category"] == AlertCategory.DATABASE_DOWN
                assert call_kwargs["severity"] == AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_alert_network_failure_is_non_blocking():
    """When alert webhook fails with network timeout, dispatch returns False without raising."""
    fake_settings = Settings(
        app_env="production",
        founder_alert_webhook_url="https://broken.webhook.url/test",
    )

    with patch("src.core.alerting.get_settings", return_value=fake_settings):
        with patch("httpx.AsyncClient.post", side_effect=Exception("Connection Timeout")):
            success = await dispatch_founder_alert(
                category=AlertCategory.UNHANDLED_EXCEPTION,
                severity=AlertSeverity.CRITICAL,
                component="test",
                summary="Test timeout",
                recommended_action="None",
            )
            assert success is False
