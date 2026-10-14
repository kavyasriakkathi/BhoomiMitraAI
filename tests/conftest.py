"""
Global pytest configuration and fixtures.
"""

import hashlib
import hmac
import json
import unittest.mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.auth.dependencies import get_current_active_user, get_current_user
from src.core.models import UserAccount
from src.main import app


# Ensure synchronous SQLAlchemy Session/Result methods on AsyncMock
# instances return MagicMock.
_orig_get_child_mock = unittest.mock.AsyncMock._get_child_mock

_SYNC_METHOD_NAMES = {
    "add",
    "add_all",
    "delete",
    "expunge",
    "expunge_all",
    "is_modified",
    "in_transaction",
    "scalars",
    "scalar_one_or_none",
    "scalar",
    "all",
    "first",
    "one",
    "one_or_none",
    "reverse",
}


def _custom_get_child_mock(self, /, **kw):
    name = kw.get("_mock_new_name") or kw.get("name")

    if name in _SYNC_METHOD_NAMES:
        return unittest.mock.MagicMock(**kw)

    return _orig_get_child_mock(self, **kw)


unittest.mock.AsyncMock._get_child_mock = _custom_get_child_mock


@pytest.fixture(autouse=True)
def default_auth_override():
    """
    Provides default admin user dependency override for test isolation.
    """
    mock_admin = UserAccount(
        id=uuid4(),
        email="admin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )

    app.dependency_overrides[get_current_user] = lambda: mock_admin
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin

    yield

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_active_user, None)


@pytest.fixture(autouse=True)
def mock_background_memory_extraction(monkeypatch, request):
    """
    Prevent untracked background memory extraction tasks from running
    against real aiosqlite connections during test execution.
    """
    if "test_trigger_background_memory_extraction" in request.node.name:
        return

    mock_trigger = unittest.mock.MagicMock(return_value=None)

    monkeypatch.setattr(
        "src.memory.service.trigger_background_memory_extraction",
        mock_trigger,
    )


# WhatsApp webhook test client with automatic HMAC signature.
TEST_WHATSAPP_APP_SECRET = "test_whatsapp_app_secret"


@pytest.fixture(autouse=True)
def webhook_test_settings(monkeypatch):
    """
    Ensure WHATSAPP_APP_SECRET matches TEST_WHATSAPP_APP_SECRET in tests
    and clear the get_settings LRU cache so the environment variable is picked up.
    """
    monkeypatch.setenv(
        "WHATSAPP_APP_SECRET",
        TEST_WHATSAPP_APP_SECRET,
    )
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "path/to/service-account.json",
    )

    from src.config import get_settings

    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


class SignedWebhookTestClient(TestClient):

    def post(self, url, *args, **kwargs):
        if url == "/webhook/whatsapp":
            headers = kwargs.get("headers")
            if headers is None:
                headers = {}
                kwargs["headers"] = headers
            elif isinstance(headers, dict):
                headers = dict(headers)
                kwargs["headers"] = headers

            has_sig = any(k.lower() == "x-hub-signature-256" for k in headers.keys()) if isinstance(headers, dict) else False

            if not has_sig and isinstance(headers, dict):
                body = None
                if "content" in kwargs:
                    content = kwargs["content"]
                    if isinstance(content, str):
                        body = content.encode("utf-8")
                    elif isinstance(content, bytes):
                        body = content
                elif "json" in kwargs:
                    payload = kwargs["json"]
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    kwargs["content"] = body
                    kwargs.pop("json")
                    if "Content-Type" not in headers and "content-type" not in headers:
                        headers["Content-Type"] = "application/json"
                elif "data" in kwargs:
                    data = kwargs["data"]
                    if isinstance(data, str):
                        body = data.encode("utf-8")
                    elif isinstance(data, bytes):
                        body = data

                if body is not None:
                    signature = hmac.new(
                        TEST_WHATSAPP_APP_SECRET.encode("utf-8"),
                        body,
                        hashlib.sha256,
                    ).hexdigest()
                    headers["X-Hub-Signature-256"] = f"sha256={signature}"

        return super().post(url, *args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Ensure all database tables are created for tests on the test database
    and dispose the engine cleanly on teardown.
    """
    import asyncio
    from src.config import get_settings
    from src.core.database import Base, engine

    settings = get_settings()
    if settings.app_env == "test":
        if engine.url.get_backend_name() != "sqlite":
            pytest.exit(
                "SAFETY CHECK FAILED: APP_ENV is 'test' but configured database backend is not SQLite. "
                "Halting test run safely before any database connection or modification.",
                returncode=1,
            )

        async def _init_models():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()

        asyncio.run(_init_models())

        yield

        async def _cleanup():
            await engine.dispose()

        try:
            asyncio.run(_cleanup())
        except Exception:
            pass
    else:
        yield
