"""
Global pytest configuration and fixtures.
"""

from uuid import uuid4
import unittest.mock
import pytest
from src.auth.dependencies import get_current_active_user, get_current_user
from src.core.models import UserAccount
from src.main import app

# Ensure synchronous SQLAlchemy Session/Result methods on AsyncMock instances return MagicMock
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
    Provides default admin user dependency override for test isolation
    so existing module tests (shops, inventory, orders, escalation) can run smoothly.
    Tests in test_auth.py explicitly clear or set specific role overrides.
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
    against real aiosqlite connections during test execution, avoiding
    thread/event-loop teardown warnings when pytest closes the loop.
    """
    if "test_trigger_background_memory_extraction" in request.node.name:
        return
    mock_trigger = unittest.mock.MagicMock(return_value=None)
    monkeypatch.setattr("src.memory.service.trigger_background_memory_extraction", mock_trigger)
