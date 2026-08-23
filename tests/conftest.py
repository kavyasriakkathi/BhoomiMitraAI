"""
Global pytest configuration and fixtures.
"""

from uuid import uuid4
import pytest
from src.auth.dependencies import get_current_active_user, get_current_user
from src.core.models import UserAccount
from src.main import app


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
