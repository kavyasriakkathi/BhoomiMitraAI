"""
BhoomiMitra AI — Startup Pilot Analytics Test Suite

Tests:
- DAU / WAU calculation
- Modality breakdown (text vs audio vs image)
- Language breakdown (Telugu vs English)
- Escalation ticket counts
- WhatsApp delivery rate
- Admin RBAC enforcement (admin vs non-admin vs unauthenticated)
- Empty database behavior
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.core.models import UserAccount
from src.auth.dependencies import get_current_user, get_current_active_user
from src.analytics.service import AnalyticsService
from src.analytics.schemas import AnalyticsSummaryResponse, AnalyticsActivityResponse

client = TestClient(app)


# =====================================================================
# 1. Analytics Service Unit Tests
# =====================================================================

@pytest.mark.asyncio
async def test_analytics_service_empty_db():
    """When DB is empty, analytics service returns zeroed summary and 100% success rate."""
    mock_db = AsyncMock()

    # Total farmers = 0
    # DAU = 0
    # WAU = 0
    # Messages today = 0
    # Total messages = 0
    # Language breakdown = empty
    # Modality breakdown = empty
    # Delivery breakdown = empty

    mock_scalars = MagicMock()
    mock_scalars.scalar.return_value = 0
    mock_scalars.all.return_value = []

    mock_db.execute.return_value = mock_scalars

    with patch("src.escalation.repository.EscalationRepository.get_all_tickets", new_callable=AsyncMock, return_value=[]):
        service = AnalyticsService(mock_db)
        summary = await service.get_summary()

    assert summary.total_farmers == 0
    assert summary.dau == 0
    assert summary.wau == 0
    assert summary.messages_today == 0
    assert summary.total_messages == 0
    assert summary.languages.telugu == 0
    assert summary.languages.english == 0
    assert summary.modality.text == 0
    assert summary.modality.audio == 0
    assert summary.modality.image == 0
    assert summary.escalation.total == 0
    assert summary.delivery.sent == 0
    assert summary.delivery.failed == 0
    assert summary.delivery.success_rate_pct == 100.0


@pytest.mark.asyncio
async def test_analytics_service_summary_calculation():
    """Analytics service correctly computes DAU, WAU, modality, language, escalation, and delivery health."""
    mock_db = AsyncMock()

    # Sequence of mock DB executions:
    # 1. total_farmers -> 10
    # 2. dau -> 6
    # 3. wau -> 9
    # 4. messages_today -> 15
    # 5. total_messages -> 45
    # 6. languages -> [("te", 8), ("en", 2)]
    # 7. modality -> [("text", 25), ("audio", 15), ("image", 5)]
    # 8. delivery -> [("sent", 40), ("failed", 5)]

    def create_mock_result(scalar_val=None, all_val=None):
        m = MagicMock()
        m.scalar.return_value = scalar_val
        m.all.return_value = all_val or []
        return m

    mock_db.execute.side_effect = [
        create_mock_result(scalar_val=10),
        create_mock_result(scalar_val=6),
        create_mock_result(scalar_val=9),
        create_mock_result(scalar_val=15),
        create_mock_result(scalar_val=45),
        create_mock_result(all_val=[("te", 8), ("en", 2)]),
        create_mock_result(all_val=[("text", 25), ("audio", 15), ("image", 5)]),
        create_mock_result(all_val=[("sent", 40), ("failed", 5)]),
    ]

    mock_tickets = [
        {"ticket_id": "ESC-1", "status": "Pending"},
        {"ticket_id": "ESC-2", "status": "Assigned"},
        {"ticket_id": "ESC-3", "status": "Resolved"},
    ]

    with patch("src.escalation.repository.EscalationRepository.get_all_tickets", new_callable=AsyncMock, return_value=mock_tickets):
        service = AnalyticsService(mock_db)
        summary = await service.get_summary()

    assert summary.total_farmers == 10
    assert summary.dau == 6
    assert summary.wau == 9
    assert summary.messages_today == 15
    assert summary.total_messages == 45

    assert summary.languages.telugu == 8
    assert summary.languages.english == 2

    assert summary.modality.text == 25
    assert summary.modality.audio == 15
    assert summary.modality.image == 5

    assert summary.escalation.total == 3
    assert summary.escalation.pending == 2
    assert summary.escalation.resolved == 1

    assert summary.delivery.sent == 40
    assert summary.delivery.failed == 5
    # 40 / 45 * 100 = 88.89%
    assert summary.delivery.success_rate_pct == 88.89


@pytest.mark.asyncio
async def test_analytics_service_activity_time_series():
    """Analytics service generates daily time-series array."""
    mock_db = AsyncMock()

    # Mock daily conversations
    c1 = MagicMock(farmer_id=uuid4(), user_message_type="text", delivery_status="sent")
    c2 = MagicMock(farmer_id=uuid4(), user_message_type="audio", delivery_status="sent")
    c3 = MagicMock(farmer_id=c1.farmer_id, user_message_type="image", delivery_status="failed")

    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = [c1, c2, c3]
    mock_db.execute.return_value = mock_res

    service = AnalyticsService(mock_db)
    activity_res = await service.get_activity(days=5)

    assert activity_res.days == 5
    assert len(activity_res.activity) == 5

    item = activity_res.activity[0]
    assert item.active_farmers == 2
    assert item.message_count == 3
    assert item.text_count == 1
    assert item.audio_count == 1
    assert item.image_count == 1
    assert item.delivery_failures == 1


# =====================================================================
# 2. RBAC and Endpoint Integration Tests
# =====================================================================

def test_analytics_rbac_admin_allowed():
    """Admin user can successfully query /analytics/summary."""
    mock_admin = UserAccount(
        id=uuid4(),
        email="admin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_admin
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin

    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "dau" in data
    assert "wau" in data
    assert "languages" in data
    assert "modality" in data
    assert "escalation" in data
    assert "delivery" in data


def test_analytics_rbac_unauthenticated_forbidden():
    """Unauthenticated requests without user identity must return 401/403."""
    app.dependency_overrides.clear()
    response = client.get("/analytics/summary")
    assert response.status_code in [401, 403]


def test_analytics_rbac_non_admin_forbidden():
    """Shop owners and agricultural experts are forbidden from /analytics endpoints."""
    mock_shop = UserAccount(
        id=uuid4(),
        email="shop@bhoomimitra.ai",
        role="shop_owner",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_shop
    app.dependency_overrides[get_current_active_user] = lambda: mock_shop

    response = client.get("/analytics/summary")
    assert response.status_code == 403
    res_json = response.json()
    err_msg = res_json.get("error", {}).get("message", "") or res_json.get("detail", "")
    assert "Access forbidden" in err_msg


def test_analytics_activity_endpoint_admin_allowed():
    """Admin user can retrieve activity time-series via /analytics/activity."""
    mock_admin = UserAccount(
        id=uuid4(),
        email="admin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_admin
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin

    response = client.get("/analytics/activity?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["days"] == 7
    assert len(data["activity"]) == 7
