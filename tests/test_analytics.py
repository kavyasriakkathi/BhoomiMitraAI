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
- Extended pilot KPI calculations (AI responses, response time, slow count, intent categorization)
- Daily activity periods ('day', 'week', 'month')
- Paginated conversation list analytics and filters
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.core.models import UserAccount, Conversation, Farmer
from src.core.database import Base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.auth.dependencies import get_current_user, get_current_active_user
from src.analytics.service import AnalyticsService, classify_intent
from src.analytics.schemas import (
    AnalyticsSummaryResponse,
    AnalyticsActivityResponse,
    PaginatedConversationsResponse,
)

client = TestClient(app)


# =====================================================================
# 1. Analytics Service Unit Tests
# =====================================================================

@pytest.mark.asyncio
async def test_analytics_service_empty_db():
    """When DB is empty, analytics service returns zeroed summary and 100% success rate."""
    mock_db = AsyncMock()

    mock_scalars = MagicMock()
    mock_scalars.scalar.return_value = 0
    mock_scalars.all.return_value = []
    mock_scalars.first.return_value = (0.0, 0)

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
    assert summary.delivery.success_rate_pct == 88.89


@pytest.mark.asyncio
async def test_analytics_service_extended_kpi_calculations():
    """Verifies extended calculations: AI responses, response times, intent breakdown, top questions, crops, districts."""
    mock_db = AsyncMock()

    def create_mock_result(scalar_val=None, all_val=None, first_val=None):
        m = MagicMock()
        m.scalar.return_value = scalar_val
        m.all.return_value = all_val or []
        m.first.return_value = first_val
        return m

    # Mock DB queries:
    # 1. total_farmers -> 20
    # 2. dau -> 12
    # 3. wau -> 18
    # 4. messages_today -> 30
    # 5. total_messages -> 100
    # 6. languages -> [("te", 16), ("en", 4)]
    # 7. modality -> [("text", 60), ("audio", 30), ("image", 10)]
    # 8. delivery -> [("sent", 95), ("failed", 5)]
    # 9. user accounts -> 2
    # 10. total_ai_responses -> 98
    # 11. active_users_today -> 10
    # 12. response time metrics (avg=3.45, slow_count=4)
    # 13. intents -> [("shops_input_availability", 15), ("fertilizer_nutrient", 20), ("weather_forecast", 10), ("market_price", 12), ("crop_health", 8), ("government_schemes", 5), ("greeting", 30)]
    # 14. top questions -> [("Urea price?", 15), ("Weather forecast?", 10)]
    # 15. crop rows -> [("Paddy", 12), ("Cotton", 8)]
    # 16. profile crop rows -> [("Chilli", 5)]
    # 17. profile districts -> [("Warangal", 10)]
    # 18. farm districts -> [("Karimnagar", 8), ("Warangal", 4)]

    mock_db.execute.side_effect = [
        create_mock_result(scalar_val=20),
        create_mock_result(scalar_val=12),
        create_mock_result(scalar_val=18),
        create_mock_result(scalar_val=30),
        create_mock_result(scalar_val=100),
        create_mock_result(all_val=[("te", 16), ("en", 4)]),
        create_mock_result(all_val=[("text", 60), ("audio", 30), ("image", 10)]),
        create_mock_result(all_val=[("sent", 95), ("failed", 5)]),
        create_mock_result(scalar_val=2),
        create_mock_result(scalar_val=98),
        create_mock_result(scalar_val=10),
        create_mock_result(first_val=(3.45, 4)),
        create_mock_result(all_val=[
            ("shops_input_availability", 15),
            ("fertilizer_nutrient", 20),
            ("weather_forecast", 10),
            ("market_price", 12),
            ("crop_health", 8),
            ("government_schemes", 5),
            ("greeting", 30),
        ]),
        create_mock_result(all_val=[("Urea price?", 15), ("Weather forecast?", 10)]),
        create_mock_result(all_val=[("Paddy", 12), ("Cotton", 8)]),
        create_mock_result(all_val=[("Chilli", 5)]),
        create_mock_result(all_val=[("Warangal", 10)]),
        create_mock_result(all_val=[("Karimnagar", 8), ("Warangal", 4)]),
    ]

    with patch("src.escalation.repository.EscalationRepository.get_all_tickets", new_callable=AsyncMock, return_value=[]):
        service = AnalyticsService(mock_db)
        summary = await service.get_summary()

    # Core volume KPIs
    assert summary.total_farmers == 20
    assert summary.total_users == 22
    assert summary.total_messages == 100
    assert summary.total_ai_responses == 98
    assert summary.dau == 12
    assert summary.active_users_today == 10
    assert summary.messages_today == 30

    # Response time KPIs
    assert summary.avg_response_time_seconds == 3.45
    assert summary.slow_responses_count == 4

    # Intent Breakdown
    assert summary.intents.shops == 15
    assert summary.intents.fertilizer == 20
    assert summary.intents.weather == 10
    assert summary.intents.market == 12
    assert summary.intents.disease == 8
    assert summary.intents.schemes == 5
    assert summary.intents.other == 30

    # Rankings
    assert len(summary.top_questions) == 2
    assert summary.top_questions[0].question == "Urea price?"
    assert summary.top_questions[0].count == 15

    assert len(summary.top_crops) == 3
    assert summary.top_crops[0].crop == "Paddy"

    assert len(summary.top_districts) == 2
    # Warangal was 10 + 4 = 14, Karimnagar was 8
    assert summary.top_districts[0].district == "Warangal"
    assert summary.top_districts[0].count == 14


def test_classify_intent_function():
    """Unit test for intent classification into shops, fertilizer, weather, market, disease, schemes, other."""
    assert classify_intent("shops_input_availability") == "shops"
    assert classify_intent("check_dealer_stock") == "shops"
    assert classify_intent("fertilizer_nutrient") == "fertilizer"
    assert classify_intent("urea_application") == "fertilizer"
    assert classify_intent("weather_forecast") == "weather"
    assert classify_intent("market_price") == "market"
    assert classify_intent("crop_health") == "disease"
    assert classify_intent("pest_identification") == "disease"
    assert classify_intent("government_schemes") == "schemes"
    assert classify_intent("pm_kisan_subsidy") == "schemes"
    assert classify_intent("greeting") == "other"
    assert classify_intent("order_status_notification") == "other"
    assert classify_intent(None) == "other"


@pytest.mark.asyncio
async def test_analytics_service_activity_time_series():
    """Analytics service generates daily time-series array."""
    mock_db = AsyncMock()

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


@pytest.mark.asyncio
async def test_analytics_service_activity_period_shorthands():
    """Tests day, week, and month period shorthand parameters."""
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    service = AnalyticsService(mock_db)

    # day -> 1
    res_day = await service.get_activity(period="day")
    assert res_day.days == 1
    assert len(res_day.activity) == 1

    # week -> 7
    res_week = await service.get_activity(period="week")
    assert res_week.days == 7
    assert len(res_week.activity) == 7

    # month -> 30
    res_month = await service.get_activity(period="month")
    assert res_month.days == 30
    assert len(res_month.activity) == 30


@pytest.mark.asyncio
async def test_analytics_service_paginated_conversations():
    """Tests paginated conversation queries, structure, and filtering in AnalyticsService."""
    mock_db = AsyncMock()

    c_id = uuid4()
    f_id = uuid4()
    now = datetime.utcnow()

    mock_conv = MagicMock()
    mock_conv.id = c_id
    mock_conv.farmer_id = f_id
    mock_conv.farmer_identifier = "+919876543210"
    mock_conv.created_at = now
    mock_conv.replied_at = now + timedelta(seconds=2)
    mock_conv.user_message = "What is the price of DAP?"
    mock_conv.ai_response = "DAP price is ₹1350."
    mock_conv.intent = "fertilizer_nutrient"
    mock_conv.user_message_type = "text"
    mock_conv.response_time_seconds = 2.15
    mock_conv.delivery_status = "sent"

    mock_count = MagicMock()
    mock_count.scalar.return_value = 45

    mock_items = MagicMock()
    mock_items.scalars.return_value.all.return_value = [mock_conv]

    mock_db.execute.side_effect = [mock_count, mock_items]

    service = AnalyticsService(mock_db)
    result = await service.get_conversations(page=2, page_size=10, intent="fertilizer_nutrient")

    assert result.total == 45
    assert result.page == 2
    assert result.page_size == 10
    assert result.total_pages == 5
    assert len(result.items) == 1

    item = result.items[0]
    assert item.id == c_id
    assert item.farmer_id == f_id
    assert item.farmer_identifier == "+919876543210"
    assert item.question == "What is the price of DAP?"
    assert item.ai_answer == "DAP price is ₹1350."
    assert item.intent == "fertilizer_nutrient"
    assert item.modality == "text"
    assert item.response_time == 2.15
    assert item.delivery_status == "sent"


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
    assert "total_farmers" in data
    assert "total_messages" in data
    assert "total_ai_responses" in data
    assert "intents" in data
    assert "top_questions" in data
    assert "top_crops" in data
    assert "top_districts" in data


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


def test_analytics_activity_endpoint_period_parameter():
    """Admin can specify period shorthand ('day', 'week', 'month')."""
    mock_admin = UserAccount(
        id=uuid4(),
        email="admin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_admin
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin

    response = client.get("/analytics/activity?period=week")
    assert response.status_code == 200
    data = response.json()
    assert data["days"] == 7
    assert data["period"] == "week"
    assert len(data["activity"]) == 7


def test_analytics_conversations_endpoint_admin_allowed():
    """Admin user can retrieve paginated conversation history via /analytics/conversations."""
    mock_admin = UserAccount(
        id=uuid4(),
        email="admin@bhoomimitra.ai",
        role="admin",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_admin
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin

    response = client.get("/analytics/conversations?page=1&page_size=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    if len(data["items"]) > 0:
        item = data["items"][0]
        assert "received_time" in item
        assert "replied_time" in item
        assert "farmer_identifier" in item
        assert "question" in item
        assert "ai_answer" in item
        assert "intent" in item
        assert "modality" in item
        assert "response_time" in item
        assert "delivery_status" in item


def test_analytics_conversations_endpoint_non_admin_forbidden():
    """Non-admin users cannot access /analytics/conversations."""
    mock_expert = UserAccount(
        id=uuid4(),
        email="expert@bhoomimitra.ai",
        role="expert",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_expert
    app.dependency_overrides[get_current_active_user] = lambda: mock_expert

    response = client.get("/analytics/conversations")
    assert response.status_code == 403


# =====================================================================
# 3. Top Questions Data Quality & Regression Tests
# =====================================================================

@pytest.mark.asyncio
async def test_analytics_top_questions_filters_automated_and_spam_messages():
    """
    Regression test verifying data-quality filters for Top Questions:
    Automated and non-farmer messages are excluded:
    - Conversation.intent == 'order_status_notification'
    - user_message starting with '[Automated'
    - messages containing 'Thank you for using LIC'
    - messages containing 'LIC' + 'WhatsApp'
    - messages containing 'Students Laptop Scheme'
    Genuine farmer questions continue to appear normally.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        farmer = Farmer(id=uuid4(), phone_number="+919876543210", preferred_language="te")
        session.add(farmer)
        await session.flush()

        convs = [
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="auto_ord_1",
                user_message="[Automated Order Notification: Accepted]",
                intent="order_status_notification",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="auto_alert_1",
                user_message="[Automated Stock Alert: Low Fertilizer]",
                intent="stock_alert",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="ord_status_custom",
                user_message="Your order status has changed to confirmed",
                intent="order_status_notification",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="lic_thankyou",
                user_message="*Thank you for using LIC's WhatsApp Services* info",
                intent="greeting",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="lic_whatsapp_combo",
                user_message="Check your LIC policy on WhatsApp now",
                intent="other",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="laptop_scheme",
                user_message="*The Applications for the Students Laptop Scheme 2026 Is Available*",
                intent="government_schemes",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="farmer_q1",
                user_message="వరి పంటలో పిలకలు బాగా రావడానికి ఏం చేయాలి?",
                intent="general_farming",
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="farmer_q2",
                user_message="What is a good fertilizer for cotton?",
                intent=None,
            ),
            Conversation(
                id=uuid4(),
                farmer_id=farmer.id,
                message_id="farmer_q3",
                user_message="Where can I buy urea?",
                intent="shops_input_availability",
            ),
        ]
        session.add_all(convs)
        await session.commit()

        service = AnalyticsService(session)
        summary = await service.get_summary()

        q_list = [tq.question for tq in summary.top_questions]

        # Verify automated and non-farmer spam messages are completely excluded
        assert "[Automated Order Notification: Accepted]" not in q_list
        assert "[Automated Stock Alert: Low Fertilizer]" not in q_list
        assert "Your order status has changed to confirmed" not in q_list
        assert "*Thank you for using LIC's WhatsApp Services* info" not in q_list
        assert "Check your LIC policy on WhatsApp now" not in q_list
        assert "*The Applications for the Students Laptop Scheme 2026 Is Available*" not in q_list

        # Verify genuine farmer questions appear in Top Questions
        assert "వరి పంటలో పిలకలు బాగా రావడానికి ఏం చేయాలి?" in q_list
        assert "What is a good fertilizer for cotton?" in q_list
        assert "Where can I buy urea?" in q_list
        assert len(summary.top_questions) == 3

