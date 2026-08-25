"""
BhoomiMitra AI — Startup Pilot Analytics Service

Computes high-priority operational and business metrics directly from existing PostgreSQL
tables (farmers, conversations, expert escalation tickets) without additional data pipelines.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from src.core.models import Farmer, Conversation
from src.escalation.repository import EscalationRepository
from src.analytics.schemas import (
    AnalyticsSummaryResponse,
    AnalyticsActivityResponse,
    DailyActivityItem,
    LanguageBreakdown,
    ModalityBreakdown,
    EscalationMetrics,
    DeliveryStatusBreakdown,
)
from src.core.logging import logger


class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_summary(self) -> AnalyticsSummaryResponse:
        """
        Compute high-level pilot KPI summary:
        - DAU / WAU
        - Message volume today & all-time
        - Language distribution (Telugu vs English)
        - Message modality (Text vs Audio vs Image)
        - Expert escalation tickets (Total, Pending, Resolved)
        - WhatsApp outbound delivery success rate
        """
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
        past_24h = now - timedelta(hours=24)
        past_7d = now - timedelta(days=7)

        # 1. Total Farmers
        total_farmers_res = await self.db.execute(select(func.count(Farmer.id)))
        total_farmers = total_farmers_res.scalar() or 0

        # 2. DAU (Unique farmers active in past 24h)
        dau_res = await self.db.execute(
            select(func.count(distinct(Conversation.farmer_id))).where(Conversation.created_at >= past_24h)
        )
        dau = dau_res.scalar() or 0

        # 3. WAU (Unique farmers active in past 7d)
        wau_res = await self.db.execute(
            select(func.count(distinct(Conversation.farmer_id))).where(Conversation.created_at >= past_7d)
        )
        wau = wau_res.scalar() or 0

        # 4. Messages Today
        msg_today_res = await self.db.execute(
            select(func.count(Conversation.id)).where(Conversation.created_at >= today_start)
        )
        messages_today = msg_today_res.scalar() or 0

        # Total Messages
        total_msg_res = await self.db.execute(select(func.count(Conversation.id)))
        total_messages = total_msg_res.scalar() or 0

        # 5. Language Breakdown (Farmers)
        lang_res = await self.db.execute(
            select(Farmer.preferred_language, func.count(Farmer.id)).group_by(Farmer.preferred_language)
        )
        lang_map = {row[0]: row[1] for row in lang_res.all()}
        languages = LanguageBreakdown(
            telugu=lang_map.get("te", 0),
            english=lang_map.get("en", 0),
            other=sum(v for k, v in lang_map.items() if k not in ["te", "en"]),
        )

        # 6. Modality Breakdown (Conversations)
        mod_res = await self.db.execute(
            select(Conversation.user_message_type, func.count(Conversation.id)).group_by(Conversation.user_message_type)
        )
        mod_map = {row[0]: row[1] for row in mod_res.all()}
        modality = ModalityBreakdown(
            text=mod_map.get("text", 0),
            audio=mod_map.get("audio", 0),
            image=mod_map.get("image", 0),
        )

        # 7. Escalation Metrics
        try:
            esc_repo = EscalationRepository(self.db)
            all_tickets = await esc_repo.get_all_tickets()
            total_esc = len(all_tickets)
            resolved_esc = len([t for t in all_tickets if str(t.get("status", "")).lower() in ["resolved", "closed"]])
            pending_esc = total_esc - resolved_esc
            escalation = EscalationMetrics(
                total=total_esc,
                pending=pending_esc,
                resolved=resolved_esc,
            )
        except Exception as esc_err:
            logger.warning(f"[ANALYTICS] Could not compute escalation metrics: {esc_err}")
            escalation = EscalationMetrics(total=0, pending=0, resolved=0)

        # 8. Delivery Status Breakdown
        del_res = await self.db.execute(
            select(Conversation.delivery_status, func.count(Conversation.id)).group_by(Conversation.delivery_status)
        )
        del_map = {row[0]: row[1] for row in del_res.all()}
        sent_count = del_map.get("sent", 0)
        failed_count = del_map.get("failed", 0)
        pending_del_count = del_map.get("pending", 0)
        total_delivered = sent_count + failed_count
        success_rate = (sent_count / total_delivered * 100.0) if total_delivered > 0 else 100.0

        delivery = DeliveryStatusBreakdown(
            sent=sent_count,
            failed=failed_count,
            pending=pending_del_count,
            success_rate_pct=round(success_rate, 2),
        )

        return AnalyticsSummaryResponse(
            total_farmers=total_farmers,
            dau=dau,
            wau=wau,
            messages_today=messages_today,
            total_messages=total_messages,
            languages=languages,
            modality=modality,
            escalation=escalation,
            delivery=delivery,
        )

    async def get_activity(self, days: int = 7) -> AnalyticsActivityResponse:
        """
        Compute daily active farmers and message modality trends over past N days.
        """
        days = min(max(1, days), 30)
        now = datetime.utcnow()
        activity_items = []

        for i in range(days - 1, -1, -1):
            day_date = (now - timedelta(days=i)).date()
            start_dt = datetime(day_date.year, day_date.month, day_date.day, 0, 0, 0)
            end_dt = datetime(day_date.year, day_date.month, day_date.day, 23, 59, 59)

            res = await self.db.execute(
                select(Conversation).where(
                    Conversation.created_at >= start_dt,
                    Conversation.created_at <= end_dt,
                )
            )
            convs = res.scalars().all()

            distinct_farmers = len(set(c.farmer_id for c in convs if c.farmer_id))
            total_msg = len(convs)
            text_cnt = len([c for c in convs if c.user_message_type == "text"])
            audio_cnt = len([c for c in convs if c.user_message_type == "audio"])
            image_cnt = len([c for c in convs if c.user_message_type == "image"])
            failed_cnt = len([c for c in convs if c.delivery_status == "failed"])

            activity_items.append(
                DailyActivityItem(
                    date=day_date.isoformat(),
                    active_farmers=distinct_farmers,
                    message_count=total_msg,
                    text_count=text_cnt,
                    audio_count=audio_cnt,
                    image_count=image_cnt,
                    delivery_failures=failed_cnt,
                )
            )

        return AnalyticsActivityResponse(days=days, activity=activity_items)
