"""
BhoomiMitra AI — Startup Pilot Analytics Schemas
"""

from typing import List
from pydantic import BaseModel, Field


class LanguageBreakdown(BaseModel):
    telugu: int = Field(0, description="Count of Telugu-preferring farmers")
    english: int = Field(0, description="Count of English-preferring farmers")
    other: int = Field(0, description="Count of other language farmers")


class ModalityBreakdown(BaseModel):
    text: int = Field(0, description="Count of text queries")
    audio: int = Field(0, description="Count of voice audio queries")
    image: int = Field(0, description="Count of camera crop diagnosis queries")


class DeliveryStatusBreakdown(BaseModel):
    sent: int = Field(0, description="Successfully delivered messages")
    failed: int = Field(0, description="Failed outbound messages")
    pending: int = Field(0, description="Pending delivery messages")
    success_rate_pct: float = Field(100.0, description="Delivery success rate percentage")


class EscalationMetrics(BaseModel):
    total: int = Field(0, description="Total escalation tickets raised")
    pending: int = Field(0, description="Pending or active triage tickets")
    resolved: int = Field(0, description="Resolved or closed tickets")


class AnalyticsSummaryResponse(BaseModel):
    total_farmers: int = Field(0, description="Total registered farmers")
    dau: int = Field(0, description="Daily Active Farmers (past 24h)")
    wau: int = Field(0, description="Weekly Active Farmers (past 7d)")
    messages_today: int = Field(0, description="Total conversational messages received today")
    total_messages: int = Field(0, description="Total all-time messages")
    languages: LanguageBreakdown
    modality: ModalityBreakdown
    escalation: EscalationMetrics
    delivery: DeliveryStatusBreakdown


class DailyActivityItem(BaseModel):
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    active_farmers: int = Field(0, description="Distinct active farmers on this date")
    message_count: int = Field(0, description="Total messages on this date")
    text_count: int = Field(0, description="Text messages on this date")
    audio_count: int = Field(0, description="Audio voice notes on this date")
    image_count: int = Field(0, description="Image diagnosis on this date")
    delivery_failures: int = Field(0, description="Delivery failures on this date")


class AnalyticsActivityResponse(BaseModel):
    days: int = Field(7, description="Number of days covered in the time-series")
    activity: List[DailyActivityItem]
