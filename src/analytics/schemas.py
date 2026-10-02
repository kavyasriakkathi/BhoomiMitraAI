"""
BhoomiMitra AI — Startup Pilot Analytics Schemas
"""

from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class LanguageBreakdown(BaseModel):
    telugu: int = Field(0, description="Count of Telugu-preferring farmers")
    english: int = Field(0, description="Count of English-preferring farmers")
    other: int = Field(0, description="Count of other language farmers")
    details: Dict[str, int] = Field(default_factory=dict, description="Detailed count by language code")


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


class IntentBreakdown(BaseModel):
    shops: int = Field(0, description="Count of shop / input queries")
    fertilizer: int = Field(0, description="Count of fertilizer queries")
    weather: int = Field(0, description="Count of weather queries")
    market: int = Field(0, description="Count of market price queries")
    disease: int = Field(0, description="Count of crop disease / health queries")
    schemes: int = Field(0, description="Count of government schemes queries")
    other: int = Field(0, description="Count of other or unclassified queries")
    details: Dict[str, int] = Field(default_factory=dict, description="Raw intent mapping to counts")


class TopQuestionItem(BaseModel):
    question: str = Field(..., description="Question text")
    count: int = Field(0, description="Number of times asked")


class TopCropItem(BaseModel):
    crop: str = Field(..., description="Crop name")
    count: int = Field(0, description="Number of farmers/farms growing this crop")


class TopDistrictItem(BaseModel):
    district: str = Field(..., description="District/location name")
    count: int = Field(0, description="Number of farmers/farms located in this district")


class AnalyticsSummaryResponse(BaseModel):
    total_farmers: int = Field(0, description="Total registered farmers")
    total_users: int = Field(0, description="Total platform users (farmers + accounts)")
    total_messages: int = Field(0, description="Total all-time messages / questions")
    total_ai_responses: int = Field(0, description="Total AI responses generated")
    dau: int = Field(0, description="Daily Active Farmers (past 24h)")
    wau: int = Field(0, description="Weekly Active Farmers (past 7d)")
    active_users_today: int = Field(0, description="Active farmers today since midnight UTC")
    messages_today: int = Field(0, description="Total conversational messages received today")
    avg_response_time_seconds: float = Field(0.0, description="Average AI response time in seconds")
    slow_responses_count: int = Field(0, description="Count of slow responses (>10s)")
    languages: LanguageBreakdown
    modality: ModalityBreakdown
    escalation: EscalationMetrics
    delivery: DeliveryStatusBreakdown
    intents: IntentBreakdown = Field(default_factory=IntentBreakdown, description="Intent distribution breakdown")
    top_questions: List[TopQuestionItem] = Field(default_factory=list, description="Top frequently asked questions")
    top_crops: List[TopCropItem] = Field(default_factory=list, description="Top crops")
    top_districts: List[TopDistrictItem] = Field(default_factory=list, description="Top districts/locations")


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
    period: Optional[str] = Field(None, description="Period shorthand: day, week, month")
    activity: List[DailyActivityItem]


class ConversationAnalyticsItem(BaseModel):
    id: UUID = Field(..., description="Conversation UUID")
    farmer_id: UUID = Field(..., description="Farmer UUID")
    farmer_identifier: Optional[str] = Field(None, description="Farmer phone number or name identifier")
    received_time: datetime = Field(..., description="Timestamp when user message was received")
    replied_time: Optional[datetime] = Field(None, description="Timestamp when AI reply was sent")
    question: Optional[str] = Field(None, description="User question / message")
    ai_answer: Optional[str] = Field(None, description="AI response text")
    intent: Optional[str] = Field(None, description="Detected intent")
    modality: Optional[str] = Field("text", description="User message modality: text, audio, image")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    delivery_status: Optional[str] = Field("pending", description="Delivery status: sent, delivered, failed, pending")


class PaginatedConversationsResponse(BaseModel):
    total: int = Field(0, description="Total matching conversations")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Page size limit")
    total_pages: int = Field(0, description="Total number of pages")
    items: List[ConversationAnalyticsItem] = Field(default_factory=list, description="List of conversations")
