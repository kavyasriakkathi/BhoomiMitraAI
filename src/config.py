"""
KrishiMitra AI — Configuration Module

Loads and validates all environment variables using Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "bhoomimitra-ai"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./dev.db")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # WhatsApp
    whatsapp_verify_token: str = Field(default="")
    whatsapp_api_token: str = Field(default="")
    whatsapp_app_secret: str = Field(default="")  # Used for HMAC-SHA256 signature verification
    whatsapp_phone_number_id: str = Field(default="")
    whatsapp_business_account_id: str = Field(default="")
    whatsapp_api_timeout_seconds: float = Field(default=15.0)
    max_media_download_bytes: int = Field(default=15_728_640)  # 15 MB boundary guard

    # AI
    openai_api_key: str = Field(default="")
    google_gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash")
    gemini_api_timeout_seconds: float = Field(default=5.0)
    ai_confidence_threshold: float = Field(default=0.75)

    # Google Cloud
    google_cloud_project_id: str = Field(default="")
    google_application_credentials: str = Field(default="")

    # Speech-to-Text
    stt_provider: str = Field(default="google")
    stt_default_language: str = Field(default="te-IN")
    stt_api_timeout_seconds: float = Field(default=10.0)

    # Expert Escalation
    expert_whatsapp_group_id: str = Field(default="")
    escalation_timeout_minutes: int = Field(default=30)

    # Market Prices (Agmarknet / data.gov.in)
    # DATA_GOV_API_KEY is OPTIONAL. If empty, the feature uses only the local DB fallback.
    data_gov_api_key: str = Field(default="")
    agmarknet_api_url: str = Field(
        default="https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    )
    agmarknet_api_timeout_seconds: float = Field(default=5.0)
    market_price_cache_ttl_seconds: int = Field(default=21600)  # 6 hours

    # Weather (OpenWeatherMap)
    # OPENWEATHER_API_KEY is OPTIONAL. If empty, the client returns mock weather forecasts.
    openweather_api_key: str = Field(default="")
    openweather_api_url: str = Field(
        default="https://api.openweathermap.org/data/2.5/forecast"
    )
    openweather_api_timeout_seconds: float = Field(default=5.0)
    weather_cache_ttl_seconds: int = Field(default=1800)  # 30 minutes

    # Authentication & JWT
    jwt_secret_key: str = Field(default="bhoomimitra-ai-secret-key-change-in-production-2026")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=15)
    auth_cookie_name: str = Field(default="access_token")
    auth_cookie_secure: bool = Field(default=False)
    auth_cookie_samesite: str = Field(default="lax")
    admin_registration_key: str = Field(default="")

    # Payment Gateway (Razorpay)
    razorpay_key_id: str = Field(default="rzp_test_bhoomimitra_mock_key")
    razorpay_key_secret: str = Field(default="rzp_test_bhoomimitra_mock_secret")
    razorpay_webhook_secret: str = Field(default="rzp_webhook_secret_mock_2026")

    # Founder Critical Alerting
    founder_alert_webhook_url: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def cookie_secure(self) -> bool:
        return self.auth_cookie_secure or self.is_production


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
