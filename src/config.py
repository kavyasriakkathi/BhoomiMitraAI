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

    # AI
    openai_api_key: str = Field(default="")
    google_gemini_api_key: str = Field(default="")
    ai_confidence_threshold: float = Field(default=0.75)

    # Google Cloud
    google_cloud_project_id: str = Field(default="")
    google_application_credentials: str = Field(default="")

    # Speech-to-Text
    stt_provider: str = Field(default="google")
    stt_default_language: str = Field(default="te-IN")

    # Expert Escalation
    expert_whatsapp_group_id: str = Field(default="")
    escalation_timeout_minutes: int = Field(default=30)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
