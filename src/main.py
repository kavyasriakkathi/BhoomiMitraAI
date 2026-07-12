"""
KrishiMitra AI — Application Entry Point

FastAPI application factory with health check and module registration.
"""

from fastapi import FastAPI
from src.config import get_settings


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""

    settings = get_settings()

    app = FastAPI(
        title="KrishiMitra AI",
        description="AI-powered WhatsApp Farming Assistant",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    # ---- Health Check ----
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "service": settings.app_name,
            "environment": settings.app_env,
        }

    # ---- Register Modules ----
    # Modules will be registered here as we build them:
    # from src.gateway.router import router as gateway_router
    # app.include_router(gateway_router, prefix="/webhook", tags=["WhatsApp"])

    return app


# Uvicorn entry point
app = create_app()
