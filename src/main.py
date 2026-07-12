"""
KrishiMitra AI — Application Entry Point

FastAPI application factory with health check and module registration.
"""

from fastapi import FastAPI
from src.config import get_settings
from src.core.logging import logger
from src.core.exceptions import (
    BhoomiMitraException, 
    bhoomimitra_exception_handler, 
    global_exception_handler
)

def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""

    settings = get_settings()

    app = FastAPI(
        title="BhoomiMitra AI",
        description="AI-powered WhatsApp Farming Assistant MVP",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    # ---- Exception Handlers ----
    app.add_exception_handler(BhoomiMitraException, bhoomimitra_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # ---- Health Check ----
    @app.get("/health", tags=["System"])
    async def health_check():
        logger.info("Health check endpoint called")
        return {
            "success": True,
            "data": {
                "status": "healthy",
                "service": settings.app_name,
                "environment": settings.app_env
            }
        }

    # ---- Register Modules ----
    # Modules will be registered here as we build them:
    # from src.gateway.router import router as gateway_router
    # app.include_router(gateway_router, prefix="/webhook", tags=["WhatsApp"])

    logger.info(f"Started {settings.app_name} in {settings.app_env} mode.")
    return app


# Uvicorn entry point
app = create_app()
