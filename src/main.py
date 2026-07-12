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

from contextlib import asynccontextmanager
from sqlalchemy import text
from src.core.database import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection validated successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise e
    yield
    # Shutdown
    await engine.dispose()
    logger.info("Database connection closed.")

def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""

    settings = get_settings()

    app = FastAPI(
        title="BhoomiMitra AI",
        description="AI-powered WhatsApp Farming Assistant MVP",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
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
    from src.gateway.router import router as gateway_router
    from src.farmers.router import router as farmers_router
    from src.farmer_profiles.router import router as farmer_profiles_router
    from src.conversation.router import router as conversation_router
    app.include_router(gateway_router, prefix="/webhook", tags=["WhatsApp"])
    app.include_router(farmers_router, prefix="/farmers", tags=["Farmers"])
    app.include_router(farmer_profiles_router, prefix="/farmer-profiles", tags=["Farmer Profiles"])
    app.include_router(conversation_router, prefix="/conversations", tags=["Conversations"])

    logger.info(f"Started {settings.app_name} in {settings.app_env} mode.")
    return app


# Uvicorn entry point
app = create_app()
