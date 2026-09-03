"""
KrishiMitra AI — Application Entry Point

FastAPI application factory with health check and module registration.
"""

from fastapi import FastAPI, Request, Response, status
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

def audit_environment_variables():
    """Logs the presence/absence and key configuration of environment variables on startup."""
    settings = get_settings()
    
    def _status(val: str) -> str:
        if not val:
            return "MISSING / NOT SET"
        return f"PRESENT (len={len(str(val))})"

    logger.info("=" * 75)
    logger.info("BHOOMIMITRA AI — ENVIRONMENT CONFIGURATION AUDIT ON SERVER BOOT")
    logger.info("=" * 75)
    logger.info(f"APP_ENV                   : {settings.app_env}")
    logger.info(f"DEBUG                     : {settings.debug}")
    logger.info(f"LOG_LEVEL                 : {settings.log_level}")
    logger.info("--- REQUIRED IN PRODUCTION ---")
    logger.info(f"DATABASE_URL              : {_status(settings.database_url)}")
    logger.info(f"WHATSAPP_PHONE_NUMBER_ID  : {settings.whatsapp_phone_number_id or 'NOT SET'}")
    logger.info(f"WHATSAPP_API_TOKEN        : {_status(settings.whatsapp_api_token)}")
    logger.info(f"WHATSAPP_VERIFY_TOKEN     : {_status(settings.whatsapp_verify_token)}")
    logger.info(f"WHATSAPP_APP_SECRET       : {_status(settings.whatsapp_app_secret)}")
    logger.info("--- OPTIONAL / FALLBACK-ENABLED SERVICES ---")
    logger.info(f"GOOGLE_GEMINI_API_KEY     : {_status(settings.google_gemini_api_key)} (Model: {settings.gemini_model})")
    logger.info(f"STT_PROVIDER              : {settings.stt_provider} (Lang: {settings.stt_default_language})")
    logger.info(f"GOOGLE_APP_CREDENTIALS    : {_status(settings.google_application_credentials)}")
    logger.info(f"OPENWEATHER_API_KEY       : {_status(settings.openweather_api_key)} (Mock fallback enabled)")
    logger.info(f"DATA_GOV_API_KEY          : {_status(settings.data_gov_api_key)} (DB fallback enabled)")
    logger.info(f"REDIS_URL                 : {_status(settings.redis_url)}")
    logger.info("--- TIMEOUT SETTINGS ---")
    logger.info(f"GEMINI_TIMEOUT            : {settings.gemini_api_timeout_seconds}s")
    logger.info(f"STT_TIMEOUT               : {settings.stt_api_timeout_seconds}s")
    logger.info(f"WEATHER_TIMEOUT           : {settings.openweather_api_timeout_seconds}s")
    logger.info(f"AGMARKNET_TIMEOUT         : {settings.agmarknet_api_timeout_seconds}s")
    logger.info(f"WHATSAPP_TIMEOUT          : {settings.whatsapp_api_timeout_seconds}s")
    logger.info("=" * 75)


def validate_production_settings():
    """
    When APP_ENV=production, validates that critical WhatsApp and DB environment variables exist.
    Fails fast with a clear configuration error without logging actual secret values.
    """
    settings = get_settings()
    if not settings.is_production:
        return

    missing_wa_vars = []
    if not settings.whatsapp_api_token:
        missing_wa_vars.append("WHATSAPP_API_TOKEN")
    if not settings.whatsapp_phone_number_id:
        missing_wa_vars.append("WHATSAPP_PHONE_NUMBER_ID")
    if not settings.whatsapp_verify_token:
        missing_wa_vars.append("WHATSAPP_VERIFY_TOKEN")
    if not settings.whatsapp_app_secret:
        missing_wa_vars.append("WHATSAPP_APP_SECRET")

    if missing_wa_vars:
        error_msg = (
            f"[FATAL CONFIG ERROR] Missing required WhatsApp settings for production: "
            f"{', '.join(missing_wa_vars)}. Server startup aborted."
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

    if hasattr(settings, "database_url") and not settings.database_url:
        error_msg = (
            "[FATAL CONFIG ERROR] Missing required production settings: "
            "DATABASE_URL. Server startup aborted."
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    try:
        audit_environment_variables()
        validate_production_settings()
        from src.core.models import Base
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database connection validated and tables created successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to the database or start application: {e}")
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
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # ---- Exception Handlers ----
    app.add_exception_handler(BhoomiMitraException, bhoomimitra_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # ---- CORS Middleware ----
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Health Check ----
    @app.get("/health", tags=["System"])
    async def health_check(response: Response):
        """
        Active dependency health check endpoint for production monitoring and Render auto-healing.
        Actively verifies PostgreSQL (SELECT 1) and Redis (PING).
        """
        db_status = "ok"
        redis_status = "ok"

        # 1. Active PostgreSQL verification
        try:
            from src.core.database import AsyncSessionLocal
            from sqlalchemy import text
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
        except Exception as db_err:
            logger.warning(f"[HEALTH CHECK] PostgreSQL check failed: {db_err}")
            db_status = "error"

        # 2. Active Redis verification
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url, socket_connect_timeout=1.5, socket_timeout=1.5)
            await r.ping()
            await r.aclose()
        except Exception as redis_err:
            logger.debug(f"[HEALTH CHECK] Redis check failed: {redis_err}")
            redis_status = "error"

        # 3. Compute overall status and HTTP code
        if db_status == "error":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            overall_status = "unhealthy"
            try:
                from src.core.alerting import dispatch_founder_alert, AlertCategory, AlertSeverity
                import asyncio
                asyncio.create_task(dispatch_founder_alert(
                    category=AlertCategory.DATABASE_DOWN,
                    severity=AlertSeverity.CRITICAL,
                    component="postgresql",
                    summary="PostgreSQL health check failed (SELECT 1 query error / connection refused).",
                    recommended_action="Inspect PostgreSQL instance on Render/Neon and verify DATABASE_URL credentials.",
                ))
            except Exception as alert_err:
                logger.debug(f"Founder alert dispatch skipped: {alert_err}")
        elif redis_status == "error":
            response.status_code = status.HTTP_200_OK
            overall_status = "degraded"
        else:
            response.status_code = status.HTTP_200_OK
            overall_status = "healthy"

        return {
            "status": overall_status,
            "database": db_status,
            "redis": redis_status,
        }

    # ---- Static Files & Web Dashboard Router ----
    # NOTE: /data-deletion, /privacy-policy routes are served by dashboard/router.py (canonical).
    import os
    from fastapi.staticfiles import StaticFiles

    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    from src.dashboard.router import router as dashboard_router
    app.include_router(dashboard_router)

    # ---- Register Modules ----
    from src.gateway.router import router as gateway_router
    from src.farmers.router import router as farmers_router
    from src.farmer_profiles.router import router as farmer_profiles_router
    from src.conversation.router import router as conversation_router
    from src.farms.router import router as farms_router
    from src.crops.router import router as crops_router
    from src.crop_health.router import router as crop_health_router
    from src.advisory.router import router as advisory_router
    from src.ai.router import router as ai_router
    from src.auth.router import router as auth_router
    from src.shops.router import router as shops_router
    from src.inventory.router import router as inventory_router
    from src.orders.router import router as orders_router
    from src.schemes.router import router as schemes_router
    from src.memory.router import router as memory_router
    from src.rag.router import router as rag_router
    from src.market.router import router as market_router
    from src.weather.router import router as weather_router
    from src.escalation.router import router as escalation_router
    from src.payments.router import router as payments_router
    from src.analytics.router import router as analytics_router

    app.include_router(auth_router, prefix="/auth", tags=["Authentication & RBAC"])
    app.include_router(gateway_router, prefix="/webhook", tags=["WhatsApp"])
    app.include_router(farmers_router, prefix="/farmers", tags=["Farmers"])
    app.include_router(farmer_profiles_router, prefix="/farmer-profiles", tags=["Farmer Profiles"])
    app.include_router(conversation_router, prefix="/conversations", tags=["Conversations"])
    app.include_router(farms_router, prefix="/farms", tags=["Farms"])
    app.include_router(crops_router, prefix="/crops", tags=["Crops"])
    app.include_router(crop_health_router, prefix="/crop-health", tags=["Crop Health"])
    app.include_router(advisory_router, prefix="/advisories", tags=["Advisories"])
    app.include_router(shops_router, prefix="/shops", tags=["Agri Shops"])
    app.include_router(inventory_router, prefix="/inventory", tags=["Inventory Management"])
    app.include_router(orders_router, prefix="/orders", tags=["Order Requests & Cart"])
    app.include_router(payments_router, prefix="/payments", tags=["Payments"])
    app.include_router(schemes_router, prefix="/schemes", tags=["Government Schemes"])
    app.include_router(memory_router, prefix="/memory", tags=["Farmer Memory Profile"])
    app.include_router(rag_router, prefix="/rag", tags=["RAG Knowledge Engine"])
    app.include_router(market_router, prefix="/market", tags=["Market Prices"])
    app.include_router(weather_router, prefix="/weather", tags=["Weather Forecast"])
    app.include_router(escalation_router, prefix="/escalation", tags=["Expert Escalation"])
    app.include_router(analytics_router, prefix="/analytics", tags=["Startup Pilot Analytics"])
    app.include_router(ai_router)


    logger.info(f"Started {settings.app_name} in {settings.app_env} mode.")
    return app


# Uvicorn entry point
app = create_app()
