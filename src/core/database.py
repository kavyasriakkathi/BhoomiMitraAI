import re
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from src.config import get_settings
from src.core.logging import logger

settings = get_settings()

db_url = settings.database_url

# Standardize Postgres URLs for asyncpg driver
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Handle SSL mode compatibility for asyncpg & Neon PostgreSQL
connect_args = {}
if "postgresql+asyncpg" in db_url:
    if "sslmode=disable" in db_url:
        connect_args["ssl"] = False
        db_url = db_url.replace("sslmode=disable", "")
    elif "sslmode=require" in db_url or "sslmode=prefer" in db_url or "sslmode=verify-full" in db_url:
        connect_args["ssl"] = True
        db_url = re.sub(r"[?&]sslmode=[^&]+", "", db_url)
    else:
        connect_args["ssl"] = True

    # Clean up trailing query string delimiters
    db_url = db_url.rstrip("?").rstrip("&")

engine_kwargs = {
    "echo": settings.debug,
}

if "sqlite" in db_url:
    connect_args["check_same_thread"] = False
else:
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    })

# Create Async Engine
engine = create_async_engine(
    db_url,
    connect_args=connect_args,
    **engine_kwargs
)

# Create Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative Base for Models
Base = declarative_base()

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope around a series of operations."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
