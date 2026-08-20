import re
from typing import AsyncGenerator, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from src.config import get_settings
from src.core.logging import logger

settings = get_settings()


def get_async_db_config(raw_url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Standardize PostgreSQL database URLs for asyncpg and build connect_args.
    Strips libpq-only query parameters (such as sslmode, channel_binding)
    that cause invalid catalog name or unexpected keyword argument errors in asyncpg.
    """
    db_url = raw_url.strip()
    connect_args: Dict[str, Any] = {}

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if "postgresql+asyncpg" in db_url:
        parsed = urlparse(db_url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        sslmode_list = query_params.pop("sslmode", None)
        sslmode = sslmode_list[0] if sslmode_list else None

        # Remove libpq parameters not supported by asyncpg connect()
        unsupported_params = [
            "channel_binding",
            "gssencmode",
            "sslrootcert",
            "sslcert",
            "sslkey",
            "target_session_attrs",
        ]
        for param in unsupported_params:
            query_params.pop(param, None)

        if sslmode == "disable":
            connect_args["ssl"] = False
        else:
            # Default to SSL enabled for postgresql+asyncpg (required by Neon PostgreSQL)
            connect_args["ssl"] = True

        # Reconstruct query string without unsupported parameters
        flat_query = []
        for k, vs in query_params.items():
            for v in vs:
                flat_query.append((k, v))

        new_query = urlencode(flat_query)
        db_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
    elif "sqlite" in db_url:
        connect_args["check_same_thread"] = False

    return db_url, connect_args


db_url, connect_args = get_async_db_config(settings.database_url)

engine_kwargs = {
    "echo": settings.debug,
}

if "sqlite" in db_url:
    pass
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

