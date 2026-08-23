import sys
import os
import asyncio
from logging.config import fileConfig

# Ensure current working directory is in sys.path
sys.path.insert(0, os.getcwd())

import sqlalchemy as sa
from sqlalchemy import pool
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import our models and settings
from src.config import get_settings
from src.core.database import Base, get_async_db_config
import src.core.models  # Ensures models are registered in Base.metadata

settings = get_settings()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Overwrite the sqlalchemy.url from the env variables instead of alembic.ini
db_url, connect_args = get_async_db_config(settings.database_url)

config.set_main_option("sqlalchemy.url", db_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

def custom_compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    """Handle SQLite UUID reflection compatibility."""
    if isinstance(metadata_type, (sa.UUID, postgresql.UUID)) or getattr(metadata_type, '__visit_name__', '') == 'UUID' or str(metadata_type).upper().startswith('UUID'):
        if isinstance(inspected_type, (sa.NUMERIC, sa.types.NullType, sa.TEXT, sa.String)):
            return False
    return None

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=custom_compare_type,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=custom_compare_type,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context."""

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
