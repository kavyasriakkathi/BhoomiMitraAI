import pytest
from src.core.database import get_async_db_config

def test_neon_db_url_parsing_with_channel_binding():
    raw_url = "postgresql://neondb_owner:pass@ep-xyz.neon.tech/neondb?sslmode=require&channel_binding=require"
    cleaned_url, connect_args = get_async_db_config(raw_url)
    
    assert cleaned_url == "postgresql+asyncpg://neondb_owner:pass@ep-xyz.neon.tech/neondb"
    assert connect_args == {"ssl": True}

def test_postgres_url_conversion():
    raw_url = "postgres://user:pass@localhost:5432/mydb?sslmode=disable"
    cleaned_url, connect_args = get_async_db_config(raw_url)
    
    assert cleaned_url == "postgresql+asyncpg://user:pass@localhost:5432/mydb"
    assert connect_args == {"ssl": False}

def test_sqlite_url_handling():
    raw_url = "sqlite+aiosqlite:///./test.db"
    cleaned_url, connect_args = get_async_db_config(raw_url)
    
    assert cleaned_url == "sqlite+aiosqlite:///./test.db"
    assert connect_args == {"check_same_thread": False}
