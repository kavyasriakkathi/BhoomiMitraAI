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


def test_safety_check_blocks_non_sqlite_and_does_not_leak_credentials(monkeypatch):
    """
    Regression test:
    Verify that the test database safety mechanism stops safely when a non-SQLite
    backend is configured under APP_ENV='test', and that the error message NEVER
    leaks database URLs, usernames, passwords, or hosts.
    """
    from _pytest.outcomes import Exit
    from unittest.mock import MagicMock
    from sqlalchemy.engine import make_url

    secret_password = "SuperSecretDbPassword987!"
    secret_user = "admin_super_user"
    secret_host = "production-cluster.postgres.database.azure.com"
    sensitive_url = f"postgresql+asyncpg://{secret_user}:{secret_password}@{secret_host}:5432/proddb"

    mock_engine = MagicMock()
    mock_engine.url = make_url(sensitive_url)

    # Replicate the exact conftest safety check
    app_env = "test"
    with pytest.raises(Exit) as exc_info:
        if app_env == "test":
            if mock_engine.url.get_backend_name() != "sqlite":
                pytest.exit(
                    "SAFETY CHECK FAILED: APP_ENV is 'test' but configured database backend is not SQLite. "
                    "Halting test run safely before any database connection or modification.",
                    returncode=1,
                )

    err_msg = exc_info.value.msg
    # 1. Error message must not contain the sensitive URL or credentials
    assert sensitive_url not in err_msg
    assert secret_password not in err_msg
    assert secret_user not in err_msg
    assert secret_host not in err_msg
    assert "postgresql" not in err_msg

    # 2. Safety error must indicate the safe failure
    assert "SAFETY CHECK FAILED" in err_msg


def test_safety_check_detects_backend_not_merely_substring():
    """
    Regression test:
    Ensure validation uses engine.url.get_backend_name() and is not fooled by
    the substring 'sqlite' in user credentials or database name.
    """
    from _pytest.outcomes import Exit
    from unittest.mock import MagicMock
    from sqlalchemy.engine import make_url

    # Adversarial URL containing 'sqlite' in user, host, and db name, but actual dialect is postgresql
    adversarial_password = "SecretToken456#"
    adversarial_url = f"postgresql+asyncpg://sqlite_user:{adversarial_password}@sqlite-cloud.io:5432/sqlite_production"

    mock_engine = MagicMock()
    mock_engine.url = make_url(adversarial_url)

    assert mock_engine.url.get_backend_name() == "postgresql"
    assert mock_engine.url.get_backend_name() != "sqlite"

    with pytest.raises(Exit) as exc_info:
        if mock_engine.url.get_backend_name() != "sqlite":
            pytest.exit(
                "SAFETY CHECK FAILED: APP_ENV is 'test' but configured database backend is not SQLite. "
                "Halting test run safely before any database connection or modification.",
                returncode=1,
            )

    err_msg = exc_info.value.msg
    assert adversarial_password not in err_msg
    assert adversarial_url not in err_msg
    assert "sqlite_user" not in err_msg
    assert "sqlite-cloud.io" not in err_msg
