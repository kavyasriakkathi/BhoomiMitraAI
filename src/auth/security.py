"""
Security utilities for password hashing (Argon2) and JWT token management.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError
from jose import JWTError, jwt

from src.config import get_settings
from src.core.exceptions import BhoomiMitraException

# Initialize Argon2 Password Hasher with standard secure parameters
_ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    if not password:
        raise ValueError("Password cannot be empty")
    return _ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against an Argon2 hash.
    Returns True if the password matches, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return _ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False
    except Exception:
        return False


def needs_rehash(hashed_password: str) -> bool:
    """Check if the password hash needs updating to newer Argon2 parameters."""
    try:
        return _ph.check_needs_rehash(hashed_password)
    except Exception:
        return True


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generate a signed JWT access token.
    Never pass sensitive plaintext data (like passwords) in payload data.
    """
    settings = get_settings()
    to_encode = data.copy()

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    })

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    Raises BhoomiMitraException (401) on invalid or expired token.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        raise BhoomiMitraException(
            message=f"Invalid or expired authentication token: {str(exc)}",
            status_code=401,
        )
