"""
BhoomiMitra AI — Authentication and Authorization Module.
"""

from src.auth.constants import UserRole
from src.auth.dependencies import (
    get_current_active_user,
    get_current_user,
    require_admin,
    require_expert,
    require_roles,
    require_shop_owner,
    verify_expert_access,
    verify_shop_access,
)
from src.auth.router import router

__all__ = [
    "router",
    "UserRole",
    "get_current_user",
    "get_current_active_user",
    "require_roles",
    "require_admin",
    "require_expert",
    "require_shop_owner",
    "verify_shop_access",
    "verify_expert_access",
]
