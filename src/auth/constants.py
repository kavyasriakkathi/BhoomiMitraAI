"""
User Role constants for BhoomiMitra AI.
"""

from enum import Enum


class UserRole(str, Enum):
    """Supported roles for BhoomiMitra dashboard users."""
    ADMIN = "admin"
    EXPERT = "expert"
    SHOP_OWNER = "shop_owner"
