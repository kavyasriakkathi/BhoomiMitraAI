"""merge stock alerts and payment migration heads

Revision ID: 012_merge_stock_alerts_and_payment_heads
Revises: 011_add_stock_alerts, 5b18a23d91c1
Create Date: 2026-09-09 22:49:52.507347

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '012_merge_stock_alerts_and_payment_heads'
down_revision = ('011_add_stock_alerts', '5b18a23d91c1')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
