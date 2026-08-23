"""create user accounts table

Revision ID: 45d4576b132b
Revises: 29c3624e4000
Create Date: 2026-08-23 16:10:54.727178

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '45d4576b132b'
down_revision = '29c3624e4000'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'user_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=30), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('expert_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('experts.id'), unique=True, nullable=True),
        sa.Column('shop_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('shops.id'), unique=True, nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index(op.f('ix_user_accounts_email'), 'user_accounts', ['email'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_user_accounts_email'), table_name='user_accounts')
    op.drop_table('user_accounts')

