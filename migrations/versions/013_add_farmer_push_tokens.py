"""add farmer_push_tokens table for FCM device registration

Revision ID: 013_add_farmer_push_tokens
Revises: 012_merge_heads
Create Date: 2026-09-15 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sqa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '013_add_farmer_push_tokens'
down_revision = '012_merge_heads'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'farmer_push_tokens',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqa.Column('farmer_id', postgresql.UUID(as_uuid=True), sqa.ForeignKey('farmers.id', ondelete='CASCADE'), nullable=False),
        sqa.Column('token', sqa.String(length=512), nullable=False, unique=True),
        sqa.Column('platform', sqa.String(length=20), nullable=False, server_default='android'),
        sqa.Column('device_model', sqa.String(length=100), nullable=True),
        sqa.Column('is_active', sqa.Boolean(), nullable=False, server_default='true'),
        sqa.Column('last_used_at', sqa.DateTime(), nullable=True),
        sqa.Column('created_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
        sqa.Column('updated_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
    )
    op.create_index('idx_farmer_push_tokens_farmer_id', 'farmer_push_tokens', ['farmer_id'])
    op.create_index('idx_farmer_push_tokens_is_active', 'farmer_push_tokens', ['is_active'])
    op.create_index('idx_farmer_push_tokens_token', 'farmer_push_tokens', ['token'])
    op.create_index('idx_farmer_push_tokens_lookup', 'farmer_push_tokens', ['farmer_id', 'is_active'])


def downgrade():
    op.drop_index('idx_farmer_push_tokens_lookup', table_name='farmer_push_tokens')
    op.drop_index('idx_farmer_push_tokens_token', table_name='farmer_push_tokens')
    op.drop_index('idx_farmer_push_tokens_is_active', table_name='farmer_push_tokens')
    op.drop_index('idx_farmer_push_tokens_farmer_id', table_name='farmer_push_tokens')
    op.drop_table('farmer_push_tokens')
