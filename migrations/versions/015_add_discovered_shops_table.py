"""add discovered_shops table for automatic location-based shop discovery

Revision ID: 015_discovered_shops
Revises: 014_response_time
Create Date: 2026-10-05 08:40:00.000000

"""
from alembic import op
import sqlalchemy as sqa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '015_discovered_shops'
down_revision = '014_response_time'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'discovered_shops',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqa.Column('provider', sqa.String(length=50), nullable=False),
        sqa.Column('provider_place_id', sqa.String(length=255), nullable=False),
        sqa.Column('shop_name', sqa.String(length=200), nullable=False),
        sqa.Column('business_type', sqa.String(length=100), nullable=True),
        sqa.Column('address', sqa.Text(), nullable=True),
        sqa.Column('latitude', sqa.Float(), nullable=False),
        sqa.Column('longitude', sqa.Float(), nullable=False),
        sqa.Column('phone_number', sqa.String(length=50), nullable=True),
        sqa.Column('maps_url', sqa.String(length=500), nullable=True),
        sqa.Column('rating', sqa.Float(), nullable=True),
        sqa.Column('user_ratings_total', sqa.Integer(), nullable=True),
        sqa.Column('is_operational', sqa.Boolean(), nullable=False, server_default='true'),
        sqa.Column('discovered_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
        sqa.Column('last_verified_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
        sqa.UniqueConstraint('provider', 'provider_place_id', name='uq_discovered_shops_provider_place_id'),
    )
    op.create_index('idx_discovered_shops_coords', 'discovered_shops', ['latitude', 'longitude'])
    op.create_index('idx_discovered_shops_name', 'discovered_shops', ['shop_name'])


def downgrade():
    op.drop_index('idx_discovered_shops_name', table_name='discovered_shops')
    op.drop_index('idx_discovered_shops_coords', table_name='discovered_shops')
    op.drop_table('discovered_shops')
