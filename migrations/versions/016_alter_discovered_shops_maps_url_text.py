"""alter discovered_shops maps_url from varchar(500) to text

Revision ID: 016_maps_url_text
Revises: 015_discovered_shops
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sqa

# revision identifiers, used by Alembic.
revision = '016_maps_url_text'
down_revision = '015_discovered_shops'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        'discovered_shops',
        'maps_url',
        existing_type=sqa.String(length=500),
        type_=sqa.Text(),
        existing_nullable=True,
    )


def downgrade():
    op.alter_column(
        'discovered_shops',
        'maps_url',
        existing_type=sqa.Text(),
        type_=sqa.String(length=500),
        existing_nullable=True,
    )
