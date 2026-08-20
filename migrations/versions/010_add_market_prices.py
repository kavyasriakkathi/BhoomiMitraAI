"""add market_prices table for daily mandi commodity price tracking

Revision ID: 010_add_market_prices
Revises: 009_add_government_schemes
Create Date: 2026-08-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sqa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '010_add_market_prices'
down_revision = '009_add_government_schemes'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'market_prices',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Commodity
        sqa.Column('commodity', sqa.String(length=100), nullable=False),
        sqa.Column('commodity_telugu', sqa.String(length=100), nullable=True),

        # Market / location
        sqa.Column('market_name', sqa.String(length=150), nullable=False),
        sqa.Column('district', sqa.String(length=100), nullable=False),
        sqa.Column('state', sqa.String(length=100), nullable=False),

        # Price data (Rs/quintal by default)
        sqa.Column('min_price', sqa.Float(), nullable=False),
        sqa.Column('max_price', sqa.Float(), nullable=False),
        sqa.Column('modal_price', sqa.Float(), nullable=False),
        sqa.Column('unit', sqa.String(length=20), nullable=False, server_default='Quintal'),

        # Metadata
        sqa.Column('price_date', sqa.DateTime(), nullable=False),
        sqa.Column('source', sqa.String(length=50), nullable=False, server_default='agmarknet_api'),

        sqa.Column('created_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP')),
        sqa.Column('updated_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP')),
    )

    # Individual column indexes (mirrors pattern used in other migrations)
    op.create_index('ix_market_prices_commodity', 'market_prices', ['commodity'])
    op.create_index('ix_market_prices_market_name', 'market_prices', ['market_name'])
    op.create_index('ix_market_prices_district', 'market_prices', ['district'])
    op.create_index('ix_market_prices_state', 'market_prices', ['state'])
    op.create_index('ix_market_prices_price_date', 'market_prices', ['price_date'])

    # Compound index for the primary lookup pattern: commodity + district + date
    op.create_index(
        'idx_market_prices_commodity_district_date',
        'market_prices',
        ['commodity', 'district', 'price_date'],
    )


def downgrade():
    op.drop_index('idx_market_prices_commodity_district_date', table_name='market_prices')
    op.drop_index('ix_market_prices_price_date', table_name='market_prices')
    op.drop_index('ix_market_prices_state', table_name='market_prices')
    op.drop_index('ix_market_prices_district', table_name='market_prices')
    op.drop_index('ix_market_prices_market_name', table_name='market_prices')
    op.drop_index('ix_market_prices_commodity', table_name='market_prices')
    op.drop_table('market_prices')
