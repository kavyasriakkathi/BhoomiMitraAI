"""add stock_alerts table for farmer product availability alerts

Revision ID: 011_add_stock_alerts
Revises: 010_add_market_prices
Create Date: 2026-09-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sqa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '011_add_stock_alerts'
down_revision = '010_add_market_prices'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'stock_alerts',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqa.Column('farmer_id', postgresql.UUID(as_uuid=True), sqa.ForeignKey('farmers.id', ondelete='CASCADE'), nullable=False),
        sqa.Column('product_name', sqa.String(length=100), nullable=False),
        sqa.Column('district', sqa.String(length=100), nullable=False),
        sqa.Column('state', sqa.String(length=100), nullable=True, server_default='Telangana'),
        sqa.Column('is_active', sqa.Boolean(), nullable=False, server_default='true'),
        sqa.Column('notified_at', sqa.DateTime(), nullable=True),
        sqa.Column('created_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
        sqa.Column('updated_at', sqa.DateTime(), nullable=False, server_default=sqa.func.now()),
    )
    op.create_index('idx_stock_alerts_farmer_id', 'stock_alerts', ['farmer_id'])
    op.create_index('idx_stock_alerts_product_name', 'stock_alerts', ['product_name'])
    op.create_index('idx_stock_alerts_district', 'stock_alerts', ['district'])
    op.create_index('idx_stock_alerts_is_active', 'stock_alerts', ['is_active'])
    op.create_index('idx_stock_alerts_lookup', 'stock_alerts', ['farmer_id', 'product_name', 'district', 'is_active'])
    op.create_index('idx_stock_alerts_trigger', 'stock_alerts', ['product_name', 'district', 'is_active'])


def downgrade():
    op.drop_index('idx_stock_alerts_trigger', table_name='stock_alerts')
    op.drop_index('idx_stock_alerts_lookup', table_name='stock_alerts')
    op.drop_index('idx_stock_alerts_is_active', table_name='stock_alerts')
    op.drop_index('idx_stock_alerts_district', table_name='stock_alerts')
    op.drop_index('idx_stock_alerts_product_name', table_name='stock_alerts')
    op.drop_index('idx_stock_alerts_farmer_id', table_name='stock_alerts')
    op.drop_table('stock_alerts')
