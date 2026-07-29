"""add order_requests table

Revision ID: 008_add_order_requests_table
Revises: 007_add_shops_and_inventory
Create Date: 2026-07-28 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008_add_order_requests_table'
down_revision = '007_add_shops_and_inventory'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'order_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('farmer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('shop_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('inventory_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('product_name', sa.String(length=150), nullable=False),
        sa.Column('brand', sa.String(length=100), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=False),
        sa.Column('unit_price', sa.Float(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('total_price', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['farmer_id'], ['farmers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shop_id'], ['shops.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['inventory_id'], ['inventory.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_order_requests_farmer_id'), 'order_requests', ['farmer_id'], unique=False)
    op.create_index(op.f('ix_order_requests_shop_id'), 'order_requests', ['shop_id'], unique=False)
    op.create_index(op.f('ix_order_requests_inventory_id'), 'order_requests', ['inventory_id'], unique=False)
    op.create_index(op.f('ix_order_requests_status'), 'order_requests', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_order_requests_status'), table_name='order_requests')
    op.drop_index(op.f('ix_order_requests_inventory_id'), table_name='order_requests')
    op.drop_index(op.f('ix_order_requests_shop_id'), table_name='order_requests')
    op.drop_index(op.f('ix_order_requests_farmer_id'), table_name='order_requests')
    op.drop_table('order_requests')
