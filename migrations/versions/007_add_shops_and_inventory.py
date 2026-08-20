"""add shops and inventory tables

Revision ID: 007_add_shops_and_inventory
Revises: 8e17dcd6d1dc
Create Date: 2026-07-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '007_add_shops_and_inventory'
down_revision = '175f73a54f4f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create Shops table
    op.create_table(
        'shops',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('shop_name', sa.String(length=150), nullable=False),
        sa.Column('owner_name', sa.String(length=100), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=True),
        sa.Column('address', sa.Text(), nullable=False),
        sa.Column('village', sa.String(length=100), nullable=True),
        sa.Column('mandal', sa.String(length=100), nullable=True),
        sa.Column('district', sa.String(length=100), nullable=True),
        sa.Column('state', sa.String(length=100), nullable=True),
        sa.Column('pin_code', sa.String(length=20), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('opening_time', sa.String(length=20), nullable=True),
        sa.Column('closing_time', sa.String(length=20), nullable=True),
        sa.Column('delivery_available', sa.Boolean(), nullable=True),
        sa.Column('home_delivery_radius_km', sa.Float(), nullable=True),
        sa.Column('google_maps_link', sa.String(length=500), nullable=True),
        sa.Column('gst_number', sa.String(length=50), nullable=True),
        sa.Column('license_number', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_shops_shop_name'), 'shops', ['shop_name'], unique=False)
    op.create_index(op.f('ix_shops_phone_number'), 'shops', ['phone_number'], unique=False)
    op.create_index(op.f('ix_shops_district'), 'shops', ['district'], unique=False)
    op.create_index(op.f('ix_shops_state'), 'shops', ['state'], unique=False)
    op.create_index(op.f('ix_shops_status'), 'shops', ['status'], unique=False)

    # Create Inventory table
    op.create_table(
        'inventory',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('shop_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('product_name', sa.String(length=150), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('brand', sa.String(length=100), nullable=False),
        sa.Column('product_description', sa.Text(), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('discount_price', sa.Float(), nullable=True),
        sa.Column('quantity_in_stock', sa.Integer(), nullable=False),
        sa.Column('minimum_stock_level', sa.Integer(), nullable=False),
        sa.Column('available', sa.Boolean(), nullable=True),
        sa.Column('expiry_date', sa.DateTime(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['shop_id'], ['shops.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_inventory_shop_id'), 'inventory', ['shop_id'], unique=False)
    op.create_index(op.f('ix_inventory_product_name'), 'inventory', ['product_name'], unique=False)
    op.create_index(op.f('ix_inventory_category'), 'inventory', ['category'], unique=False)
    op.create_index(op.f('ix_inventory_brand'), 'inventory', ['brand'], unique=False)
    op.create_index(op.f('ix_inventory_available'), 'inventory', ['available'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_inventory_available'), table_name='inventory')
    op.drop_index(op.f('ix_inventory_brand'), table_name='inventory')
    op.drop_index(op.f('ix_inventory_category'), table_name='inventory')
    op.drop_index(op.f('ix_inventory_product_name'), table_name='inventory')
    op.drop_index(op.f('ix_inventory_shop_id'), table_name='inventory')
    op.drop_table('inventory')

    op.drop_index(op.f('ix_shops_status'), table_name='shops')
    op.drop_index(op.f('ix_shops_state'), table_name='shops')
    op.drop_index(op.f('ix_shops_district'), table_name='shops')
    op.drop_index(op.f('ix_shops_phone_number'), table_name='shops')
    op.drop_index(op.f('ix_shops_shop_name'), table_name='shops')
    op.drop_table('shops')
