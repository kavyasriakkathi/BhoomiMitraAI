"""add farms table

Revision ID: 002_add_farms
Revises: 001_initial_schema
Create Date: 2026-07-12 07:28:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_add_farms'
down_revision = '3a084c71d948'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('farms',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('farmer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('farm_name', sa.String(length=100), nullable=False),
        sa.Column('land_size_acres', sa.Float(), nullable=False),
        sa.Column('soil_type', sa.String(length=50), nullable=True),
        sa.Column('irrigation_type', sa.String(length=50), nullable=True),
        sa.Column('village', sa.String(length=100), nullable=True),
        sa.Column('district', sa.String(length=50), nullable=True),
        sa.Column('state', sa.String(length=50), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['farmer_id'], ['farmers.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_farms_farmer_id'), 'farms', ['farmer_id'], unique=False)
    op.create_index(op.f('ix_farms_district'), 'farms', ['district'], unique=False)
    op.create_index(op.f('ix_farms_state'), 'farms', ['state'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_farms_state'), table_name='farms')
    op.drop_index(op.f('ix_farms_district'), table_name='farms')
    op.drop_index(op.f('ix_farms_farmer_id'), table_name='farms')
    op.drop_table('farms')
