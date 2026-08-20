"""add government schemes and scheme applications tables

Revision ID: 009_add_government_schemes
Revises: 008_add_order_requests_table
Create Date: 2026-07-28 15:15:00.000000

"""
from alembic import op
import sqlalchemy as sqa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '009_add_government_schemes'
down_revision = '008_add_order_requests_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'government_schemes',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqa.Column('scheme_name', sqa.String(length=200), nullable=False),
        sqa.Column('scheme_code', sqa.String(length=50), nullable=False, unique=True),
        sqa.Column('category', sqa.String(length=100), nullable=False),
        sqa.Column('state', sqa.String(length=100), server_default='All India'),
        sqa.Column('district', sqa.String(length=100), nullable=True),
        sqa.Column('crop_type', sqa.String(length=100), server_default='All Crops'),
        sqa.Column('min_land_acres', sqa.Float(), server_default='0.0'),
        sqa.Column('max_land_acres', sqa.Float(), nullable=True),
        sqa.Column('description', sqa.Text(), nullable=False),
        sqa.Column('benefits_summary', sqa.Text(), nullable=False),
        sqa.Column('eligibility_criteria', sqa.Text(), nullable=False),
        sqa.Column('required_documents', sqa.Text(), nullable=False),
        sqa.Column('application_deadline', sqa.DateTime(), nullable=True),
        sqa.Column('official_portal_url', sqa.String(length=500), nullable=True),
        sqa.Column('is_active', sqa.Boolean(), server_default='true'),
        sqa.Column('created_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP')),
        sqa.Column('updated_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP'))
    )

    op.create_table(
        'scheme_applications',
        sqa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqa.Column('farmer_id', postgresql.UUID(as_uuid=True), sqa.ForeignKey('farmers.id'), nullable=False),
        sqa.Column('scheme_id', postgresql.UUID(as_uuid=True), sqa.ForeignKey('government_schemes.id'), nullable=False),
        sqa.Column('status', sqa.String(length=50), server_default='Eligible'),
        sqa.Column('notes', sqa.Text(), nullable=True),
        sqa.Column('created_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP')),
        sqa.Column('updated_at', sqa.DateTime(), server_default=sqa.text('CURRENT_TIMESTAMP'))
    )


def downgrade():
    op.drop_table('scheme_applications')
    op.drop_table('government_schemes')
