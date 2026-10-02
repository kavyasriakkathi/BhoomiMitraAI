"""add replied_at and response_time_seconds to conversations

Revision ID: 014_response_time
Revises: 013_add_farmer_push_tokens
Create Date: 2026-09-29 00:35:00.000000

"""
from alembic import op
import sqlalchemy as sqa

# revision identifiers, used by Alembic.
revision = '014_response_time'
down_revision = '013_add_farmer_push_tokens'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('conversations', sqa.Column('replied_at', sqa.DateTime(), nullable=True))
    op.add_column('conversations', sqa.Column('response_time_seconds', sqa.Float(), nullable=True))


def downgrade():
    op.drop_column('conversations', 'response_time_seconds')
    op.drop_column('conversations', 'replied_at')
