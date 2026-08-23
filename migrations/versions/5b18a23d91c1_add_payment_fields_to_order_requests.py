"""add payment fields to order requests
 
Revision ID: 5b18a23d91c1
Revises: 45d4576b132b
Create Date: 2026-08-23 17:07:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5b18a23d91c1'
down_revision = '45d4576b132b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('order_requests', schema=None) as batch_op:
        batch_op.add_column(sa.Column('payment_status', sa.String(length=20), server_default='Pending', nullable=False))
        batch_op.add_column(sa.Column('payment_method', sa.String(length=50), server_default='Online', nullable=True))
        batch_op.add_column(sa.Column('razorpay_order_id', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('razorpay_payment_id', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('razorpay_signature', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('paid_at', sa.DateTime(), nullable=True))
        batch_op.create_index('ix_order_requests_payment_status', ['payment_status'], unique=False)
        batch_op.create_index('ix_order_requests_razorpay_order_id', ['razorpay_order_id'], unique=False)
        batch_op.create_index('ix_order_requests_razorpay_payment_id', ['razorpay_payment_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('order_requests', schema=None) as batch_op:
        batch_op.drop_index('ix_order_requests_razorpay_payment_id')
        batch_op.drop_index('ix_order_requests_razorpay_order_id')
        batch_op.drop_index('ix_order_requests_payment_status')
        batch_op.drop_column('paid_at')
        batch_op.drop_column('razorpay_signature')
        batch_op.drop_column('razorpay_payment_id')
        batch_op.drop_column('razorpay_order_id')
        batch_op.drop_column('payment_method')
        batch_op.drop_column('payment_status')
