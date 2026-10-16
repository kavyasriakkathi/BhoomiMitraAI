"""ensure conversations message_id has unique constraint

Revision ID: 017_conversations_message_id_unique
Revises: 016_maps_url_text
Create Date: 2026-10-16 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '017_conversations_message_id_unique'
down_revision = '016_maps_url_text'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # 1. Check if table exists
    table_names = inspector.get_table_names()
    if 'conversations' not in table_names:
        return

    # Check if unique constraint or unique index on message_id already exists
    unique_constraints = inspector.get_unique_constraints('conversations')
    indexes = inspector.get_indexes('conversations')

    has_unique = any(
        'message_id' in (uc.get('column_names') or [])
        for uc in unique_constraints
    ) or any(
        idx.get('unique') and 'message_id' in (idx.get('column_names') or [])
        for idx in indexes
    )

    if has_unique:
        # Constraint or unique index already present in database
        return

    # Use connection context for executing queries across SQLAlchemy versions
    conn = bind if not hasattr(bind, "connect") else bind.connect()
    try:
        # 2. Check for existing duplicate message_id rows before creating constraint.
        # Non-destructive: Do NOT delete any conversations or messages.
        # Disambiguate duplicate message_id values by appending suffix to duplicate copies.
        dup_rows = conn.execute(sa.text("""
            SELECT message_id
            FROM conversations
            WHERE message_id IS NOT NULL AND message_id != ''
            GROUP BY message_id
            HAVING COUNT(*) > 1
        """)).fetchall()

        for row in dup_rows:
            dup_msg_id = row[0]
            # Fetch all matching rows ordered by created_at or id
            matching_rows = conn.execute(sa.text(
                "SELECT id FROM conversations WHERE message_id = :mid ORDER BY created_at ASC, id ASC"
            ), {"mid": dup_msg_id}).fetchall()

            # Keep the first row as the canonical message_id; disambiguate the rest without deleting
            for idx, (conv_id,) in enumerate(matching_rows[1:], start=1):
                new_msg_id = f"{dup_msg_id}__dup_{idx}_{str(conv_id)[:8]}"
                conn.execute(sa.text(
                    "UPDATE conversations SET message_id = :new_id WHERE id = :cid"
                ), {"new_id": new_msg_id, "cid": conv_id})

        if hasattr(conn, "commit"):
            conn.commit()
    finally:
        if hasattr(bind, "connect") and conn is not bind:
            conn.close()

    # 3. Safely create unique index on message_id
    op.create_index(
        'uq_conversations_message_id',
        'conversations',
        ['message_id'],
        unique=True,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    indexes = inspector.get_indexes('conversations')
    has_uq_idx = any(idx.get('name') == 'uq_conversations_message_id' for idx in indexes)
    if has_uq_idx:
        op.drop_index('uq_conversations_message_id', table_name='conversations')

