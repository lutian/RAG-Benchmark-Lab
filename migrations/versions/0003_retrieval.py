"""Retrieval: full-text language per collection and a tsvector per chunk.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "collections",
        sa.Column("fulltext_language", sa.Text(), nullable=False, server_default="english"),
    )
    # Added nullable, backfilled from existing chunks, then made mandatory.
    op.add_column("chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))
    op.execute(
        "UPDATE chunks "
        "SET search_vector = to_tsvector(c.fulltext_language::regconfig, chunks.text) "
        "FROM collections c WHERE c.id = chunks.collection_id"
    )
    op.alter_column("chunks", "search_vector", nullable=False)
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
    op.drop_column("collections", "fulltext_language")
