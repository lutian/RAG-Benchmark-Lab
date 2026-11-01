"""Embeddings: pgvector extension, collection binding and chunk vectors.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-30
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Requires a role allowed to create extensions (the Compose database owner is).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("collections", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.add_column("collections", sa.Column("embedding_dimensions", sa.Integer(), nullable=True))
    op.create_check_constraint(
        "ck_collections_embedding_pair",
        "collections",
        "(embedding_model IS NULL) = (embedding_dimensions IS NULL)",
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("config_key", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.CheckConstraint("vector_dims(embedding) = dimensions", name="ck_embedding_dims"),
    )
    op.create_index(
        "ix_chunk_embeddings_chunk_set", "chunk_embeddings", ["collection_id", "config_key"]
    )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
    op.drop_constraint("ck_collections_embedding_pair", "collections")
    op.drop_column("collections", "embedding_dimensions")
    op.drop_column("collections", "embedding_model")
