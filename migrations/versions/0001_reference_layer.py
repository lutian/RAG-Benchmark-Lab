"""Reference layer: collections, chunking configs, documents, elements and chunks.

Revision ID: 0001
Revises:
Create Date: 2026-10-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "chunking_configs",
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("collection_id", "content_hash", name="uq_documents_collection_hash"),
    )
    op.create_table(
        "document_elements",
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset", name="ck_element_span"
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("config_key", sa.Text(), nullable=False),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_numbers", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("section_path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["collection_id", "config_key"],
            ["chunking_configs.collection_id", "chunking_configs.key"],
            ondelete="CASCADE",
            name="fk_chunks_chunk_set",
        ),
        sa.UniqueConstraint("document_id", "config_key", "chunk_index", name="uq_chunks_position"),
        sa.CheckConstraint("start_offset >= 0 AND end_offset > start_offset", name="ck_chunk_span"),
    )
    op.create_index("ix_chunks_chunk_set", "chunks", ["collection_id", "config_key"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("document_elements")
    op.drop_table("documents")
    op.drop_table("chunking_configs")
    op.drop_table("collections")
