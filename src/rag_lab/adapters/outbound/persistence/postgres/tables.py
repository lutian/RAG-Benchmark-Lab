"""Relational schema. Alembic migrations in ``migrations/`` create exactly these tables."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

metadata = MetaData()

collections = Table(
    "collections",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("name", Text, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

chunking_configs = Table(
    "chunking_configs",
    metadata,
    Column(
        "collection_id",
        Uuid,
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("key", Text, primary_key=True),
    Column("position", Integer, nullable=False),
    Column("strategy", Text, nullable=False),
    Column("chunk_size", Integer, nullable=False),
    Column("chunk_overlap", Integer, nullable=False),
)

documents = Table(
    "documents",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "collection_id",
        Uuid,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("filename", Text, nullable=False),
    Column("media_type", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("collection_id", "content_hash", name="uq_documents_collection_hash"),
)

document_elements = Table(
    "document_elements",
    metadata,
    Column(
        "document_id",
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("position", Integer, primary_key=True),
    Column("kind", Text, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("label", Text, nullable=False),
    Column("level", Integer, nullable=False),
    Column("path", ARRAY(Text), nullable=False),
    CheckConstraint("start_offset >= 0 AND end_offset > start_offset", name="ck_element_span"),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("collection_id", Uuid, nullable=False),
    Column("config_key", Text, nullable=False),
    Column(
        "document_id",
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("chunk_index", Integer, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("page_numbers", ARRAY(Integer), nullable=False),
    Column("section_path", ARRAY(Text), nullable=False),
    ForeignKeyConstraint(
        ["collection_id", "config_key"],
        ["chunking_configs.collection_id", "chunking_configs.key"],
        ondelete="CASCADE",
        name="fk_chunks_chunk_set",
    ),
    UniqueConstraint("document_id", "config_key", "chunk_index", name="uq_chunks_position"),
    CheckConstraint("start_offset >= 0 AND end_offset > start_offset", name="ck_chunk_span"),
    Index("ix_chunks_chunk_set", "collection_id", "config_key"),
)
