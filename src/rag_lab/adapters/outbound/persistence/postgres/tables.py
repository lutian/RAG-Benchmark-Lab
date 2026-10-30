"""Relational schema. Alembic migrations in ``migrations/`` create exactly these tables."""

from pgvector.sqlalchemy import Vector
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
    # Embedding binding: set on first indexing, both NULL or both set.
    Column("embedding_model", Text, nullable=True),
    Column("embedding_dimensions", Integer, nullable=True),
    CheckConstraint(
        "(embedding_model IS NULL) = (embedding_dimensions IS NULL)",
        name="ck_collections_embedding_pair",
    ),
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

# Vectors live in their own table so a rebuild with another model is a plain DELETE.
# The column has no fixed dimension (collections may use different models); the CHECK
# keeps every row consistent with the dimension recorded next to it.
chunk_embeddings = Table(
    "chunk_embeddings",
    metadata,
    Column(
        "chunk_id",
        BigInteger,
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("collection_id", Uuid, nullable=False),
    Column("config_key", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("embedding", Vector(), nullable=False),
    CheckConstraint("vector_dims(embedding) = dimensions", name="ck_embedding_dims"),
    Index("ix_chunk_embeddings_chunk_set", "collection_id", "config_key"),
)
