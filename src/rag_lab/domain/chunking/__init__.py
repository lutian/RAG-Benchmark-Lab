"""Chunking: configurable strategies that cut a document's canonical text into spans."""

from rag_lab.domain.chunking.config import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunking.strategies import chunk_document

__all__ = ["ChunkingConfig", "ChunkingStrategy", "chunk_document"]
