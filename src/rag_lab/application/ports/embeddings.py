"""Embedding ports: the model that produces vectors and the store that keeps them."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.embeddings import EmbeddingSpec


class Embedder(Protocol):
    """Turns texts into vectors. Implementations raise ``EmbeddingProviderError``."""

    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingRepository(Protocol):
    """Vectors per chunk, always scoped by collection and chunk set."""

    def pending(self, collection_id: UUID, config_key: str, limit: int) -> list[Chunk]:
        """Chunks of the chunk set that have no vector yet, in reading order."""
        ...

    def save(
        self,
        collection_id: UUID,
        config_key: str,
        model: str,
        items: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> None: ...

    def count(self, collection_id: UUID, config_key: str) -> int: ...

    def delete_collection(self, collection_id: UUID) -> None:
        """Drop every vector of the collection (used when rebuilding the index)."""
        ...
