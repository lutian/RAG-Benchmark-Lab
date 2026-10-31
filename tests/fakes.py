"""Test doubles shared across suites."""

import hashlib
from collections.abc import Sequence

from rag_lab.application.errors import EmbeddingProviderError
from rag_lab.domain.embeddings import EmbeddingSpec


class FakeEmbedder:
    """Deterministic embedder: the same text always maps to the same unit-ish vector.

    ``fail_on_call`` makes the n-th ``embed_documents`` call (1-based) raise, to test
    resumable indexing; ``wrong_dimensions`` returns vectors of the wrong length.
    """

    def __init__(
        self,
        model: str = "fake-embed",
        dimensions: int = 8,
        fail_on_call: int | None = None,
        wrong_dimensions: bool = False,
    ) -> None:
        self._spec = EmbeddingSpec(model, dimensions)
        self._fail_on_call = fail_on_call
        self._wrong_dimensions = wrong_dimensions
        self.batches: list[list[str]] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._fail_on_call == len(self.batches):
            raise EmbeddingProviderError("fake provider outage")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        size = self._spec.dimensions + (1 if self._wrong_dimensions else 0)
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i % len(digest)] / 255 for i in range(size)]


class ConceptEmbedder:
    """Embeds texts as counts of *concepts*: words mapped to the same dimension.

    Synonyms share a dimension, so vector search can match a paraphrase that
    full-text search misses — the behaviour the retrieval tests need to observe.
    Texts with no known word map to a small constant vector (never all zeros).
    """

    def __init__(self, concepts: dict[str, int], dimensions: int, model: str = "concepts") -> None:
        self._spec = EmbeddingSpec(model, dimensions)
        self._concepts = concepts
        self.queries: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.01] * self._spec.dimensions
        for word in text.lower().replace(".", " ").replace(",", " ").split():
            if word in self._concepts:
                vector[self._concepts[word]] += 1.0
        return vector
