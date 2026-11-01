"""Test doubles shared across suites."""

import hashlib
from collections.abc import Callable, Sequence

from rag_lab.application.errors import (
    EmbeddingProviderError,
    GenerationProviderError,
    RerankerError,
)
from rag_lab.application.ports.generation import ChatMessage, Completion
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


class FakeReranker:
    """Scores 1.0 for texts containing ``prefer`` and 0.0 otherwise; records every call."""

    def __init__(self, prefer: str = "", fail: bool = False) -> None:
        self._prefer = prefer.lower()
        self._fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def name(self) -> str:
        return "fake-reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self._fail:
            raise RerankerError("fake reranker outage")
        return [1.0 if self._prefer and self._prefer in t.lower() else 0.0 for t in texts]


class FakeChatModel:
    """Returns a scripted reply (or one computed from the prompt) and records prompts."""

    def __init__(self, reply: str | Callable[[str], str] = "", fail: bool = False) -> None:
        self._reply = reply
        self._fail = fail
        self.prompts: list[list[ChatMessage]] = []

    @property
    def name(self) -> str:
        return "fake-chat"

    def complete(self, messages: Sequence[ChatMessage]) -> Completion:
        self.prompts.append(list(messages))
        if self._fail:
            raise GenerationProviderError("fake LLM outage")
        user = messages[-1].content
        text = self._reply(user) if callable(self._reply) else self._reply
        return Completion(text=text, model="fake-chat-1", prompt_tokens=10, completion_tokens=5)
