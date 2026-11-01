"""Reranker port: scores (query, text) pairs jointly, unlike embeddings that score them apart."""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Implementations raise ``RerankerError`` on failure."""

    @property
    def name(self) -> str:
        """Model or endpoint label shown in results."""
        ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, in input order; higher is more relevant."""
        ...
