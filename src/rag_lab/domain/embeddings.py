"""Embedding specification: which model produced a vector and how long the vector is.

Vectors from different models (or dimensions) live in different spaces and cannot
be compared, so a collection records the spec it was indexed with.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("embedding model name is required")
        if self.dimensions <= 0:
            raise ValueError(f"embedding dimensions must be positive, got {self.dimensions}")

    @property
    def label(self) -> str:
        return f"{self.model} ({self.dimensions}d)"
