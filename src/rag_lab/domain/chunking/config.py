"""Chunking configuration value object."""

from dataclasses import dataclass
from enum import StrEnum


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    STRUCTURE = "structure"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """How to chunk: strategy plus size and overlap, both measured in characters.

    Characters (not tokens) keep chunking independent of any embedding tokenizer,
    so the same configuration means the same chunks for every provider.
    """

    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}")

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``recursive-800-120``; names a chunk set."""
        return f"{self.strategy.value}-{self.chunk_size}-{self.chunk_overlap}"

    @classmethod
    def from_key(cls, key: str) -> "ChunkingConfig":
        strategy, size, overlap = key.rsplit("-", 2)
        return cls(ChunkingStrategy(strategy), int(size), int(overlap))
