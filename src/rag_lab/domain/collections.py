"""Collections: the isolation boundary. Every query is scoped to exactly one collection.

A collection also owns the list of chunking configurations to benchmark: every
document in it is chunked with each configuration, producing one chunk set per config.
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from rag_lab.domain.chunking.config import ChunkingConfig


@dataclass(frozen=True, slots=True)
class Collection:
    id: UUID
    name: str
    chunking_configs: tuple[ChunkingConfig, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(cls, name: str, chunking_configs: tuple[ChunkingConfig, ...] = ()) -> "Collection":
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("collection name is required")
        keys = [config.key for config in chunking_configs]
        if len(keys) != len(set(keys)):
            raise ValueError("chunking configs must be unique")
        return cls(id=uuid4(), name=cleaned, chunking_configs=chunking_configs)

    def with_chunking_config(self, config: ChunkingConfig) -> "Collection":
        if config in self.chunking_configs:
            raise ValueError(f"chunking config {config.key!r} already exists")
        return replace(self, chunking_configs=(*self.chunking_configs, config))
