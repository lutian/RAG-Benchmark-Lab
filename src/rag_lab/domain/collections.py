"""Collections: the isolation boundary. Every query is scoped to exactly one collection.

A collection also owns the list of chunking configurations to benchmark: every
document in it is chunked with each configuration, producing one chunk set per config.
Once indexed, it is bound to one embedding spec (model + dimension).
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from rag_lab.domain.chunking.config import ChunkingConfig
from rag_lab.domain.embeddings import EmbeddingSpec


class EmbeddingMismatchError(ValueError):
    """The collection is already indexed with a different embedding spec."""


# Text search configurations shipped with PostgreSQL. "simple" lowercases without
# stemming or stop words: a language-neutral baseline for mixed corpora.
FULLTEXT_LANGUAGES = frozenset(
    {
        "simple", "arabic", "armenian", "basque", "catalan", "danish", "dutch", "english",
        "finnish", "french", "german", "greek", "hindi", "hungarian", "indonesian", "irish",
        "italian", "lithuanian", "nepali", "norwegian", "portuguese", "romanian", "russian",
        "serbian", "spanish", "swedish", "tamil", "turkish", "yiddish",
    }
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Collection:
    id: UUID
    name: str
    chunking_configs: tuple[ChunkingConfig, ...] = ()
    embedding: EmbeddingSpec | None = None
    # Stemming and stop words for full-text search; fixed once documents are indexed.
    fulltext_language: str = "english"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        name: str,
        chunking_configs: tuple[ChunkingConfig, ...] = (),
        fulltext_language: str = "english",
    ) -> "Collection":
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("collection name is required")
        keys = [config.key for config in chunking_configs]
        if len(keys) != len(set(keys)):
            raise ValueError("chunking configs must be unique")
        if fulltext_language not in FULLTEXT_LANGUAGES:
            raise ValueError(
                f"unknown full-text language {fulltext_language!r}; "
                f"use one of {', '.join(sorted(FULLTEXT_LANGUAGES))}"
            )
        return cls(
            id=uuid4(),
            name=cleaned,
            chunking_configs=chunking_configs,
            fulltext_language=fulltext_language,
        )

    def with_chunking_config(self, config: ChunkingConfig) -> "Collection":
        if config in self.chunking_configs:
            raise ValueError(f"chunking config {config.key!r} already exists")
        return replace(self, chunking_configs=(*self.chunking_configs, config))

    def bind_embedding(self, spec: EmbeddingSpec) -> "Collection":
        """Bind to ``spec`` on first indexing; any other spec afterwards is an error."""
        if self.embedding is None:
            return replace(self, embedding=spec)
        if self.embedding != spec:
            raise EmbeddingMismatchError(
                f"collection {self.name!r} is indexed with {self.embedding.label}, "
                f"not {spec.label}; rebuild the index or create a new collection"
            )
        return self

    def unbind_embedding(self) -> "Collection":
        return replace(self, embedding=None)
