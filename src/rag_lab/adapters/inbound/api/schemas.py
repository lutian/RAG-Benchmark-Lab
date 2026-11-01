"""HTTP request/response models. They map to domain objects but are never used inside it."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from rag_lab.application.services.indexing import IndexingReport, IndexStatus
from rag_lab.application.services.ingestion import IngestionResult
from rag_lab.application.services.retrieval import MAX_TOP_K, SearchResult
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document
from rag_lab.domain.embeddings import EmbeddingSpec
from rag_lab.domain.retrieval import SearchMethod


class ChunkingConfigIn(BaseModel):
    strategy: ChunkingStrategy
    chunk_size: int = Field(gt=0, le=20_000)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def to_domain(self) -> ChunkingConfig:
        return ChunkingConfig(self.strategy, self.chunk_size, self.chunk_overlap)


class ChunkingConfigOut(BaseModel):
    key: str
    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_domain(cls, config: ChunkingConfig) -> "ChunkingConfigOut":
        return cls(
            key=config.key,
            strategy=config.strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )


class EmbeddingOut(BaseModel):
    model: str
    dimensions: int

    @classmethod
    def from_domain(cls, spec: EmbeddingSpec | None) -> "EmbeddingOut | None":
        return cls(model=spec.model, dimensions=spec.dimensions) if spec else None


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    chunking_configs: list[ChunkingConfigIn] | None = Field(
        default=None,
        description="Chunk sets to build for every document. Defaults to one per strategy.",
    )
    fulltext_language: str | None = Field(
        default=None,
        description="PostgreSQL text search configuration (e.g. english, portuguese, simple). "
        "Defaults to DEFAULT_FULLTEXT_LANGUAGE.",
    )


class CollectionOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    chunking_configs: list[ChunkingConfigOut]
    fulltext_language: str
    embedding: EmbeddingOut | None = Field(
        description="Embedding model the collection is bound to (null until first indexed)."
    )

    @classmethod
    def from_domain(cls, collection: Collection) -> "CollectionOut":
        return cls(
            id=collection.id,
            name=collection.name,
            created_at=collection.created_at,
            chunking_configs=[
                ChunkingConfigOut.from_domain(c) for c in collection.chunking_configs
            ],
            embedding=EmbeddingOut.from_domain(collection.embedding),
            fulltext_language=collection.fulltext_language,
        )


class DocumentSummaryOut(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    media_type: str
    content_hash: str
    characters: int
    metadata: dict[str, str]
    created_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> "DocumentSummaryOut":
        return cls(
            id=document.id,
            collection_id=document.collection_id,
            filename=document.filename,
            media_type=document.media_type.value,
            content_hash=document.content_hash,
            characters=len(document.text),
            metadata=dict(document.metadata),
            created_at=document.created_at,
        )


class IngestionOut(DocumentSummaryOut):
    duplicate: bool
    chunk_counts: dict[str, int]

    @classmethod
    def from_result(cls, result: IngestionResult) -> "IngestionOut":
        summary = DocumentSummaryOut.from_domain(result.document)
        return cls(
            **summary.model_dump(),
            duplicate=result.duplicate,
            chunk_counts=result.chunk_counts,
        )


class ElementOut(BaseModel):
    kind: str
    start: int
    end: int
    label: str
    level: int
    path: list[str]


class DocumentDetailOut(DocumentSummaryOut):
    text: str
    elements: list[ElementOut]

    @classmethod
    def from_document(cls, document: Document) -> "DocumentDetailOut":
        return cls(
            **DocumentSummaryOut.from_domain(document).model_dump(),
            text=document.text,
            elements=[
                ElementOut(
                    kind=e.kind.value,
                    start=e.span.start,
                    end=e.span.end,
                    label=e.label,
                    level=e.level,
                    path=list(e.path),
                )
                for e in document.elements
            ],
        )


class ChunkOut(BaseModel):
    document_id: UUID
    index: int
    start: int
    end: int
    text: str
    page_numbers: list[int]
    section_path: list[str]

    @classmethod
    def from_domain(cls, chunk: Chunk) -> "ChunkOut":
        return cls(
            document_id=chunk.document_id,
            index=chunk.index,
            start=chunk.span.start,
            end=chunk.span.end,
            text=chunk.text,
            page_numbers=list(chunk.page_numbers),
            section_path=list(chunk.section_path),
        )


class IndexingReportOut(BaseModel):
    config_key: str
    embedding: EmbeddingOut
    total: int = Field(description="Chunks in the chunk set.")
    already_indexed: int = Field(description="Chunks that already had a vector.")
    embedded: int = Field(description="Chunks embedded by this run.")
    seconds: float

    @classmethod
    def from_report(cls, report: IndexingReport) -> "IndexingReportOut":
        return cls(
            config_key=report.config_key,
            embedding=EmbeddingOut(
                model=report.embedding.model, dimensions=report.embedding.dimensions
            ),
            total=report.total,
            already_indexed=report.already_indexed,
            embedded=report.embedded,
            seconds=report.seconds,
        )


class ChunkSetStatusOut(BaseModel):
    config_key: str
    total: int
    embedded: int


class IndexStatusOut(BaseModel):
    embedding: EmbeddingOut | None
    chunk_sets: list[ChunkSetStatusOut]

    @classmethod
    def from_status(cls, status: IndexStatus) -> "IndexStatusOut":
        return cls(
            embedding=EmbeddingOut.from_domain(status.embedding),
            chunk_sets=[
                ChunkSetStatusOut(config_key=s.config_key, total=s.total, embedded=s.embedded)
                for s in status.chunk_sets
            ],
        )


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    chunk_set: str = Field(description="Chunking config key, e.g. recursive-800-120.")
    method: SearchMethod = SearchMethod.HYBRID
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    document_ids: list[UUID] | None = Field(
        default=None, description="Only search these documents (null = all)."
    )


class SearchHitOut(ChunkOut):
    rank: int
    score: float = Field(
        description="vector: cosine similarity; fulltext: ts_rank_cd; hybrid: RRF score; "
        "hybrid_rerank: reranker score."
    )
    ranks: dict[str, int] = Field(
        description="1-based rank per stage: vector, fulltext, and hybrid (before reranking)."
    )


class SearchTimingsOut(BaseModel):
    embed_ms: float
    search_ms: float
    rerank_ms: float
    total_ms: float


class SearchOut(BaseModel):
    query: str
    chunk_set: str
    method: SearchMethod
    hits: list[SearchHitOut]
    timings: SearchTimingsOut

    @classmethod
    def from_result(cls, result: SearchResult) -> "SearchOut":
        return cls(
            query=result.query,
            chunk_set=result.config_key,
            method=result.method,
            hits=[
                SearchHitOut(
                    **ChunkOut.from_domain(hit.chunk).model_dump(),
                    rank=hit.rank,
                    score=hit.score,
                    ranks=dict(hit.ranks),
                )
                for hit in result.hits
            ],
            timings=SearchTimingsOut(
                embed_ms=result.timings.embed_ms,
                search_ms=result.timings.search_ms,
                rerank_ms=result.timings.rerank_ms,
                total_ms=result.timings.total_ms,
            ),
        )
