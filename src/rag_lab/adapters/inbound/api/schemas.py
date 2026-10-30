"""HTTP request/response models. They map to domain objects but are never used inside it."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from rag_lab.application.services.ingestion import IngestionResult
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document


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


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    chunking_configs: list[ChunkingConfigIn] | None = Field(
        default=None,
        description="Chunk sets to build for every document. Defaults to one per strategy.",
    )


class CollectionOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    chunking_configs: list[ChunkingConfigOut]

    @classmethod
    def from_domain(cls, collection: Collection) -> "CollectionOut":
        return cls(
            id=collection.id,
            name=collection.name,
            created_at=collection.created_at,
            chunking_configs=[
                ChunkingConfigOut.from_domain(c) for c in collection.chunking_configs
            ],
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
