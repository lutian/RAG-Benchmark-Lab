"""HTTP request/response models. They map to domain objects but are never used inside it."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from rag_lab.application.reporting import best_strategy, headline_k
from rag_lab.application.run_codec import run_to_payload
from rag_lab.application.services.answering import Answer
from rag_lab.application.services.golden import GoldenQuestionView, ImportReport
from rag_lab.application.services.indexing import IndexingReport, IndexStatus
from rag_lab.application.services.ingestion import IngestionResult
from rag_lab.application.services.retrieval import MAX_TOP_K, SearchHit, SearchResult
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document
from rag_lab.domain.embeddings import EmbeddingSpec
from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    BenchmarkRun,
    QuestionOrigin,
    QuestionStatus,
    StrategySummary,
)
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


class CollectionRenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


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


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    chunk_set: str = Field(description="Chunking config key, e.g. recursive-800-120.")
    method: SearchMethod = SearchMethod.HYBRID
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, description="Sources given to the LLM.")
    document_ids: list[UUID] | None = None


class CitationOut(BaseModel):
    number: int = Field(description="Source number as written in the answer, e.g. 2 for [2].")
    source: SearchHitOut


class AnswerTimingsOut(BaseModel):
    retrieval: SearchTimingsOut
    generation_ms: float
    total_ms: float = Field(description="Whole request: retrieval + generation.")


class AnswerOut(BaseModel):
    question: str
    chunk_set: str
    method: SearchMethod
    answer: str
    abstained: bool
    abstention_reason: str | None = Field(
        description="no_evidence_retrieved (retrieval found nothing) or model_abstained."
    )
    citations: list[CitationOut]
    citations_valid: bool = Field(
        description="At least one citation and all of them point to provided sources."
    )
    invalid_citations: list[int]
    sources: list[SearchHitOut] = Field(description="Every chunk given to the LLM, numbered 1..n.")
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    timings: AnswerTimingsOut

    @classmethod
    def from_answer(cls, answer: Answer) -> "AnswerOut":
        def hit_out(hit: SearchHit) -> SearchHitOut:
            return SearchHitOut(
                **ChunkOut.from_domain(hit.chunk).model_dump(),
                rank=hit.rank,
                score=hit.score,
                ranks=dict(hit.ranks),
            )

        retrieval = answer.timings.retrieval
        return cls(
            question=answer.question,
            chunk_set=answer.config_key,
            method=answer.method,
            answer=answer.text,
            abstained=answer.abstained,
            abstention_reason=answer.abstention_reason,
            citations=[
                CitationOut(number=c.number, source=hit_out(c.hit)) for c in answer.citations
            ],
            citations_valid=answer.citation_check.valid,
            invalid_citations=list(answer.citation_check.invalid),
            sources=[hit_out(hit) for hit in answer.sources],
            model=answer.model,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            timings=AnswerTimingsOut(
                retrieval=SearchTimingsOut(
                    embed_ms=retrieval.embed_ms,
                    search_ms=retrieval.search_ms,
                    rerank_ms=retrieval.rerank_ms,
                    total_ms=retrieval.total_ms,
                ),
                generation_ms=answer.timings.generation_ms,
                total_ms=answer.timings.total_ms,
            ),
        )


class EvidenceOut(BaseModel):
    document_id: UUID
    filename: str
    start: int
    end: int
    excerpt: str


class GoldenQuestionOut(BaseModel):
    id: UUID
    text: str
    status: QuestionStatus
    origin: QuestionOrigin
    created_at: datetime
    evidence: list[EvidenceOut]

    @classmethod
    def from_view(cls, view: GoldenQuestionView) -> "GoldenQuestionOut":
        q = view.question
        return cls(
            id=q.id,
            text=q.text,
            status=q.status,
            origin=q.origin,
            created_at=q.created_at,
            evidence=[
                EvidenceOut(
                    document_id=e.document_id,
                    filename=e.filename,
                    start=e.span.start,
                    end=e.span.end,
                    excerpt=e.excerpt,
                )
                for e in view.evidence
            ],
        )


class ManualQuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    document_id: UUID
    excerpt: str = Field(
        min_length=1,
        max_length=20_000,
        description="Text copied from the document that answers the question.",
    )


class GenerateQuestionsIn(BaseModel):
    count: int = Field(default=10, ge=1, le=200)
    seed: int = Field(default=0, description="Same seed + same documents = same passages.")


class QuestionUpdateIn(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=2_000)
    status: QuestionStatus | None = None


class ImportReportOut(BaseModel):
    imported: int
    skipped: int = Field(description="Questions whose documents are not in this collection.")

    @classmethod
    def from_report(cls, report: ImportReport) -> "ImportReportOut":
        return cls(imported=report.imported, skipped=report.skipped)


class BenchmarkIn(BaseModel):
    chunk_sets: list[str] | None = Field(default=None, description="Default: every chunk set.")
    methods: list[SearchMethod] | None = Field(default=None, description="Default: all four.")
    ks: list[int] = Field(default=[1, 5, 10], min_length=1)
    mrr_cutoff: int = Field(default=10, ge=1, le=100)
    generation: bool = Field(
        default=False,
        description="Also answer every question with every strategy (one LLM call each).",
    )
    judge: bool = Field(
        default=False, description="LLM judge of answer support (needs generation)."
    )
    answer_top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)

    def to_config(self, all_chunk_sets: list[str]) -> BenchmarkConfig:
        return BenchmarkConfig(
            chunk_sets=tuple(self.chunk_sets or all_chunk_sets),
            methods=tuple(self.methods or list(SearchMethod)),
            ks=tuple(sorted(set(self.ks))),
            mrr_cutoff=self.mrr_cutoff,
            generation=self.generation,
            judge=self.judge,
            answer_top_k=self.answer_top_k,
        )


class StrategySummaryOut(BaseModel):
    chunk_set: str
    method: SearchMethod
    questions: int
    recall: dict[int, float]
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    cpu_ms_mean: float
    abstention_rate: float | None
    citation_valid_rate: float | None
    supported_rate: float | None
    tokens: int | None


class BenchmarkRunOut(BaseModel):
    id: UUID
    created_at: datetime
    question_count: int
    config: dict[str, object]
    environment: dict[str, str]
    headline_k: int = Field(description="The Recall@k used to rank strategies (5 when measured).")
    best: StrategySummaryOut = Field(description="Same rule as the report and the CLI.")
    summaries: list[StrategySummaryOut]

    @classmethod
    def from_run(cls, run: BenchmarkRun) -> "BenchmarkRunOut":
        def summary(s: StrategySummary) -> StrategySummaryOut:
            return StrategySummaryOut(
                chunk_set=s.chunk_set,
                method=s.method,
                questions=s.questions,
                recall=s.recall,
                mrr=s.mrr,
                latency_p50_ms=s.latency_p50_ms,
                latency_p95_ms=s.latency_p95_ms,
                cpu_ms_mean=s.cpu_ms_mean,
                abstention_rate=s.abstention_rate,
                citation_valid_rate=s.citation_valid_rate,
                supported_rate=s.supported_rate,
                tokens=s.tokens,
            )

        k = headline_k(run.config)
        return cls(
            id=run.id,
            created_at=run.created_at,
            question_count=run.question_count,
            config=run_to_payload(run)["config"],
            environment=run.environment,
            headline_k=k,
            best=summary(best_strategy(run.summaries, k)),
            summaries=[summary(s) for s in run.summaries],
        )
