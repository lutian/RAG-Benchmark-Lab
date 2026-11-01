"""Repository implementations over a SQLAlchemy connection owned by the unit of work."""

import builtins
from collections.abc import Sequence, Set
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    Text,
    bindparam,
    cast,
    delete,
    func,
    insert,
    literal,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlalchemy.exc import IntegrityError

from rag_lab.adapters.outbound.persistence.postgres import tables as t
from rag_lab.adapters.outbound.persistence.run_codec import run_from_payload, run_to_payload
from rag_lab.application.errors import ConflictError
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document, ElementKind, MediaType, StructuralElement
from rag_lab.domain.embeddings import EmbeddingSpec
from rag_lab.domain.evaluation import (
    BenchmarkRun,
    Evidence,
    GoldenQuestion,
    QuestionOrigin,
    QuestionStatus,
)
from rag_lab.domain.retrieval import ScoredChunk
from rag_lab.domain.spans import Span


def _write(connection: Connection, statement: Any, rows: Any = None) -> None:
    """Execute a write, translating constraint violations into a ConflictError."""
    try:
        with connection.begin_nested():
            connection.execute(statement, rows) if rows is not None else connection.execute(
                statement
            )
    except IntegrityError as error:
        raise ConflictError(f"conflicting data: {error.orig}") from error


class PostgresCollectionRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, collection: Collection) -> None:
        _write(
            self._connection,
            insert(t.collections).values(
                id=collection.id,
                name=collection.name,
                fulltext_language=collection.fulltext_language,
                created_at=collection.created_at,
            ),
        )
        self._save_embedding(collection)
        self._save_configs(collection)

    def update(self, collection: Collection) -> None:
        self._connection.execute(
            update(t.collections)
            .where(t.collections.c.id == collection.id)
            .values(name=collection.name)
        )
        self._save_embedding(collection)
        self._save_configs(collection)

    def _save_embedding(self, collection: Collection) -> None:
        spec = collection.embedding
        self._connection.execute(
            update(t.collections)
            .where(t.collections.c.id == collection.id)
            .values(
                embedding_model=spec.model if spec else None,
                embedding_dimensions=spec.dimensions if spec else None,
            )
        )

    def _save_configs(self, collection: Collection) -> None:
        existing = set(
            self._connection.scalars(
                select(t.chunking_configs.c.key).where(
                    t.chunking_configs.c.collection_id == collection.id
                )
            )
        )
        rows = [
            {
                "collection_id": collection.id,
                "key": config.key,
                "position": position,
                "strategy": config.strategy.value,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
            for position, config in enumerate(collection.chunking_configs)
            if config.key not in existing
        ]
        if rows:
            _write(self._connection, insert(t.chunking_configs), rows)

    def get(self, collection_id: UUID) -> Collection | None:
        row = self._connection.execute(
            select(t.collections).where(t.collections.c.id == collection_id)
        ).first()
        return self._to_domain(row) if row else None

    def get_by_name(self, name: str) -> Collection | None:
        row = self._connection.execute(
            select(t.collections).where(t.collections.c.name == name)
        ).first()
        return self._to_domain(row) if row else None

    def list(self) -> list[Collection]:
        rows = self._connection.execute(
            select(t.collections).order_by(t.collections.c.created_at, t.collections.c.id)
        )
        return [self._to_domain(row) for row in rows]

    def _to_domain(self, row: Row[Any]) -> Collection:
        configs = self._connection.execute(
            select(t.chunking_configs)
            .where(t.chunking_configs.c.collection_id == row.id)
            .order_by(t.chunking_configs.c.position)
        )
        return Collection(
            id=row.id,
            name=row.name,
            chunking_configs=tuple(
                ChunkingConfig(ChunkingStrategy(c.strategy), c.chunk_size, c.chunk_overlap)
                for c in configs
            ),
            embedding=(
                EmbeddingSpec(row.embedding_model, row.embedding_dimensions)
                if row.embedding_model is not None
                else None
            ),
            fulltext_language=row.fulltext_language,
            created_at=row.created_at,
        )


class PostgresDocumentRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, document: Document) -> None:
        _write(
            self._connection,
            insert(t.documents).values(
                id=document.id,
                collection_id=document.collection_id,
                filename=document.filename,
                media_type=document.media_type.value,
                content_hash=document.content_hash,
                text=document.text,
                metadata=dict(document.metadata),
                created_at=document.created_at,
            ),
        )
        if document.elements:
            _write(
                self._connection,
                insert(t.document_elements),
                [
                    {
                        "document_id": document.id,
                        "position": position,
                        "kind": element.kind.value,
                        "start_offset": element.span.start,
                        "end_offset": element.span.end,
                        "label": element.label,
                        "level": element.level,
                        "path": list(element.path),
                    }
                    for position, element in enumerate(document.elements)
                ],
            )

    def get(self, collection_id: UUID, document_id: UUID) -> Document | None:
        row = self._connection.execute(
            select(t.documents).where(
                t.documents.c.id == document_id,
                t.documents.c.collection_id == collection_id,
            )
        ).first()
        return self._to_domain(row) if row else None

    def list(self, collection_id: UUID) -> list[Document]:
        rows = self._connection.execute(
            select(t.documents)
            .where(t.documents.c.collection_id == collection_id)
            .order_by(t.documents.c.created_at, t.documents.c.id)
        )
        return [self._to_domain(row) for row in rows]

    def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        row = self._connection.execute(
            select(t.documents).where(
                t.documents.c.collection_id == collection_id,
                t.documents.c.content_hash == content_hash,
            )
        ).first()
        return self._to_domain(row) if row else None

    def _to_domain(self, row: Row[Any]) -> Document:
        elements = self._connection.execute(
            select(t.document_elements)
            .where(t.document_elements.c.document_id == row.id)
            .order_by(t.document_elements.c.position)
        )
        return Document(
            id=row.id,
            collection_id=row.collection_id,
            filename=row.filename,
            media_type=MediaType(row.media_type),
            content_hash=row.content_hash,
            text=row.text,
            elements=tuple(
                StructuralElement(
                    kind=ElementKind(e.kind),
                    span=Span(e.start_offset, e.end_offset),
                    label=e.label,
                    level=e.level,
                    path=tuple(e.path),
                )
                for e in elements
            ),
            metadata=dict(row.metadata),
            created_at=row.created_at,
        )


class PostgresChunkRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def replace(
        self, collection_id: UUID, config_key: str, document_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        self._connection.execute(
            delete(t.chunks).where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
                t.chunks.c.document_id == document_id,
            )
        )
        if not chunks:
            return
        language = self._connection.scalar(
            select(t.collections.c.fulltext_language).where(t.collections.c.id == collection_id)
        )
        statement = insert(t.chunks).values(
            search_vector=func.to_tsvector(
                literal(language or "simple", REGCONFIG), bindparam("search_text", type_=Text)
            )
        )
        _write(
            self._connection,
            statement,
            [
                {
                    "collection_id": collection_id,
                    "config_key": config_key,
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "start_offset": chunk.span.start,
                    "end_offset": chunk.span.end,
                    "text": chunk.text,
                    "page_numbers": list(chunk.page_numbers),
                    "section_path": list(chunk.section_path),
                    "search_text": chunk.text,
                }
                for chunk in chunks
            ],
        )

    def count(self, collection_id: UUID, config_key: str) -> int:
        return int(
            self._connection.scalar(
                select(func.count()).where(
                    t.chunks.c.collection_id == collection_id,
                    t.chunks.c.config_key == config_key,
                )
            )
            or 0
        )

    def list(
        self, collection_id: UUID, config_key: str, document_id: UUID | None = None
    ) -> list[Chunk]:
        query = (
            select(t.chunks)
            .join(t.documents, t.documents.c.id == t.chunks.c.document_id)
            .where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
            )
            .order_by(t.documents.c.created_at, t.documents.c.id, t.chunks.c.chunk_index)
        )
        if document_id is not None:
            query = query.where(t.chunks.c.document_id == document_id)
        return [_chunk_from_row(row) for row in self._connection.execute(query)]


def _chunk_from_row(row: Row[Any]) -> Chunk:
    return Chunk(
        document_id=row.document_id,
        collection_id=row.collection_id,
        index=row.chunk_index,
        span=Span(row.start_offset, row.end_offset),
        text=row.text,
        page_numbers=tuple(row.page_numbers),
        section_path=tuple(row.section_path),
    )


class PostgresEmbeddingRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def pending(self, collection_id: UUID, config_key: str, limit: int) -> list[Chunk]:
        query = (
            select(t.chunks)
            .join(t.documents, t.documents.c.id == t.chunks.c.document_id)
            .outerjoin(t.chunk_embeddings, t.chunk_embeddings.c.chunk_id == t.chunks.c.id)
            .where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
                t.chunk_embeddings.c.chunk_id.is_(None),
            )
            .order_by(t.documents.c.created_at, t.documents.c.id, t.chunks.c.chunk_index)
            .limit(limit)
        )
        return [_chunk_from_row(row) for row in self._connection.execute(query)]

    def save(
        self,
        collection_id: UUID,
        config_key: str,
        model: str,
        items: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> None:
        if not items:
            return
        positions = [(chunk.document_id, chunk.index) for chunk, _ in items]
        ids = {
            (row.document_id, row.chunk_index): row.id
            for row in self._connection.execute(
                select(t.chunks.c.id, t.chunks.c.document_id, t.chunks.c.chunk_index).where(
                    t.chunks.c.collection_id == collection_id,
                    t.chunks.c.config_key == config_key,
                    tuple_(t.chunks.c.document_id, t.chunks.c.chunk_index).in_(positions),
                )
            )
        }
        missing = [p for p in positions if p not in ids]
        if missing:
            raise ConflictError(f"{len(missing)} chunks no longer exist in {config_key!r}")
        _write(
            self._connection,
            insert(t.chunk_embeddings),
            [
                {
                    "chunk_id": ids[(chunk.document_id, chunk.index)],
                    "collection_id": collection_id,
                    "config_key": config_key,
                    "model": model,
                    "dimensions": len(vector),
                    "embedding": list(vector),
                }
                for chunk, vector in items
            ],
        )

    def count(self, collection_id: UUID, config_key: str) -> int:
        return int(
            self._connection.scalar(
                select(func.count()).where(
                    t.chunk_embeddings.c.collection_id == collection_id,
                    t.chunk_embeddings.c.config_key == config_key,
                )
            )
            or 0
        )

    def delete_collection(self, collection_id: UUID) -> None:
        self._connection.execute(
            delete(t.chunk_embeddings).where(t.chunk_embeddings.c.collection_id == collection_id)
        )


class PostgresSearchRepository:
    """Exact vector search (pgvector ``<=>``) and full-text search (``tsvector``)."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def vector(
        self,
        collection_id: UUID,
        config_key: str,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        # Exact scan over the chunk set: no ANN approximation in benchmark results.
        distance = t.chunk_embeddings.c.embedding.cosine_distance(
            cast(list(query_vector), Vector())
        )
        query = (
            select(t.chunks, (1 - distance).label("score"))
            .join(t.chunk_embeddings, t.chunk_embeddings.c.chunk_id == t.chunks.c.id)
            .where(
                t.chunk_embeddings.c.collection_id == collection_id,
                t.chunk_embeddings.c.config_key == config_key,
            )
            .order_by(distance)
        )
        return self._run(query, limit, document_ids)

    def fulltext(
        self,
        collection_id: UUID,
        config_key: str,
        query: str,
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        language = (
            select(t.collections.c.fulltext_language)
            .where(t.collections.c.id == collection_id)
            .scalar_subquery()
        )
        # plainto_tsquery ANDs every term, which is far too strict for questions;
        # rewriting "a & b" as "a | b" keeps its normalisation but matches any term.
        terms = cast(
            func.replace(
                cast(func.plainto_tsquery(cast(language, REGCONFIG), query), Text), " & ", " | "
            ),
            TSQUERY,
        )
        score = func.ts_rank_cd(t.chunks.c.search_vector, terms, 1)
        statement = (
            select(t.chunks, score.label("score"))
            .where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
                t.chunks.c.search_vector.op("@@")(terms),
            )
            .order_by(score.desc())
        )
        return self._run(statement, limit, document_ids)

    def _run(
        self, query: Select[Any], limit: int, document_ids: Set[UUID] | None
    ) -> list[ScoredChunk]:
        query = query.join(t.documents, t.documents.c.id == t.chunks.c.document_id).order_by(
            # Deterministic ties: reading order.
            t.documents.c.created_at,
            t.documents.c.id,
            t.chunks.c.chunk_index,
        )
        if document_ids is not None:
            condition: ColumnElement[bool] = t.chunks.c.document_id.in_(list(document_ids))
            query = query.where(condition)
        rows = self._connection.execute(query.limit(limit))
        return [ScoredChunk(_chunk_from_row(row), float(row.score)) for row in rows]


class PostgresGoldenQuestionRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, question: GoldenQuestion) -> None:
        _write(
            self._connection,
            insert(t.golden_questions).values(
                id=question.id,
                collection_id=question.collection_id,
                text=question.text,
                origin=question.origin.value,
                status=question.status.value,
                created_at=question.created_at,
            ),
        )
        _write(
            self._connection,
            insert(t.golden_evidence),
            [
                {
                    "question_id": question.id,
                    "position": position,
                    "document_id": evidence.document_id,
                    "start_offset": evidence.span.start,
                    "end_offset": evidence.span.end,
                }
                for position, evidence in enumerate(question.evidence)
            ],
        )

    def update(self, question: GoldenQuestion) -> None:
        self._connection.execute(
            update(t.golden_questions)
            .where(
                t.golden_questions.c.id == question.id,
                t.golden_questions.c.collection_id == question.collection_id,
            )
            .values(text=question.text, status=question.status.value)
        )

    def get(self, collection_id: UUID, question_id: UUID) -> GoldenQuestion | None:
        found = self._select(collection_id, [t.golden_questions.c.id == question_id])
        return found[0] if found else None

    def list(
        self, collection_id: UUID, status: QuestionStatus | None = None
    ) -> list[GoldenQuestion]:
        conditions = [] if status is None else [t.golden_questions.c.status == status.value]
        return self._select(collection_id, conditions)

    def delete(self, collection_id: UUID, question_id: UUID) -> bool:
        result = self._connection.execute(
            delete(t.golden_questions).where(
                t.golden_questions.c.id == question_id,
                t.golden_questions.c.collection_id == collection_id,
            )
        )
        return bool(result.rowcount)

    def _select(
        self, collection_id: UUID, conditions: "builtins.list[ColumnElement[bool]]"
    ) -> "builtins.list[GoldenQuestion]":
        rows = self._connection.execute(
            select(t.golden_questions)
            .where(t.golden_questions.c.collection_id == collection_id, *conditions)
            .order_by(t.golden_questions.c.created_at, t.golden_questions.c.id)
        ).all()
        if not rows:
            return []
        evidence: dict[UUID, builtins.list[Evidence]] = {}
        for e in self._connection.execute(
            select(t.golden_evidence)
            .where(t.golden_evidence.c.question_id.in_([row.id for row in rows]))
            .order_by(t.golden_evidence.c.question_id, t.golden_evidence.c.position)
        ):
            evidence.setdefault(e.question_id, []).append(
                Evidence(e.document_id, Span(e.start_offset, e.end_offset))
            )
        return [
            GoldenQuestion(
                id=row.id,
                collection_id=row.collection_id,
                text=row.text,
                evidence=tuple(evidence.get(row.id, ())),
                origin=QuestionOrigin(row.origin),
                status=QuestionStatus(row.status),
                created_at=row.created_at,
            )
            for row in rows
        ]


class PostgresBenchmarkRunRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, run: BenchmarkRun) -> None:
        _write(
            self._connection,
            insert(t.benchmark_runs).values(
                id=run.id,
                collection_id=run.collection_id,
                created_at=run.created_at,
                payload=run_to_payload(run),
            ),
        )

    def get(self, collection_id: UUID, run_id: UUID) -> BenchmarkRun | None:
        row = self._connection.execute(
            select(t.benchmark_runs).where(
                t.benchmark_runs.c.id == run_id,
                t.benchmark_runs.c.collection_id == collection_id,
            )
        ).first()
        return (
            run_from_payload(row.id, row.collection_id, row.created_at, row.payload)
            if row
            else None
        )

    def list(self, collection_id: UUID) -> list[BenchmarkRun]:
        rows = self._connection.execute(
            select(t.benchmark_runs)
            .where(t.benchmark_runs.c.collection_id == collection_id)
            .order_by(t.benchmark_runs.c.created_at.desc(), t.benchmark_runs.c.id)
        )
        return [run_from_payload(r.id, r.collection_id, r.created_at, r.payload) for r in rows]
