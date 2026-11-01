"""Golden dataset use cases: build, review and share the questions the benchmark uses.

* **Synthetic generation.** Each document's canonical text is cut into passages with
  the recursive chunker (independent of the collection's chunk sets). Passages are
  sampled with a seeded RNG, round-robin across documents, skipping passages already
  used as evidence. The LLM writes one question per passage; the passage span becomes
  the evidence. Every question starts as a **draft**.
* **Manual questions.** A reviewer types a question and pastes the evidence excerpt;
  it is located in the canonical text (whitespace-insensitive).
* **Review.** Edit the text, approve or reject. Only approved questions are benchmarked.
* **Export / import.** Evidence is written with the document *content hash*, so a
  dataset can be imported into another collection or database holding the same files.
"""

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from rag_lab.application.errors import InvalidRequestError, NotFoundError
from rag_lab.application.ports.generation import ChatModel
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.prompts import SKIP, question_messages
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy, chunk_document
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document
from rag_lab.domain.evaluation import (
    Evidence,
    GoldenQuestion,
    QuestionOrigin,
    QuestionStatus,
    locate_excerpt,
)
from rag_lab.domain.spans import Span

EXPORT_FORMAT = 1
MAX_IMPORT_QUESTIONS = 5_000
MAX_EXCERPT_CHARS = 20_000  # same limit as a manual excerpt; bounds locate_excerpt


@dataclass(frozen=True, slots=True)
class EvidenceView:
    document_id: UUID
    filename: str
    span: Span
    excerpt: str


@dataclass(frozen=True, slots=True)
class GoldenQuestionView:
    """A question with its evidence resolved to filenames and text, for review screens."""

    question: GoldenQuestion
    evidence: tuple[EvidenceView, ...]


@dataclass(frozen=True, slots=True)
class ImportReport:
    imported: int
    # Questions whose evidence points at documents missing from the target collection.
    skipped: int


class GoldenDatasetService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        chat: ChatModel,
        passage_chars: int = 400,
        min_passage_chars: int = 100,
    ) -> None:
        self._uow_factory = uow_factory
        self._chat = chat
        self._passage = ChunkingConfig(ChunkingStrategy.RECURSIVE, passage_chars, 0)
        self._min_passage_chars = min_passage_chars

    # --- generation ----------------------------------------------------------------
    def generate(self, collection_id: UUID, count: int, seed: int = 0) -> list[GoldenQuestion]:
        """Ask the LLM for up to ``count`` questions; each is committed as soon as it exists."""
        if count <= 0:
            raise InvalidRequestError("count must be positive")
        passages = self._sample_passages(collection_id, count, seed)
        created = []
        for evidence, text in passages:
            reply = self._chat.complete(question_messages(text)).text.strip().strip('"')
            if not reply or reply.upper().startswith(SKIP):
                continue
            question = self._new(collection_id, reply, (evidence,), QuestionOrigin.SYNTHETIC)
            with self._uow_factory() as uow:
                uow.golden.add(question)
                uow.commit()
            created.append(question)
        return created

    def _sample_passages(
        self, collection_id: UUID, count: int, seed: int
    ) -> list[tuple[Evidence, str]]:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            documents = sorted(uow.documents.list(collection_id), key=lambda d: d.created_at)
            used = [e for q in uow.golden.list(collection_id) for e in q.evidence]
        rng = random.Random(seed)
        groups = []
        for document in documents:
            passages = [
                (Evidence(document.id, chunk.span), chunk.text)
                for chunk in chunk_document(document, self._passage)
                if chunk.span.length >= self._min_passage_chars
                and not any(
                    e.document_id == document.id and e.span.overlap(chunk.span) for e in used
                )
            ]
            rng.shuffle(passages)
            groups.append(passages)
        rng.shuffle(groups)
        # Round-robin across documents so a long document does not dominate the dataset.
        sampled: list[tuple[Evidence, str]] = []
        while len(sampled) < count and any(groups):
            for group in groups:
                if group and len(sampled) < count:
                    sampled.append(group.pop())
        return sampled

    # --- manual questions and review ----------------------------------------------
    def add_manual(
        self, collection_id: UUID, text: str, document_id: UUID, excerpt: str
    ) -> GoldenQuestion:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            document = uow.documents.get(collection_id, document_id)
            if document is None:
                raise NotFoundError(f"document {document_id} not found in this collection")
            span = locate_excerpt(document.text, excerpt)
            if span is None:
                raise InvalidRequestError(
                    f"evidence excerpt not found in {document.filename!r}; paste text "
                    "copied from the document"
                )
            question = self._new(
                collection_id, text, (Evidence(document_id, span),), QuestionOrigin.MANUAL
            )
            uow.golden.add(question)
            uow.commit()
        return question

    def questions(
        self, collection_id: UUID, status: QuestionStatus | None = None
    ) -> list[GoldenQuestion]:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            return uow.golden.list(collection_id, status)

    def views(
        self, collection_id: UUID, status: QuestionStatus | None = None
    ) -> list[GoldenQuestionView]:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            questions = uow.golden.list(collection_id, status)
            documents = {d.id: d for d in uow.documents.list(collection_id)}
        return [self._view(q, documents) for q in questions]

    def view(self, collection_id: UUID, question: GoldenQuestion) -> GoldenQuestionView:
        with self._uow_factory() as uow:
            documents = {d.id: d for d in uow.documents.list(collection_id)}
        return self._view(question, documents)

    @staticmethod
    def _view(question: GoldenQuestion, documents: Mapping[UUID, Document]) -> GoldenQuestionView:
        return GoldenQuestionView(
            question,
            tuple(
                EvidenceView(
                    e.document_id,
                    documents[e.document_id].filename,
                    e.span,
                    documents[e.document_id].text[e.span.start : e.span.end],
                )
                for e in question.evidence
            ),
        )

    def update(
        self,
        collection_id: UUID,
        question_id: UUID,
        text: str | None = None,
        status: QuestionStatus | None = None,
    ) -> GoldenQuestion:
        with self._uow_factory() as uow:
            question = uow.golden.get(collection_id, question_id)
            if question is None:
                raise NotFoundError(f"golden question {question_id} not found")
            try:
                if text is not None:
                    question = question.with_text(text)
            except ValueError as error:
                raise InvalidRequestError(str(error)) from error
            if status is not None:
                question = question.with_status(status)
            uow.golden.update(question)
            uow.commit()
        return question

    def delete(self, collection_id: UUID, question_id: UUID) -> None:
        with self._uow_factory() as uow:
            if not uow.golden.delete(collection_id, question_id):
                raise NotFoundError(f"golden question {question_id} not found")
            uow.commit()

    # --- export / import ----------------------------------------------------------
    def export(self, collection_id: UUID) -> dict[str, Any]:
        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            documents = {d.id: d for d in uow.documents.list(collection_id)}
            questions = uow.golden.list(collection_id)
        return {
            "format_version": EXPORT_FORMAT,
            "collection": collection.name,
            "questions": [
                {
                    "text": q.text,
                    "status": q.status.value,
                    "origin": q.origin.value,
                    "evidence": [
                        {
                            "content_hash": documents[e.document_id].content_hash,
                            "filename": documents[e.document_id].filename,
                            "start": e.span.start,
                            "end": e.span.end,
                            "excerpt": documents[e.document_id].text[e.span.start : e.span.end],
                        }
                        for e in q.evidence
                    ],
                }
                for q in questions
            ],
        }

    def import_(self, collection_id: UUID, payload: Mapping[str, Any]) -> ImportReport:
        """Import an exported dataset; any malformed entry rejects the whole payload (422)."""
        if payload.get("format_version") != EXPORT_FORMAT:
            raise InvalidRequestError(
                f"unsupported golden dataset format (expected {EXPORT_FORMAT})"
            )
        items = payload.get("questions", [])
        if not isinstance(items, list) or len(items) > MAX_IMPORT_QUESTIONS:
            raise InvalidRequestError(
                f"'questions' must be a list of at most {MAX_IMPORT_QUESTIONS} entries"
            )
        imported = skipped = 0
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            by_hash = {d.content_hash: d for d in uow.documents.list(collection_id)}
            for position, item in enumerate(items, start=1):
                try:
                    question = self._imported_question(collection_id, item, by_hash)
                except (KeyError, TypeError, ValueError) as error:
                    raise InvalidRequestError(
                        f"question #{position} is malformed: {error}"
                    ) from error
                if question is None:
                    skipped += 1
                    continue
                uow.golden.add(question)
                imported += 1
            uow.commit()
        return ImportReport(imported, skipped)

    def _imported_question(
        self, collection_id: UUID, item: Mapping[str, Any], by_hash: Mapping[str, Document]
    ) -> GoldenQuestion | None:
        """The question with evidence resolved in this collection, or None to skip it.

        Skipped: a document is missing, or the excerpt is found neither at the exported
        offsets nor anywhere else in the document (stale offsets are never kept).
        """
        evidence = []
        for e in item["evidence"]:
            document = by_hash.get(e["content_hash"])
            excerpt = str(e["excerpt"])
            # Malformed, not merely missing: an oversized excerpt is never a valid export
            # (exports come from spans of real chunks), so it rejects the payload.
            if len(excerpt) > MAX_EXCERPT_CHARS:
                raise ValueError(f"excerpt longer than {MAX_EXCERPT_CHARS} characters")
            if document is None:
                return None
            start, end = int(e["start"]), int(e["end"])
            if 0 <= start < end <= len(document.text) and document.text[start:end] == excerpt:
                span: Span | None = Span(start, end)
            else:
                # Same file, different parsing: find the text instead.
                span = locate_excerpt(document.text, excerpt)
            if span is None:
                return None
            evidence.append(Evidence(document.id, span))
        question = self._new(
            collection_id,
            str(item["text"]),
            tuple(evidence),
            QuestionOrigin(item.get("origin", "manual")),
        )
        return question.with_status(QuestionStatus(item.get("status", "draft")))

    # --- helpers ------------------------------------------------------------------
    @staticmethod
    def _new(
        collection_id: UUID, text: str, evidence: tuple[Evidence, ...], origin: QuestionOrigin
    ) -> GoldenQuestion:
        try:
            return GoldenQuestion.create(collection_id, text, evidence, origin)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error

    @staticmethod
    def _require_collection(uow: UnitOfWork, collection_id: UUID) -> Collection:
        collection = uow.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"collection {collection_id} not found")
        return collection
