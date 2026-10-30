"""Chunks: retrievable units derived from a document by a chunking strategy."""

from dataclasses import dataclass
from uuid import UUID

from rag_lab.domain.documents import Document
from rag_lab.domain.spans import Span


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span of a document plus denormalised location metadata used for citations.

    ``text`` always equals ``document.text[span.start:span.end]``; build chunks with
    :meth:`from_document` so that invariant cannot be broken.
    """

    document_id: UUID
    collection_id: UUID
    index: int
    span: Span
    text: str
    page_numbers: tuple[int, ...]
    section_path: tuple[str, ...]

    @classmethod
    def from_document(cls, document: Document, span: Span, index: int) -> "Chunk":
        return cls(
            document_id=document.id,
            collection_id=document.collection_id,
            index=index,
            span=span,
            text=document.excerpt(span),
            page_numbers=document.page_numbers_for(span),
            section_path=document.section_path_for(span),
        )
