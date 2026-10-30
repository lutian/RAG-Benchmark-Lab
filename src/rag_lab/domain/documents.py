"""Documents and their structure: the chunking-independent reference layer.

A :class:`Document` stores one canonical text produced once at ingestion time.
Pages and sections are recorded as :class:`StructuralElement` spans over that text,
so any chunking strategy (and any golden label) can be mapped back to pages and
headings without re-parsing the original file.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from rag_lab.domain.spans import Span


class MediaType(StrEnum):
    PDF = "application/pdf"
    MARKDOWN = "text/markdown"
    TEXT = "text/plain"


class ElementKind(StrEnum):
    PAGE = "page"
    SECTION = "section"


@dataclass(frozen=True, slots=True)
class StructuralElement:
    """A page or a section of the canonical text.

    Sections cover only their own content (up to the next heading of any level);
    ``path`` carries the heading breadcrumb, e.g. ``("Install", "Docker")``.
    For pages, ``label`` is the 1-based page number.
    """

    kind: ElementKind
    span: Span
    label: str
    level: int
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    collection_id: UUID
    filename: str
    media_type: MediaType
    content_hash: str
    text: str
    elements: tuple[StructuralElement, ...]
    metadata: Mapping[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"document {self.filename!r} has empty text")
        for element in self.elements:
            if element.span.end > len(self.text):
                raise ValueError(
                    f"element {element.label!r} span {element.span} is outside the text "
                    f"(length {len(self.text)})"
                )

    def excerpt(self, span: Span) -> str:
        return self.text[span.start : span.end]

    def section_path_for(self, span: Span) -> tuple[str, ...]:
        """Heading breadcrumb of the deepest section containing ``span.start``."""
        containing = [
            element
            for element in self.elements
            if element.kind is ElementKind.SECTION and element.span.contains(span.start)
        ]
        if not containing:
            return ()
        return max(containing, key=lambda element: element.level).path

    def page_numbers_for(self, span: Span) -> tuple[int, ...]:
        """1-based numbers of every page the span overlaps (empty if paginated text absent)."""
        return tuple(
            int(element.label)
            for element in self.elements
            if element.kind is ElementKind.PAGE and element.span.overlap(span) > 0
        )
