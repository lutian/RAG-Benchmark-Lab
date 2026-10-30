"""Documents hold the canonical text that every chunking and golden span refers to."""

from uuid import uuid4

import pytest

from rag_lab.domain.documents import (
    Document,
    ElementKind,
    MediaType,
    StructuralElement,
)
from rag_lab.domain.spans import Span

TEXT = "# Intro\nHello world.\n## Details\nMore text here.\n"
#       0123456 7890123456789 0123456789 0123456789012345
INTRO = StructuralElement(ElementKind.SECTION, Span(0, 21), "Intro", level=1, path=("Intro",))
DETAILS = StructuralElement(
    ElementKind.SECTION, Span(21, 48), "Details", level=2, path=("Intro", "Details")
)


def make_document(**overrides: object) -> Document:
    fields: dict[str, object] = {
        "id": uuid4(),
        "collection_id": uuid4(),
        "filename": "notes.md",
        "media_type": MediaType.MARKDOWN,
        "content_hash": "abc",
        "text": TEXT,
        "elements": (INTRO, DETAILS),
        "metadata": {"title": "Notes"},
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def test_excerpt_returns_canonical_text_slice() -> None:
    document = make_document()

    assert document.excerpt(Span(8, 20)) == "Hello world."


def test_rejects_elements_outside_text_bounds() -> None:
    outside = StructuralElement(ElementKind.PAGE, Span(0, 999), "Page 1", level=0, path=())

    with pytest.raises(ValueError, match="outside"):
        make_document(elements=(outside,))


def test_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_document(text="   \n", elements=())


def test_section_path_is_the_deepest_section_containing_the_span_start() -> None:
    document = make_document()

    assert document.section_path_for(Span(25, 40)) == ("Intro", "Details")
    assert document.section_path_for(Span(2, 30)) == ("Intro",)


def test_page_numbers_lists_every_page_the_span_touches() -> None:
    pages = (
        StructuralElement(ElementKind.PAGE, Span(0, 20), "1", level=0, path=()),
        StructuralElement(ElementKind.PAGE, Span(20, 48), "2", level=0, path=()),
    )
    document = make_document(media_type=MediaType.PDF, elements=pages)

    assert document.page_numbers_for(Span(5, 10)) == (1,)
    assert document.page_numbers_for(Span(15, 30)) == (1, 2)


def test_documents_without_pages_report_no_page_numbers() -> None:
    assert make_document().page_numbers_for(Span(0, 5)) == ()
