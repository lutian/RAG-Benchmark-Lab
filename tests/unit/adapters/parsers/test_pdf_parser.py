from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.pdf import PdfParser
from rag_lab.application.errors import DocumentParsingError
from rag_lab.domain.documents import ElementKind, MediaType


def test_each_page_becomes_a_page_element_over_the_canonical_text(
    make_pdf: Callable[..., bytes],
) -> None:
    data = make_pdf(["Alpha page content.", "Beta page content."], title="Quarterly")

    parsed = PdfParser().parse("q.pdf", data)

    pages = [e for e in parsed.elements if e.kind is ElementKind.PAGE]
    assert [p.label for p in pages] == ["1", "2"]
    assert "Alpha page content." in parsed.text[pages[0].span.start : pages[0].span.end]
    assert "Beta page content." in parsed.text[pages[1].span.start : pages[1].span.end]
    assert parsed.media_type is MediaType.PDF
    assert parsed.metadata["page_count"] == "2"
    assert parsed.metadata["title"] == "Quarterly"


def test_blank_pages_keep_numbering_of_following_pages(make_pdf: Callable[..., bytes]) -> None:
    parsed = PdfParser().parse("gap.pdf", make_pdf(["First.", "", "Third."]))

    labels = [e.label for e in parsed.elements if e.kind is ElementKind.PAGE]
    assert labels == ["1", "3"]


def test_pdf_without_extractable_text_is_rejected(make_pdf: Callable[..., bytes]) -> None:
    with pytest.raises(DocumentParsingError, match="no extractable text"):
        PdfParser().parse("scan.pdf", make_pdf(["", ""]))


def test_corrupted_pdf_is_rejected() -> None:
    with pytest.raises(DocumentParsingError, match="could not read"):
        PdfParser().parse("bad.pdf", b"%PDF-1.7 garbage")
