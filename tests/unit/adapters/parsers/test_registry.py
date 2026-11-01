from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.errors import DocumentParsingError, UnsupportedDocumentError
from rag_lab.domain.documents import MediaType


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("a.txt", MediaType.TEXT), ("a.md", MediaType.MARKDOWN), ("A.MARKDOWN", MediaType.MARKDOWN)],
)
def test_dispatches_text_formats_by_extension(filename: str, expected: MediaType) -> None:
    assert ParserRegistry().parse(filename, b"# hello\nworld").media_type is expected


def test_dispatches_pdf(make_pdf: Callable[..., bytes]) -> None:
    assert ParserRegistry().parse("r.pdf", make_pdf(["Hi."])).media_type is MediaType.PDF


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedDocumentError, match=r"\.docx"):
        ParserRegistry().parse("contract.docx", b"PK...")


def test_rejects_pdf_extension_without_pdf_signature() -> None:
    with pytest.raises(DocumentParsingError, match="not a PDF"):
        ParserRegistry().parse("fake.pdf", b"just text")


def test_rejects_empty_files() -> None:
    with pytest.raises(DocumentParsingError, match="empty"):
        ParserRegistry().parse("a.txt", b"  \n ")
