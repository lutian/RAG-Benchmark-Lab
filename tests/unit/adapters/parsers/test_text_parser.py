from rag_lab.adapters.outbound.parsers.text import TextParser
from rag_lab.domain.documents import MediaType


def test_normalises_line_endings_and_strips_nul_characters() -> None:
    parsed = TextParser().parse("notes.txt", b"line one\r\nline\x00 two\rline three")

    assert parsed.text == "line one\nline two\nline three"
    assert parsed.media_type is MediaType.TEXT


def test_falls_back_to_latin1_when_not_utf8() -> None:
    parsed = TextParser().parse("legacy.txt", "café".encode("latin-1"))

    assert parsed.text == "café"
    assert parsed.metadata["encoding"] == "latin-1"


def test_paragraphs_are_recorded_as_sections() -> None:
    parsed = TextParser().parse("a.txt", b"First para.\n\nSecond para.\nstill second.")

    spans = [parsed.text[e.span.start : e.span.end] for e in parsed.elements]
    assert spans == ["First para.\n\n", "Second para.\nstill second."]
