"""Plain-text parser."""

from rag_lab.adapters.outbound.parsers.normalize import normalize_text, paragraph_starts
from rag_lab.application.ports.parsing import ParsedDocument
from rag_lab.domain.documents import ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span


def decode(data: bytes) -> tuple[str, str]:
    """Decode as UTF-8 (BOM tolerated), falling back to Latin-1 which never fails."""
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


class TextParser:
    """Keeps the text as-is (normalised) and records paragraphs as structure."""

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        raw, encoding = decode(data)
        text = normalize_text(raw)
        starts = paragraph_starts(text)
        elements = tuple(
            StructuralElement(
                ElementKind.PARAGRAPH,
                Span(start, end),
                label=f"Paragraph {number}",
                level=1,
                path=(),
            )
            for number, (start, end) in enumerate(
                zip(starts, [*starts[1:], len(text)], strict=True), start=1
            )
        )
        return ParsedDocument(
            media_type=MediaType.TEXT,
            text=text,
            elements=elements,
            metadata={"encoding": encoding, "title": filename.rsplit(".", 1)[0]},
        )
