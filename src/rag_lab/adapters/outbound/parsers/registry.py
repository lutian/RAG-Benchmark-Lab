"""Chooses the parser from the file extension and validates the raw bytes first."""

from pathlib import PurePath

from rag_lab.adapters.outbound.parsers.markdown import MarkdownParser
from rag_lab.adapters.outbound.parsers.pdf import PdfParser
from rag_lab.adapters.outbound.parsers.text import TextParser
from rag_lab.application.errors import DocumentParsingError, UnsupportedDocumentError
from rag_lab.application.ports.parsing import DocumentParser, ParsedDocument

PDF_SIGNATURE = b"%PDF-"


class ParserRegistry:
    """DocumentParser that dispatches by extension (``.pdf``, ``.md``, ``.markdown``, ``.txt``)."""

    def __init__(self) -> None:
        markdown = MarkdownParser()
        self._parsers: dict[str, DocumentParser] = {
            ".pdf": PdfParser(),
            ".md": markdown,
            ".markdown": markdown,
            ".txt": TextParser(),
        }

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(self._parsers)

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        extension = PurePath(filename).suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise UnsupportedDocumentError(
                f"unsupported file type {extension or '(none)'!r} for {filename!r}; "
                f"supported: {', '.join(self._parsers)}"
            )
        if not data.strip():
            raise DocumentParsingError(f"file {filename!r} is empty")
        if extension == ".pdf" and not data.lstrip().startswith(PDF_SIGNATURE):
            raise DocumentParsingError(f"file {filename!r} is not a PDF (missing %PDF- header)")
        parsed = parser.parse(filename, data)
        if not parsed.text.strip():
            raise DocumentParsingError(f"file {filename!r} has no extractable text")
        return parsed
