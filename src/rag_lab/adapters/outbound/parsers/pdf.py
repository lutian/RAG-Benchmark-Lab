"""PDF parser based on pypdf: one page element per page with extractable text."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_lab.adapters.outbound.parsers.normalize import normalize_text
from rag_lab.application.errors import DocumentParsingError
from rag_lab.application.ports.parsing import ParsedDocument
from rag_lab.domain.documents import ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span

PAGE_SEPARATOR = "\n\n"


class PdfParser:
    """Concatenates page texts into one canonical text and records each page's span.

    Pages without extractable text (blank or scanned) produce no element, but the
    remaining pages keep their real page numbers so citations stay correct.
    OCR is out of scope: a PDF with no text layer at all is rejected.
    """

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                reader.decrypt("")
            page_texts = [normalize_text(page.extract_text() or "") for page in reader.pages]
        except (PdfReadError, ValueError, KeyError, TypeError) as error:
            raise DocumentParsingError(f"could not read PDF {filename!r}: {error}") from error

        parts: list[str] = []
        elements: list[StructuralElement] = []
        offset = 0
        for number, page_text in enumerate(page_texts, start=1):
            content = page_text.strip()
            if not content:
                continue
            if parts:
                parts.append(PAGE_SEPARATOR)
                offset += len(PAGE_SEPARATOR)
            parts.append(content)
            elements.append(
                StructuralElement(
                    ElementKind.PAGE,
                    Span(offset, offset + len(content)),
                    label=str(number),
                    level=0,
                    path=(),
                )
            )
            offset += len(content)

        if not elements:
            raise DocumentParsingError(
                f"PDF {filename!r} has no extractable text (scanned PDFs need OCR)"
            )

        metadata = {"page_count": str(len(page_texts))}
        info = reader.metadata
        if info is not None:
            if info.title:
                metadata["title"] = str(info.title)
            if info.author:
                metadata["author"] = str(info.author)
        metadata.setdefault("title", filename.rsplit(".", 1)[0])

        return ParsedDocument(
            media_type=MediaType.PDF,
            text="".join(parts),
            elements=tuple(elements),
            metadata=metadata,
        )
