"""Parsing port: turns raw file bytes into canonical text plus structure."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from rag_lab.domain.documents import MediaType, StructuralElement


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser output: the canonical text and its structure, before identity is assigned."""

    media_type: MediaType
    text: str
    elements: tuple[StructuralElement, ...]
    metadata: Mapping[str, str]


class DocumentParser(Protocol):
    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        """Parse ``data``; raise UnsupportedDocumentError or DocumentParsingError on failure."""
        ...
