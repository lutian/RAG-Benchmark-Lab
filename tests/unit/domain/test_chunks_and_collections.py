"""Chunks are spans of a document enriched with the metadata of where they come from."""

from uuid import uuid4

import pytest

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document, ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span

TEXT = "Page one text. Page two text."


def make_pdf() -> Document:
    return Document(
        id=uuid4(),
        collection_id=uuid4(),
        filename="report.pdf",
        media_type=MediaType.PDF,
        content_hash="h",
        text=TEXT,
        elements=(
            StructuralElement(ElementKind.PAGE, Span(0, 15), "1", level=0, path=()),
            StructuralElement(ElementKind.PAGE, Span(15, 29), "2", level=0, path=()),
        ),
        metadata={},
    )


def test_chunk_from_document_copies_text_and_location_metadata() -> None:
    document = make_pdf()

    chunk = Chunk.from_document(document, Span(10, 20), index=3)

    assert chunk.text == TEXT[10:20]
    assert chunk.document_id == document.id
    assert chunk.collection_id == document.collection_id
    assert chunk.index == 3
    assert chunk.page_numbers == (1, 2)
    assert chunk.section_path == ()


def test_collection_name_is_trimmed_and_required() -> None:
    assert Collection.create("  Contracts 2025 ").name == "Contracts 2025"
    with pytest.raises(ValueError, match="name"):
        Collection.create("   ")


def test_collection_rejects_duplicate_chunking_configs() -> None:
    from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy

    config = ChunkingConfig(ChunkingStrategy.FIXED, 100, 0)

    with pytest.raises(ValueError, match="unique"):
        Collection.create("c", (config, config))
    with pytest.raises(ValueError, match="already exists"):
        Collection.create("c", (config,)).with_chunking_config(config)
