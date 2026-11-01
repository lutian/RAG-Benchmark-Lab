"""Chunking strategies as pure functions from (text, structure, config) to spans.

All strategies share three guarantees, enforced by property-based tests:

* a chunk is always an exact slice of the canonical text (offsets never drift);
* no chunk is longer than ``chunk_size`` characters, and none is whitespace-only;
* every non-whitespace character of the document belongs to at least one chunk.
"""

from collections.abc import Sequence

from rag_lab.domain.chunking.config import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.documents import Document, ElementKind, StructuralElement
from rag_lab.domain.spans import Span

# Coarse to fine: paragraphs, lines, sentences, words, then raw characters.
RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Apply ``config`` to ``document`` and return chunks in reading order."""
    whole = Span(0, len(document.text))
    match config.strategy:
        case ChunkingStrategy.FIXED:
            spans = fixed_spans(document.text, config)
        case ChunkingStrategy.RECURSIVE:
            spans = _merge(_atoms(document.text, whole, config.chunk_size), config)
        case ChunkingStrategy.STRUCTURE:
            spans = structure_spans(document.text, document.elements, config)
    finished = _trim_and_dedupe(document.text, spans)
    return [Chunk.from_document(document, span, index) for index, span in enumerate(finished)]


def fixed_spans(text: str, config: ChunkingConfig) -> list[Span]:
    """Sliding windows of ``chunk_size`` characters, stepping ``size - overlap``.

    Deliberately naive: it ignores words and sentences. It is the baseline the other
    strategies are compared against.
    """
    step = config.chunk_size - config.chunk_overlap
    spans: list[Span] = []
    start = 0
    while start < len(text):
        end = min(start + config.chunk_size, len(text))
        spans.append(Span(start, end))
        if end == len(text):
            break
        start += step
    return spans


def structure_spans(
    text: str, elements: Sequence[StructuralElement], config: ChunkingConfig
) -> list[Span]:
    """Respect the document's own structure.

    * Sections (Markdown headings) and pages (PDF) are hard boundaries: a chunk never
      spans two of them; oversized units are split recursively inside the unit.
    * Paragraphs (plain text) are soft boundaries: small consecutive paragraphs are
      merged up to ``chunk_size``.
    * Without structure the strategy degrades to recursive chunking.
    """
    for kind, hard in (
        (ElementKind.SECTION, True),
        (ElementKind.PAGE, True),
        (ElementKind.PARAGRAPH, False),
    ):
        selected = sorted((e.span for e in elements if e.kind is kind), key=lambda s: s.start)
        if selected:
            units = _units_with_gaps(selected, len(text))
            if hard:
                return [
                    span
                    for unit in units
                    for span in _merge(_atoms(text, unit, config.chunk_size), config)
                ]
            atoms = [atom for unit in units for atom in _atoms(text, unit, config.chunk_size)]
            return _merge(atoms, config)
    return _merge(_atoms(text, Span(0, len(text)), config.chunk_size), config)


def _units_with_gaps(spans: Sequence[Span], text_length: int) -> list[Span]:
    """Element spans plus the uncovered gaps between them, so no text is ever lost."""
    units: list[Span] = []
    cursor = 0
    for span in spans:
        start = max(span.start, cursor)
        if start > cursor:
            units.append(Span(cursor, start))
        if span.end > start:
            units.append(Span(start, span.end))
            cursor = span.end
    if cursor < text_length:
        units.append(Span(cursor, text_length))
    return units


def _atoms(
    text: str, span: Span, size: int, separators: Sequence[str] = RECURSIVE_SEPARATORS
) -> list[Span]:
    """Split ``span`` into contiguous pieces no longer than ``size``.

    Uses the coarsest separator present; pieces that are still too long are split
    again with the finer separators. Separators stay attached to the preceding piece.
    """
    if span.length <= size:
        return [span]
    for position, separator in enumerate(separators):
        if separator == "":
            break
        pieces = _split_on(text, span, separator)
        if len(pieces) > 1:
            return [
                atom
                for piece in pieces
                for atom in _atoms(text, piece, size, separators[position + 1 :])
            ]
    return [Span(s, min(s + size, span.end)) for s in range(span.start, span.end, size)]


def _split_on(text: str, span: Span, separator: str) -> list[Span]:
    pieces: list[Span] = []
    start = span.start
    found = text.find(separator, start, span.end)
    while found != -1:
        end = found + len(separator)
        if end >= span.end:
            break
        pieces.append(Span(start, end))
        start = end
        found = text.find(separator, start, span.end)
    pieces.append(Span(start, span.end))
    return pieces


def _merge(atoms: Sequence[Span], config: ChunkingConfig) -> list[Span]:
    """Greedily pack contiguous atoms into chunks of at most ``chunk_size``.

    When a chunk is closed, its trailing atoms (up to ``chunk_overlap`` characters)
    are carried into the next chunk to preserve context across the cut.
    """
    size, overlap = config.chunk_size, config.chunk_overlap
    chunks: list[Span] = []
    current: list[Span] = []
    current_length = 0
    for atom in atoms:
        if current and current_length + atom.length > size:
            chunks.append(Span(current[0].start, current[-1].end))
            tail: list[Span] = []
            tail_length = 0
            while current:
                candidate = current[-1].length
                if tail_length + candidate > overlap or (
                    tail_length + candidate + atom.length > size
                ):
                    break
                tail.insert(0, current.pop())
                tail_length += candidate
            current, current_length = tail, tail_length
        current.append(atom)
        current_length += atom.length
    if current:
        chunks.append(Span(current[0].start, current[-1].end))
    return chunks


def _trim_and_dedupe(text: str, spans: Sequence[Span]) -> list[Span]:
    """Strip edge whitespace, drop whitespace-only spans and consecutive duplicates."""
    result: list[Span] = []
    for span in spans:
        segment = text[span.start : span.end]
        stripped = segment.strip()
        if not stripped:
            continue
        start = span.start + (len(segment) - len(segment.lstrip()))
        trimmed = Span(start, start + len(stripped))
        if not result or result[-1] != trimmed:
            result.append(trimmed)
    return result
