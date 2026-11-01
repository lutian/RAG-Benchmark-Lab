"""Chunking strategies cut the canonical text into spans; they never alter the text."""

from uuid import uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy, chunk_document
from rag_lab.domain.documents import Document, ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span


def make_document(text: str, elements: tuple[StructuralElement, ...] = ()) -> Document:
    return Document(
        id=uuid4(),
        collection_id=uuid4(),
        filename="doc.txt",
        media_type=MediaType.TEXT,
        content_hash="h",
        text=text,
        elements=elements,
        metadata={},
    )


def section(start: int, end: int, title: str) -> StructuralElement:
    return StructuralElement(ElementKind.SECTION, Span(start, end), title, level=1, path=(title,))


# --- fixed -----------------------------------------------------------------------------


def test_fixed_windows_advance_by_size_minus_overlap() -> None:
    document = make_document("abcdefghij" * 3)  # 30 chars
    config = ChunkingConfig(ChunkingStrategy.FIXED, chunk_size=12, chunk_overlap=2)

    spans = [chunk.span for chunk in chunk_document(document, config)]

    assert spans == [Span(0, 12), Span(10, 22), Span(20, 30)]


# --- recursive -------------------------------------------------------------------------


def test_recursive_prefers_paragraph_boundaries() -> None:
    text = "Alpha one two.\n\nBeta three four.\n\nGamma five six."
    config = ChunkingConfig(ChunkingStrategy.RECURSIVE, chunk_size=20, chunk_overlap=0)

    texts = [chunk.text for chunk in chunk_document(make_document(text), config)]

    assert texts == ["Alpha one two.", "Beta three four.", "Gamma five six."]


def test_recursive_falls_back_to_words_inside_long_paragraphs() -> None:
    text = "one two three four five six seven eight"
    config = ChunkingConfig(ChunkingStrategy.RECURSIVE, chunk_size=15, chunk_overlap=0)

    texts = [chunk.text for chunk in chunk_document(make_document(text), config)]

    assert texts == ["one two three", "four five six", "seven eight"]
    assert all(" " not in (t[0], t[-1]) for t in texts)


def test_recursive_overlap_repeats_trailing_pieces() -> None:
    text = "aa bb cc dd ee ff"
    config = ChunkingConfig(ChunkingStrategy.RECURSIVE, chunk_size=9, chunk_overlap=3)

    texts = [chunk.text for chunk in chunk_document(make_document(text), config)]

    assert texts == ["aa bb cc", "cc dd ee", "ee ff"]


# --- structure -------------------------------------------------------------------------


def test_structure_never_crosses_section_boundaries() -> None:
    text = "# A\nshort a.\n# B\nshort b.\n"
    elements = (section(0, 13, "A"), section(13, 26, "B"))
    config = ChunkingConfig(ChunkingStrategy.STRUCTURE, chunk_size=500, chunk_overlap=0)

    chunks = chunk_document(make_document(text, elements), config)

    assert [c.text for c in chunks] == ["# A\nshort a.", "# B\nshort b."]
    assert [c.section_path for c in chunks] == [("A",), ("B",)]


def test_structure_splits_oversized_sections_recursively() -> None:
    body = "word " * 40
    text = "# Big\n" + body
    config = ChunkingConfig(ChunkingStrategy.STRUCTURE, chunk_size=50, chunk_overlap=0)

    chunks = chunk_document(make_document(text, (section(0, len(text), "Big"),)), config)

    assert len(chunks) > 1
    assert all(c.span.length <= 50 for c in chunks)
    assert all(c.section_path == ("Big",) for c in chunks)


def test_structure_keeps_text_outside_sections() -> None:
    text = "Preamble here.\n# A\nbody.\n"
    config = ChunkingConfig(ChunkingStrategy.STRUCTURE, chunk_size=500, chunk_overlap=0)

    chunks = chunk_document(make_document(text, (section(15, len(text), "A"),)), config)

    assert [c.text for c in chunks] == ["Preamble here.", "# A\nbody."]


def test_structure_merges_small_paragraphs_up_to_chunk_size() -> None:
    text = "p1.\n\np2.\n\np3."
    paragraphs = tuple(
        StructuralElement(ElementKind.PARAGRAPH, Span(s, e), f"P{i}", level=1, path=())
        for i, (s, e) in enumerate([(0, 5), (5, 10), (10, 13)], start=1)
    )
    config = ChunkingConfig(ChunkingStrategy.STRUCTURE, chunk_size=10, chunk_overlap=0)

    texts = [c.text for c in chunk_document(make_document(text, paragraphs), config)]

    assert texts == ["p1.\n\np2.", "p3."]


def test_structure_without_elements_behaves_like_recursive() -> None:
    text = "Alpha one two.\n\nBeta three four."
    structure = ChunkingConfig(ChunkingStrategy.STRUCTURE, 20, 0)
    recursive = ChunkingConfig(ChunkingStrategy.RECURSIVE, 20, 0)
    document = make_document(text)

    assert [c.span for c in chunk_document(document, structure)] == [
        c.span for c in chunk_document(document, recursive)
    ]


def test_chunks_are_numbered_in_reading_order() -> None:
    config = ChunkingConfig(ChunkingStrategy.FIXED, 5, 0)

    chunks = chunk_document(make_document("x" * 23), config)

    assert [c.index for c in chunks] == list(range(len(chunks)))


# --- invariants for every strategy -----------------------------------------------------

texts = st.text(alphabet=st.sampled_from("ab cd\n.é"), min_size=1, max_size=400).filter(
    lambda t: t.strip() != ""
)
configs = st.builds(
    lambda strategy, size, overlap_ratio: ChunkingConfig(
        strategy, size, int((size - 1) * overlap_ratio)
    ),
    st.sampled_from(list(ChunkingStrategy)),
    st.integers(min_value=1, max_value=120),
    st.floats(min_value=0, max_value=0.9),
)


@settings(max_examples=300, deadline=None)
@given(text=texts, config=configs)
def test_invariants_hold_for_any_text_and_config(text: str, config: ChunkingConfig) -> None:
    document = make_document(text, (section(0, len(text), "All"),))

    chunks = chunk_document(document, config)

    covered = set[int]()
    for chunk in chunks:
        assert chunk.text == text[chunk.span.start : chunk.span.end]
        assert chunk.span.length <= config.chunk_size
        assert chunk.text.strip(), "whitespace-only chunk"
        covered.update(range(chunk.span.start, chunk.span.end))
    non_whitespace = {i for i, char in enumerate(text) if not char.isspace()}
    assert non_whitespace <= covered, "some content is not covered by any chunk"
    starts = [c.span.start for c in chunks]
    assert starts == sorted(starts)
