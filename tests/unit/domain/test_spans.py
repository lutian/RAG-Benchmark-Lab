"""Spans are half-open character intervals over a document's canonical text."""

import pytest

from rag_lab.domain.spans import Span


def test_length_is_end_minus_start() -> None:
    assert Span(10, 25).length == 15


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (7, 3)])
def test_rejects_negative_or_empty_spans(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="span"):
        Span(start, end)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (Span(0, 10), Span(5, 15), 5),
        (Span(0, 10), Span(10, 20), 0),  # touching, half-open: no overlap
        (Span(0, 10), Span(2, 4), 2),
        (Span(5, 6), Span(0, 100), 1),
        (Span(0, 3), Span(7, 9), 0),
    ],
)
def test_overlap_counts_shared_characters(a: Span, b: Span, expected: int) -> None:
    assert a.overlap(b) == expected
    assert b.overlap(a) == expected


def test_contains_offset_is_half_open() -> None:
    span = Span(3, 6)
    assert span.contains(3)
    assert span.contains(5)
    assert not span.contains(6)
    assert not span.contains(2)
