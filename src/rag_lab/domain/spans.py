"""Character spans: the unit of reference shared by chunks, citations and golden labels.

A span is a half-open interval ``[start, end)`` over a document's canonical text.
Because chunks, citations and golden "relevant passages" are all expressed as spans
of the same text, they stay comparable no matter which chunking strategy produced
the chunks.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """Half-open character interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end}): need 0 <= start < end")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: "Span") -> int:
        """Number of characters shared with ``other`` (0 when disjoint or touching)."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.end
