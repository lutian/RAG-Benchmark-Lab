"""Grounded answers: numbered citations and abstention.

The LLM sees the retrieved chunks as numbered sources and must cite them as ``[n]``
(or ``[n, m]``). When the sources do not contain the answer it must reply with the
:data:`ABSTENTION` sentinel. Both rules are checked here with plain parsing, so the
"valid citations" metric is deterministic and costs nothing to compute.
"""

import re
from dataclasses import dataclass

ABSTENTION = "NO_ANSWER"

# "[3]" or "[1, 2]": numbers only and not glued to a word, so "list[0]" or "[1.5]" are
# not citations.
CITATION_PATTERN = re.compile(r"(?<!\w)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def extract_citations(text: str) -> tuple[int, ...]:
    """Cited source numbers, unique, in order of first appearance."""
    seen: dict[int, None] = {}
    for match in CITATION_PATTERN.finditer(text):
        for number in match.group(1).split(","):
            seen.setdefault(int(number), None)
    return tuple(seen)


_LEADING_MARKUP = re.compile(r"^[\s*_>#`\"']+")


def is_abstention(text: str) -> bool:
    """The reply starts with the sentinel, ignoring leading whitespace (any Unicode
    space) and markdown/quote characters: ``* _ > # ` " '``."""
    return _LEADING_MARKUP.sub("", text).upper().startswith(ABSTENTION)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited: tuple[int, ...]
    # Cited numbers that do not match any provided source (hallucinated references).
    invalid: tuple[int, ...]

    @property
    def valid(self) -> bool:
        """At least one citation, and every citation points to a provided source."""
        return bool(self.cited) and not self.invalid


def check_citations(text: str, source_count: int) -> CitationCheck:
    cited = extract_citations(text)
    return CitationCheck(cited, tuple(n for n in cited if not 1 <= n <= source_count))
