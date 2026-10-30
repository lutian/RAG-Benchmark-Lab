"""Text normalisation shared by every parser, so offsets are computed on identical text."""

import re

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


def normalize_text(text: str) -> str:
    """Unify line endings and drop NUL characters (PostgreSQL text cannot store them)."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def paragraph_starts(text: str) -> list[int]:
    """Offsets where paragraphs begin (first non-blank char, then after each blank line)."""
    first = len(text) - len(text.lstrip())
    starts = [first] if first < len(text) else []
    starts.extend(m.end() for m in _PARAGRAPH_BREAK.finditer(text) if m.end() < len(text))
    return starts
