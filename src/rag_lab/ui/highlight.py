"""Turn chunk spans into highlighted HTML so chunk boundaries can be inspected visually."""

import html
import itertools
import re
from collections.abc import Sequence

from rag_lab.domain.answers import CITATION_PATTERN


def render_chunks(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Render ``text`` with each chunk wrapped in a ``<mark>``.

    Consecutive chunks alternate between two colours; regions covered by more than
    one chunk (overlap) get a third class. Text is HTML-escaped.
    """
    boundaries = sorted({0, len(text), *(s for s, _ in spans), *(e for _, e in spans)})
    parts: list[str] = []
    for start, end in itertools.pairwise(boundaries):
        segment = html.escape(text[start:end], quote=False)
        covering = [i for i, (s, e) in enumerate(spans) if s <= start and end <= e]
        if not covering:
            parts.append(segment)
        elif len(covering) == 1:
            css = "rl-chunk-a" if covering[0] % 2 == 0 else "rl-chunk-b"
            parts.append(f'<mark class="{css}" title="chunk {covering[0]}">{segment}</mark>')
        else:
            label = ", ".join(str(i) for i in covering)
            parts.append(f'<mark class="rl-overlap" title="chunks {label}">{segment}</mark>')
    return "".join(parts)


def as_preview_block(fragment: str) -> str:
    """Wrap rendered HTML in a single-line block so Markdown never re-interprets it."""
    single_line = fragment.replace("\n", "<br>").replace("$", "&#36;")
    return f'<div class="rl-preview">{single_line}</div>'


def highlight_terms(text: str, query: str, min_length: int = 3) -> str:
    """Escape ``text`` and mark every occurrence of the query's words (prefix match).

    Words shorter than ``min_length`` are skipped so "a"/"of" do not light up the page.
    This is a visual aid, not the full-text matcher: it does no stemming.
    """
    words = sorted(
        {w for w in re.findall(r"\w+", query.lower()) if len(w) >= min_length},
        key=len,
        reverse=True,
    )
    if not words:
        return html.escape(text, quote=False)
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
    parts: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[position : match.start()], quote=False))
        parts.append(f'<mark class="rl-evidence">{html.escape(match.group(), quote=False)}</mark>')
        position = match.end()
    parts.append(html.escape(text[position:], quote=False))
    return "".join(parts)


def render_answer(text: str, invalid: set[int]) -> str:
    """Escape an answer and turn ``[n]`` citations into badges.

    Uses the domain's citation pattern, so badges match the API's citation verdict;
    a group containing a number from ``invalid`` (as reported by the API) is flagged.
    """
    parts: list[str] = []
    position = 0
    for match in CITATION_PATTERN.finditer(text):
        parts.append(html.escape(text[position : match.start()], quote=False))
        numbers = {int(n) for n in match.group(1).split(",")}
        label = html.escape(match.group(), quote=False)
        if not numbers & invalid:
            parts.append(f'<span class="rl-cite">{label}</span>')
        else:
            parts.append(
                f'<span class="rl-cite invalid" title="not a provided source">{label}</span>'
            )
        position = match.end()
    parts.append(html.escape(text[position:], quote=False))
    return "".join(parts)
