"""Turn chunk spans into highlighted HTML so chunk boundaries can be inspected visually."""

import html
import itertools
from collections.abc import Sequence


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
