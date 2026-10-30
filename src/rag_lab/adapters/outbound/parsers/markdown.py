"""Markdown parser: ATX headings become sections with heading breadcrumbs."""

import re

from rag_lab.adapters.outbound.parsers.normalize import normalize_text
from rag_lab.adapters.outbound.parsers.text import decode
from rag_lab.application.ports.parsing import ParsedDocument
from rag_lab.domain.documents import ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _front_matter(text: str) -> dict[str, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}
    pairs = (line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    return {key.strip(): value.strip().strip("\"'") for key, value in pairs if key.strip()}


def _headings(text: str) -> list[tuple[int, int, str]]:
    """``(offset, level, title)`` for every heading outside fenced code blocks."""
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    offset = 0
    for line in text.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            fence = None if fence == marker else (fence or marker)
        elif fence is None and (heading := _HEADING.match(line)):
            headings.append((offset, len(heading.group(1)), heading.group(2)))
        offset += len(line) + 1
    return headings


class MarkdownParser:
    """Sections span from a heading to the next heading of any level.

    The parent chain is kept in ``path`` so a chunk inside "Docker" under "Install"
    can be cited as ``Install > Docker``.
    """

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        raw, encoding = decode(data)
        text = normalize_text(raw)
        headings = _headings(text)

        elements: list[StructuralElement] = []
        stack: list[tuple[int, str]] = []
        ends = [offset for offset, _, _ in headings[1:]] + [len(text)]
        for (start, level, title), end in zip(headings, ends, strict=True):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            elements.append(
                StructuralElement(
                    ElementKind.SECTION,
                    Span(start, end),
                    label=title,
                    level=level,
                    path=tuple(name for _, name in stack),
                )
            )

        metadata = {"encoding": encoding, **_front_matter(text)}
        if "title" not in metadata:
            first_h1 = next((title for _, level, title in headings if level == 1), None)
            metadata["title"] = first_h1 or filename.rsplit(".", 1)[0]

        return ParsedDocument(
            media_type=MediaType.MARKDOWN,
            text=text,
            elements=tuple(elements),
            metadata=metadata,
        )
