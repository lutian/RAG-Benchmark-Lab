from rag_lab.adapters.outbound.parsers.markdown import MarkdownParser
from rag_lab.domain.documents import ElementKind, MediaType

DOC = """---
title: Install Guide
author: Ana
---
Preamble text.

# Install
Run the installer.

## Docker
```bash
# not a heading
docker compose up
```

# FAQ
Questions.
"""


def sections(text: str) -> dict[str, tuple[tuple[str, ...], str]]:
    parsed = MarkdownParser().parse("guide.md", text.encode())
    return {
        e.label: (e.path, parsed.text[e.span.start : e.span.end])
        for e in parsed.elements
        if e.kind is ElementKind.SECTION
    }


def test_sections_follow_atx_headings_with_breadcrumb_paths() -> None:
    result = sections(DOC)

    assert list(result) == ["Install", "Docker", "FAQ"]
    assert result["Install"] == (("Install",), "# Install\nRun the installer.\n\n")
    assert result["Docker"][0] == ("Install", "Docker")
    assert result["FAQ"] == (("FAQ",), "# FAQ\nQuestions.\n")


def test_headings_inside_code_fences_are_ignored() -> None:
    assert "not a heading" not in sections(DOC)


def test_front_matter_feeds_metadata() -> None:
    parsed = MarkdownParser().parse("guide.md", DOC.encode())

    assert parsed.metadata["title"] == "Install Guide"
    assert parsed.metadata["author"] == "Ana"
    assert parsed.media_type is MediaType.MARKDOWN


def test_title_defaults_to_first_h1() -> None:
    parsed = MarkdownParser().parse("x.md", b"intro\n# Real Title\nbody\n")

    assert parsed.metadata["title"] == "Real Title"
