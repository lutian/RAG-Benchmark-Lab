"""Chunk boundaries are rendered as alternating highlights over the escaped text."""

from rag_lab.ui.highlight import render_chunks


def test_alternates_colours_between_consecutive_chunks() -> None:
    html = render_chunks("aaabbb", [(0, 3), (3, 6)])

    assert html == (
        '<mark class="rl-chunk-a" title="chunk 0">aaa</mark>'
        '<mark class="rl-chunk-b" title="chunk 1">bbb</mark>'
    )


def test_overlapping_regions_get_their_own_class() -> None:
    html = render_chunks("abcdef", [(0, 4), (2, 6)])

    assert '<mark class="rl-overlap" title="chunks 0, 1">cd</mark>' in html


def test_text_outside_chunks_is_plain_and_html_is_escaped() -> None:
    html = render_chunks("<b> x", [(4, 5)])

    assert html.startswith("&lt;b&gt; ")
    assert html.endswith('<mark class="rl-chunk-a" title="chunk 0">x</mark>')


def test_query_terms_are_marked_as_evidence_case_insensitively() -> None:
    from rag_lab.ui.highlight import highlight_terms

    assert highlight_terms("Banks & the bank rate", "BANK rate") == (
        '<mark class="rl-evidence">Bank</mark>s &amp; the '
        '<mark class="rl-evidence">bank</mark> <mark class="rl-evidence">rate</mark>'
    )


def test_short_query_words_are_not_highlighted() -> None:
    from rag_lab.ui.highlight import highlight_terms

    assert highlight_terms("an apple <b>", "an") == "an apple &lt;b&gt;"
