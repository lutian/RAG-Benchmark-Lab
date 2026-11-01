"""Retrieval domain: search methods, ranked hits and reciprocal rank fusion."""

from uuid import uuid4

import pytest

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import FULLTEXT_LANGUAGES, Collection
from rag_lab.domain.retrieval import SearchMethod, reciprocal_rank_fusion
from rag_lab.domain.spans import Span

DOCUMENT = uuid4()
COLLECTION = uuid4()


def chunk(index: int) -> Chunk:
    return Chunk(
        document_id=DOCUMENT,
        collection_id=COLLECTION,
        index=index,
        span=Span(index * 10, index * 10 + 10),
        text=f"chunk {index}",
        page_numbers=(),
        section_path=(),
    )


def test_search_methods_are_the_four_compared_strategies() -> None:
    assert [m.value for m in SearchMethod] == ["vector", "fulltext", "hybrid", "hybrid_rerank"]


def test_fusion_rewards_chunks_ranked_high_by_both_methods() -> None:
    a, b, c = chunk(0), chunk(1), chunk(2)

    fused = reciprocal_rank_fusion({"vector": [a, b], "fulltext": [b, c]}, k=60, limit=10)

    assert [hit.chunk for hit in fused] == [b, a, c]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ranks == {"vector": 2, "fulltext": 1}
    assert fused[1].ranks == {"vector": 1}


def test_fusion_breaks_ties_by_first_ranking_then_reading_order() -> None:
    a, b = chunk(0), chunk(1)

    fused = reciprocal_rank_fusion({"vector": [b], "fulltext": [a]}, k=60, limit=10)

    # Equal scores: the chunk from the first ranking wins.
    assert [hit.chunk for hit in fused] == [b, a]


def test_fusion_truncates_to_limit_and_validates_arguments() -> None:
    chunks = [chunk(i) for i in range(5)]

    assert len(reciprocal_rank_fusion({"vector": chunks}, k=60, limit=3)) == 3
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion({"vector": chunks}, k=0, limit=3)
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion({"vector": chunks}, k=60, limit=0)


def test_collection_full_text_language_defaults_to_english_and_is_validated() -> None:
    assert Collection.create("docs").fulltext_language == "english"
    assert Collection.create("docs", fulltext_language="portuguese").fulltext_language == (
        "portuguese"
    )
    assert {"simple", "english", "portuguese", "spanish"} <= FULLTEXT_LANGUAGES
    with pytest.raises(ValueError, match="full-text language"):
        Collection.create("docs", fulltext_language="klingon")
