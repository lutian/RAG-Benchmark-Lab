"""Retrieval use case: vector, full-text and hybrid search over one chunk set."""

from uuid import UUID, uuid4

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import ConflictError, InvalidRequestError, NotFoundError
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.application.services.retrieval import RetrievalService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.retrieval import SearchMethod
from tests.fakes import ConceptEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)
CONCEPTS = {
    "car": 0, "automobile": 0, "vehicle": 0,
    "bank": 1,
    "money": 2, "savings": 2, "interest": 2,
    "tires": 3,
    "river": 4, "fishing": 4,
}  # fmt: skip
DOCS = {
    "cars.txt": b"The automobile needs new tires before winter.",
    "money.txt": b"The bank raised interest rates on savings accounts.",
    "river.txt": b"Fishing on the river bank at dawn is quiet.",
}


class Lab:
    def __init__(self) -> None:
        self.uow = InMemoryUnitOfWork()
        self.embedder = ConceptEmbedder(CONCEPTS, dimensions=5)
        self.ingestion = IngestionService(uow_factory=lambda: self.uow, parser=ParserRegistry())
        self.collection = self.ingestion.create_collection("docs", (CONFIG,))
        self.documents = {
            name: self.ingestion.ingest_document(self.collection.id, name, data).document.id
            for name, data in DOCS.items()
        }
        self.retrieval = RetrievalService(
            uow_factory=lambda: self.uow, embedder=self.embedder, candidates=10, rrf_k=60
        )

    def index(self) -> None:
        IndexingService(uow_factory=lambda: self.uow, embedder=self.embedder).index_chunk_set(
            self.collection.id, CONFIG.key
        )

    def search(self, query: str, method: SearchMethod, **kwargs: object) -> list[UUID]:
        result = self.retrieval.search(self.collection.id, CONFIG.key, query, method, **kwargs)  # type: ignore[arg-type]
        return [hit.chunk.document_id for hit in result.hits]


@pytest.fixture
def lab() -> Lab:
    lab = Lab()
    lab.index()
    return lab


def test_full_text_matches_exact_terms_only(lab: Lab) -> None:
    assert lab.search("interest rates", SearchMethod.FULLTEXT) == [lab.documents["money.txt"]]
    # No document contains the word "car": lexical search cannot bridge synonyms.
    assert lab.search("car", SearchMethod.FULLTEXT) == []


def test_vector_search_matches_a_paraphrase(lab: Lab) -> None:
    hits = lab.search("car", SearchMethod.VECTOR)

    assert hits[0] == lab.documents["cars.txt"]
    assert len(hits) == 3  # vector search always ranks every candidate


def test_hybrid_fuses_both_rankings_and_reports_each_rank(lab: Lab) -> None:
    result = lab.retrieval.search(
        lab.collection.id, CONFIG.key, "bank savings", SearchMethod.HYBRID
    )

    top = result.hits[0]
    assert top.chunk.document_id == lab.documents["money.txt"]
    assert top.rank == 1
    assert top.ranks == {"vector": 1, "fulltext": 1}
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))


def test_single_method_hits_carry_their_own_rank_and_score(lab: Lab) -> None:
    result = lab.retrieval.search(lab.collection.id, CONFIG.key, "car", SearchMethod.VECTOR)

    assert [hit.ranks for hit in result.hits] == [{"vector": i} for i in (1, 2, 3)]
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_document_filter_and_top_k(lab: Lab) -> None:
    river = lab.documents["river.txt"]

    assert lab.search("bank", SearchMethod.HYBRID, document_ids={river}) == [river]
    assert len(lab.search("bank", SearchMethod.VECTOR, top_k=1)) == 1


def test_full_text_does_not_call_the_embedder_and_reports_timings(lab: Lab) -> None:
    result = lab.retrieval.search(lab.collection.id, CONFIG.key, "bank", SearchMethod.FULLTEXT)

    assert lab.embedder.queries == []
    assert result.timings.embed_ms == 0
    assert result.timings.total_ms >= result.timings.search_ms >= 0
    assert result.method is SearchMethod.FULLTEXT


def test_vector_search_requires_an_indexed_chunk_set() -> None:
    lab = Lab()

    with pytest.raises(ConflictError, match="not indexed"):
        lab.search("car", SearchMethod.VECTOR)
    # Full-text search needs no embeddings.
    assert lab.search("tires", SearchMethod.FULLTEXT) == [lab.documents["cars.txt"]]


def test_vector_search_refuses_a_partially_indexed_chunk_set(lab: Lab) -> None:
    lab.ingestion.ingest_document(lab.collection.id, "new.txt", b"A brand new vehicle.")

    with pytest.raises(ConflictError, match=r"3/4 chunks"):
        lab.search("car", SearchMethod.HYBRID)


def test_vector_search_refuses_another_embedding_model(lab: Lab) -> None:
    lab.retrieval = RetrievalService(
        uow_factory=lambda: lab.uow, embedder=ConceptEmbedder(CONCEPTS, 5, model="other")
    )

    with pytest.raises(ConflictError, match="indexed with concepts"):
        lab.search("car", SearchMethod.VECTOR)


def test_unknown_collection_or_chunk_set_is_not_found(lab: Lab) -> None:
    with pytest.raises(NotFoundError):
        lab.retrieval.search(uuid4(), CONFIG.key, "car", SearchMethod.FULLTEXT)
    with pytest.raises(NotFoundError, match="chunk set"):
        lab.retrieval.search(lab.collection.id, "nope", "car", SearchMethod.FULLTEXT)


def test_blank_query_and_bad_top_k_are_rejected(lab: Lab) -> None:
    with pytest.raises(InvalidRequestError, match="query"):
        lab.search("   ", SearchMethod.FULLTEXT)
    with pytest.raises(InvalidRequestError, match="top_k"):
        lab.search("car", SearchMethod.FULLTEXT, top_k=0)
