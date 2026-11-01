"""Retrieval: search one chunk set with the selected method. Scoped by ``collection_id``."""

from uuid import UUID

from fastapi import APIRouter

from rag_lab.adapters.inbound.api.routes.collections import ContainerDep
from rag_lab.adapters.inbound.api.schemas import SearchIn, SearchOut

router = APIRouter(prefix="/collections", tags=["retrieval"])


@router.post("/{collection_id}/search")
def search(collection_id: UUID, body: SearchIn, container: ContainerDep) -> SearchOut:
    """Rank chunks of ``chunk_set`` for ``query``.

    ``vector``, ``hybrid`` and ``hybrid_rerank`` need the chunk set fully indexed with
    the configured embedding model (409 otherwise); ``fulltext`` works right after
    upload. ``hybrid_rerank`` reorders the best ``RERANK_CANDIDATES`` hybrid hits with
    the configured reranker (502 if it fails).
    """
    result = container.retrieval.search(
        collection_id,
        body.chunk_set,
        body.query,
        body.method,
        top_k=body.top_k,
        document_ids=set(body.document_ids) if body.document_ids is not None else None,
    )
    return SearchOut.from_result(result)
