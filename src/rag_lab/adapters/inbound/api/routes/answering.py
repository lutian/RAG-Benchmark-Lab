"""Answering: retrieve, generate a cited answer or abstain. Scoped by ``collection_id``."""

from uuid import UUID

from fastapi import APIRouter

from rag_lab.adapters.inbound.api.routes.collections import ContainerDep
from rag_lab.adapters.inbound.api.schemas import AnswerIn, AnswerOut

router = APIRouter(prefix="/collections", tags=["answering"])


@router.post("/{collection_id}/answer")
def answer(collection_id: UUID, body: AnswerIn, container: ContainerDep) -> AnswerOut:
    """Answer ``question`` from the ``top_k`` chunks retrieved with ``method``.

    The answer cites sources as ``[n]``; ``citations_valid`` is computed by parsing,
    not by another LLM. ``abstained`` is true when nothing was retrieved or the model
    found the sources insufficient (``abstention_reason`` says which). 502 when the
    LLM provider fails.
    """
    result = container.answering.answer(
        collection_id,
        body.chunk_set,
        body.question,
        method=body.method,
        top_k=body.top_k,
        document_ids=set(body.document_ids) if body.document_ids is not None else None,
    )
    return AnswerOut.from_answer(result)
