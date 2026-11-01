"""Golden dataset: generate, add, review and share evaluation questions."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Response, status

from rag_lab.adapters.inbound.api.routes.collections import ContainerDep
from rag_lab.adapters.inbound.api.schemas import (
    GenerateQuestionsIn,
    GoldenQuestionOut,
    ImportReportOut,
    ManualQuestionIn,
    QuestionUpdateIn,
)
from rag_lab.domain.evaluation import QuestionStatus

router = APIRouter(prefix="/collections/{collection_id}/golden", tags=["golden dataset"])


@router.get("")
def list_questions(
    collection_id: UUID, container: ContainerDep, status: QuestionStatus | None = None
) -> list[GoldenQuestionOut]:
    return [GoldenQuestionOut.from_view(v) for v in container.golden.views(collection_id, status)]


@router.post("", status_code=status.HTTP_201_CREATED)
def add_question(
    collection_id: UUID, body: ManualQuestionIn, container: ContainerDep
) -> GoldenQuestionOut:
    """Add your own question; the excerpt is located in the document (422 if absent)."""
    question = container.golden.add_manual(
        collection_id, body.question, body.document_id, body.excerpt
    )
    return GoldenQuestionOut.from_view(container.golden.view(collection_id, question))


@router.post("/generate", status_code=status.HTTP_201_CREATED)
def generate_questions(
    collection_id: UUID, body: GenerateQuestionsIn, container: ContainerDep
) -> list[GoldenQuestionOut]:
    """Ask the LLM for up to ``count`` draft questions from seeded passages (LLM_* settings)."""
    created = container.golden.generate(collection_id, body.count, body.seed)
    return [GoldenQuestionOut.from_view(container.golden.view(collection_id, q)) for q in created]


@router.patch("/{question_id}")
def update_question(
    collection_id: UUID, question_id: UUID, body: QuestionUpdateIn, container: ContainerDep
) -> GoldenQuestionOut:
    """Edit the text and/or set the status (draft, approved, rejected)."""
    question = container.golden.update(collection_id, question_id, body.text, body.status)
    return GoldenQuestionOut.from_view(container.golden.view(collection_id, question))


@router.delete("/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_question(collection_id: UUID, question_id: UUID, container: ContainerDep) -> Response:
    container.golden.delete(collection_id, question_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/export")
def export_dataset(collection_id: UUID, container: ContainerDep) -> dict[str, Any]:
    """Portable JSON: evidence identified by document content hash, span and excerpt."""
    return container.golden.export(collection_id)


@router.post("/import")
def import_dataset(
    collection_id: UUID, container: ContainerDep, payload: Annotated[dict[str, Any], Body()]
) -> ImportReportOut:
    """Import an exported dataset; questions whose documents are missing are skipped."""
    return ImportReportOut.from_report(container.golden.import_(collection_id, payload))
