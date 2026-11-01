"""Golden dataset use cases: synthetic generation, manual questions, review, export/import."""

from uuid import uuid4

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import InvalidRequestError, NotFoundError
from rag_lab.application.services.golden import GoldenDatasetService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import QuestionOrigin, QuestionStatus
from tests.fakes import FakeChatModel

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)
POLICY = (
    b"Refunds are issued within 14 days of a return request. Items must be unused and in "
    b"their original packaging. Shipping costs are not refunded unless the item arrived "
    b"damaged or the wrong item was sent by the warehouse team.\n\n"
    b"Exchanges follow the same rules as refunds. A customer may exchange an item for a "
    b"different size or colour once per order, provided the replacement is in stock at the "
    b"time of the request and the original item is returned first."
)
SHIPPING = (
    b"Standard shipping takes three to five business days inside the country. Express "
    b"shipping arrives the next business day when the order is placed before noon. "
    b"International orders may take up to three weeks and can incur customs duties that "
    b"are paid by the customer on delivery."
)


def question_from_passage(prompt: str) -> str:
    passage = prompt.split("Passage:\n", 1)[1]
    return f"What does the policy say about {passage.split()[0].lower()}?"


class Lab:
    def __init__(self, reply: object = question_from_passage) -> None:
        self.uow = InMemoryUnitOfWork()
        self.chat = FakeChatModel(reply)  # type: ignore[arg-type]
        ingestion = IngestionService(uow_factory=lambda: self.uow, parser=ParserRegistry())
        self.collection = ingestion.create_collection("docs", (CONFIG,))
        self.policy = ingestion.ingest_document(self.collection.id, "policy.txt", POLICY).document
        self.shipping = ingestion.ingest_document(
            self.collection.id, "shipping.txt", SHIPPING
        ).document
        self.service = GoldenDatasetService(uow_factory=lambda: self.uow, chat=self.chat)


def test_generates_draft_questions_labelled_with_their_source_passage() -> None:
    lab = Lab()

    questions = lab.service.generate(lab.collection.id, count=3, seed=7)

    assert len(questions) == 3
    for question in questions:
        assert question.origin is QuestionOrigin.SYNTHETIC
        assert question.status is QuestionStatus.DRAFT
        [evidence] = question.evidence
        document = lab.policy if evidence.document_id == lab.policy.id else lab.shipping
        passage = document.text[evidence.span.start : evidence.span.end]
        assert f"Passage:\n{passage}" in lab.chat.prompts.pop(0)[-1].content
    assert {q.evidence[0].document_id for q in questions} == {lab.policy.id, lab.shipping.id}
    assert {q.id for q in lab.service.questions(lab.collection.id)} == {q.id for q in questions}


def test_generation_is_reproducible_for_a_seed_and_skips_used_passages() -> None:
    first, second = Lab(), Lab()

    a = first.service.generate(first.collection.id, count=2, seed=1)
    b = second.service.generate(second.collection.id, count=2, seed=1)
    more = first.service.generate(first.collection.id, count=10, seed=1)

    assert [q.evidence[0].span for q in a] == [q.evidence[0].span for q in b]
    used = {q.evidence[0].span for q in a}
    assert not used & {q.evidence[0].span for q in more}


def test_passages_the_model_skips_produce_no_question() -> None:
    lab = Lab(reply="SKIP")

    assert lab.service.generate(lab.collection.id, count=2, seed=0) == []


def test_manual_questions_locate_their_evidence_ignoring_whitespace() -> None:
    lab = Lab()

    question = lab.service.add_manual(
        lab.collection.id,
        "How long do refunds take?",
        lab.policy.id,
        "issued   within\n14 days",
    )

    [evidence] = question.evidence
    assert lab.policy.text[evidence.span.start : evidence.span.end] == "issued within 14 days"
    assert question.origin is QuestionOrigin.MANUAL
    with pytest.raises(InvalidRequestError, match="not found"):
        lab.service.add_manual(lab.collection.id, "Q?", lab.policy.id, "free unicorns")
    with pytest.raises(NotFoundError):
        lab.service.add_manual(lab.collection.id, "Q?", uuid4(), "refunds")


def test_review_edits_approves_rejects_and_deletes() -> None:
    lab = Lab()
    question = lab.service.add_manual(lab.collection.id, "Q?", lab.policy.id, "Refunds")

    edited = lab.service.update(lab.collection.id, question.id, text="Edited?")
    approved = lab.service.update(lab.collection.id, question.id, status=QuestionStatus.APPROVED)

    assert edited.text == "Edited?"
    assert approved.status is QuestionStatus.APPROVED
    assert lab.service.questions(lab.collection.id, QuestionStatus.APPROVED) == [approved]
    lab.service.delete(lab.collection.id, question.id)
    assert lab.service.questions(lab.collection.id) == []
    with pytest.raises(NotFoundError):
        lab.service.update(lab.collection.id, question.id, text="x")
    with pytest.raises(NotFoundError):
        lab.service.delete(lab.collection.id, question.id)


def test_export_and_import_move_a_dataset_between_collections_by_content_hash() -> None:
    source, target = Lab(), Lab()
    q = source.service.add_manual(source.collection.id, "Q?", source.policy.id, "14 days")
    source.service.update(source.collection.id, q.id, status=QuestionStatus.APPROVED)

    exported = source.service.export(source.collection.id)
    report = target.service.import_(target.collection.id, exported)

    [imported] = target.service.questions(target.collection.id)
    assert (report.imported, report.skipped) == (1, 0)
    assert imported.text == "Q?"
    assert imported.status is QuestionStatus.APPROVED
    assert imported.evidence[0].document_id == target.policy.id
    assert exported["questions"][0]["evidence"][0]["excerpt"] == "14 days"


def test_import_skips_questions_whose_documents_are_missing() -> None:
    lab = Lab()
    payload = {
        "format_version": 1,
        "questions": [
            {
                "text": "Q?",
                "status": "approved",
                "origin": "manual",
                "evidence": [
                    {"content_hash": "nope", "filename": "x.txt", "start": 0, "end": 5,
                     "excerpt": "Refun"}
                ],
            }
        ],
    }  # fmt: skip

    report = lab.service.import_(lab.collection.id, payload)

    assert (report.imported, report.skipped) == (0, 1)
    with pytest.raises(InvalidRequestError, match="format"):
        lab.service.import_(lab.collection.id, {"questions": []})
