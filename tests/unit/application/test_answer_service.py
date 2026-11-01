"""Answer use case: retrieve, build numbered context, generate, check citations."""

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import GenerationProviderError, InvalidRequestError
from rag_lab.application.services.answering import AnswerService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.application.services.retrieval import RetrievalService
from rag_lab.domain.answers import ABSTENTION
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.retrieval import SearchMethod
from tests.fakes import FakeChatModel, FakeEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)


class Lab:
    def __init__(self, chat: FakeChatModel) -> None:
        self.uow = InMemoryUnitOfWork()
        self.chat = chat
        ingestion = IngestionService(uow_factory=lambda: self.uow, parser=ParserRegistry())
        self.collection = ingestion.create_collection("docs", (CONFIG,))
        ingestion.ingest_document(self.collection.id, "rates.txt", b"The bank raised rates.")
        ingestion.ingest_document(self.collection.id, "river.txt", b"The river bank is quiet.")
        retrieval = RetrievalService(uow_factory=lambda: self.uow, embedder=FakeEmbedder())
        self.service = AnswerService(retrieval=retrieval, chat=chat)

    def ask(self, question: str, **kwargs: object):  # type: ignore[no-untyped-def]
        return self.service.answer(
            self.collection.id,
            CONFIG.key,
            question,
            method=kwargs.pop("method", SearchMethod.FULLTEXT),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )


def test_answer_cites_numbered_sources_and_checks_them() -> None:
    lab = Lab(FakeChatModel("The bank raised its rates [1]."))

    answer = lab.ask("Who raised rates?")

    assert answer.text == "The bank raised its rates [1]."
    assert not answer.abstained
    assert answer.citation_check.valid
    [citation] = answer.citations
    assert citation.number == 1
    assert citation.hit.chunk.text == "The bank raised rates."
    assert answer.sources[0].chunk.text == "The bank raised rates."
    assert answer.model == "fake-chat-1"
    assert (answer.prompt_tokens, answer.completion_tokens) == (10, 5)


def test_prompt_numbers_sources_and_states_the_rules() -> None:
    lab = Lab(FakeChatModel("ok [1]"))

    lab.ask("Who raised rates?")

    [system, user] = lab.chat.prompts[0]
    assert system.role == "system"
    assert ABSTENTION in system.content
    assert "[1]" in system.content
    assert user.content.startswith("Sources:")
    assert "[1]" in user.content and "The bank raised rates." in user.content
    assert user.content.rstrip().endswith("Question: Who raised rates?")


def test_citations_to_unknown_sources_make_the_answer_invalid() -> None:
    lab = Lab(FakeChatModel("Rates rose [1][5]."))

    answer = lab.ask("Who raised rates?")

    assert not answer.citation_check.valid
    assert answer.citation_check.invalid == (5,)
    assert [c.number for c in answer.citations] == [1]


def test_model_abstention_is_reported_as_a_generation_decision() -> None:
    lab = Lab(FakeChatModel(f"{ABSTENTION}: the sources do not mention dates."))

    answer = lab.ask("When did the bank raise rates?")

    assert answer.abstained
    assert answer.abstention_reason == "model_abstained"
    assert answer.citations == ()


def test_no_retrieved_evidence_abstains_without_calling_the_llm() -> None:
    lab = Lab(FakeChatModel("should not be called"))

    answer = lab.ask("unicorns?")

    assert answer.abstained
    assert answer.abstention_reason == "no_evidence_retrieved"
    assert answer.sources == ()
    assert lab.chat.prompts == []
    assert answer.timings.generation_ms == 0


def test_timings_split_retrieval_and_generation() -> None:
    lab = Lab(FakeChatModel("Yes [1]."))

    answer = lab.ask("bank", top_k=1)

    assert len(answer.sources) == 1
    assert answer.timings.retrieval.total_ms >= 0
    assert answer.timings.generation_ms >= 0
    assert answer.timings.total_ms >= answer.timings.generation_ms


def test_llm_failures_propagate_and_blank_questions_are_rejected() -> None:
    with pytest.raises(GenerationProviderError, match="outage"):
        Lab(FakeChatModel(fail=True)).ask("bank")
    with pytest.raises(InvalidRequestError, match="query"):
        Lab(FakeChatModel("x")).ask("  ")
