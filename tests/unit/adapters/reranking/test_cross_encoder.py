"""The local cross-encoder adapter, with the model replaced by a recording fake."""

from collections.abc import Sequence

import pytest

from rag_lab.adapters.outbound.reranking.cross_encoder import CrossEncoderReranker
from rag_lab.application.errors import RerankerError


class FakeModel:
    def __init__(self) -> None:
        self.pairs: list[list[tuple[str, str]]] = []

    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: object) -> list[float]:
        self.pairs.append(list(pairs))
        return [float(len(text)) for _, text in pairs]


def test_scores_query_text_pairs_and_loads_the_model_once() -> None:
    loads: list[str] = []
    model = FakeModel()

    def load(name: str) -> FakeModel:
        loads.append(name)
        return model

    reranker = CrossEncoderReranker("cross-encoder/tiny", loader=load)

    assert loads == []  # lazy: nothing is loaded until the first query
    assert reranker.score("q", ["a", "bbb"]) == [1.0, 3.0]
    assert reranker.score("q", ["cc"]) == [2.0]
    assert loads == ["cross-encoder/tiny"]
    assert model.pairs[0] == [("q", "a"), ("q", "bbb")]
    assert reranker.name == "cross-encoder/tiny"


def test_empty_input_does_not_load_the_model() -> None:
    def load(name: str) -> FakeModel:
        raise AssertionError("must not load")

    assert CrossEncoderReranker("m", loader=load).score("q", []) == []


def test_loading_failures_are_reranker_errors() -> None:
    def load(name: str) -> FakeModel:
        raise ModuleNotFoundError("No module named 'sentence_transformers'")

    with pytest.raises(RerankerError, match="sentence_transformers"):
        CrossEncoderReranker("m", loader=load).score("q", ["a"])
