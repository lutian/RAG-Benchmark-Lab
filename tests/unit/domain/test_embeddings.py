"""A collection is bound to the embedding model and dimension used to index it."""

import pytest

from rag_lab.domain.collections import Collection, EmbeddingMismatchError
from rag_lab.domain.embeddings import EmbeddingSpec

SMALL = EmbeddingSpec("text-embedding-3-small", 1536)
NOMIC = EmbeddingSpec("nomic-embed-text", 768)


@pytest.mark.parametrize(("model", "dimensions"), [("", 768), ("  ", 768), ("m", 0), ("m", -1)])
def test_spec_requires_model_and_positive_dimensions(model: str, dimensions: int) -> None:
    with pytest.raises(ValueError, match="embedding"):
        EmbeddingSpec(model, dimensions)


def test_new_collections_are_unbound() -> None:
    assert Collection.create("docs").embedding is None


def test_binding_an_unbound_collection_records_the_spec() -> None:
    bound = Collection.create("docs").bind_embedding(SMALL)

    assert bound.embedding == SMALL


def test_binding_the_same_spec_again_is_a_no_op() -> None:
    bound = Collection.create("docs").bind_embedding(SMALL)

    assert bound.bind_embedding(SMALL) == bound


def test_binding_a_different_spec_is_rejected_with_guidance() -> None:
    bound = Collection.create("docs").bind_embedding(SMALL)

    with pytest.raises(EmbeddingMismatchError, match="rebuild"):
        bound.bind_embedding(NOMIC)


def test_unbinding_allows_a_new_spec() -> None:
    bound = Collection.create("docs").bind_embedding(SMALL)

    assert bound.unbind_embedding().bind_embedding(NOMIC).embedding == NOMIC
