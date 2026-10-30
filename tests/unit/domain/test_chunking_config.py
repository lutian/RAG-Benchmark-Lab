"""Chunking configurations are validated value objects with a stable key."""

import pytest

from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy


def test_key_identifies_strategy_and_parameters() -> None:
    config = ChunkingConfig(ChunkingStrategy.RECURSIVE, chunk_size=800, chunk_overlap=120)

    assert config.key == "recursive-800-120"


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_rejects_invalid_size_or_overlap(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        ChunkingConfig(ChunkingStrategy.FIXED, chunk_size=size, chunk_overlap=overlap)


def test_parses_from_key() -> None:
    config = ChunkingConfig.from_key("structure-1200-0")

    assert config == ChunkingConfig(ChunkingStrategy.STRUCTURE, 1200, 0)
