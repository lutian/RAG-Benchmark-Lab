"""The rag-lab CLI runs benchmarks and writes reports (in-memory container)."""

import json
from pathlib import Path

import pytest

from rag_lab.bootstrap import Container, build_in_memory_container
from rag_lab.cli import main
from rag_lab.config import Settings
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import QuestionStatus
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 40, 0)


@pytest.fixture
def container() -> Container:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    c = build_in_memory_container(settings, FakeEmbedder(), FakeReranker(), FakeChatModel("ok [1]"))
    collection = c.ingestion.create_collection("handbook", (CONFIG,))
    document = c.ingestion.ingest_document(
        collection.id, "faq.txt", b"Refunds are issued within 14 days. Shipping is free."
    ).document
    c.indexing.index_chunk_set(collection.id, CONFIG.key)
    question = c.golden.add_manual(
        collection.id, "When are refunds issued?", document.id, "14 days"
    )
    c.golden.update(collection.id, question.id, status=QuestionStatus.APPROVED)
    return c


def test_benchmark_writes_markdown_and_json_and_prints_the_best_strategy(
    container: Container, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["benchmark", "--collection", "handbook", "--method", "fulltext", "--out", str(tmp_path)],
        container,
    )

    out = capsys.readouterr().out
    [markdown] = tmp_path.glob("*.md")
    [data] = tmp_path.glob("*.json")
    assert code == 0
    assert "best fixed-40-0 + fulltext (Recall@5 100.0%" in out
    assert markdown.read_text().startswith("# Benchmark report — handbook")
    assert json.loads(data.read_text())["question_count"] == 1


def test_runs_and_report_commands_reuse_stored_runs(
    container: Container, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        ["benchmark", "--collection", "handbook", "--method", "vector", "--out", str(tmp_path)],
        container,
    )
    capsys.readouterr()

    main(["runs", "--collection", "handbook"], container)
    run_id = capsys.readouterr().out.split()[0]
    code = main(
        ["report", "--collection", "handbook", "--run", run_id, "--out", str(tmp_path / "again")],
        container,
    )

    assert code == 0
    assert len(list((tmp_path / "again").glob("*.md"))) == 1


def test_errors_are_reported_with_exit_code_1(
    container: Container, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["benchmark", "--collection", "missing"], container) == 1
    assert main(["benchmark", "--collection", "handbook", "--judge"], container) == 1
    assert "error:" in capsys.readouterr().err


def test_best_strategy_uses_the_requested_k(
    container: Container, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["benchmark", "--collection", "handbook", "--method", "fulltext", "--k", "1",
          "--k", "3", "--out", str(tmp_path)], container)  # fmt: skip

    assert "(Recall@3 100.0%" in capsys.readouterr().out


def test_demo_runs_the_sample_corpus_end_to_end_and_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    container = build_in_memory_container(
        settings, FakeEmbedder(), FakeReranker(), FakeChatModel("ok [1]")
    )
    examples = Path(__file__).resolve().parents[2] / "examples"
    argv = ["demo", "--corpus", str(examples / "corpus"), "--golden",
            str(examples / "golden.json"), "--out", str(tmp_path)]  # fmt: skip

    first = main(argv, container)
    out = capsys.readouterr().out
    second = main(argv, container)
    again = capsys.readouterr().out

    assert (first, second) == (0, 0)
    assert "golden dataset: 52 questions (0 skipped)" in out
    assert "ingested faq.txt" in out
    assert "kept faq.txt" in again and "golden dataset" not in again
    assert len(list(tmp_path.glob("*.md"))) == 2
    [collection] = container.ingestion.list_collections()
    assert len(container.golden.questions(collection.id)) == 52
