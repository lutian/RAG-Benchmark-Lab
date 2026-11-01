"""Command line: run benchmarks and export reports without the UI.

    rag-lab benchmark --collection handbook [--method hybrid ...] [--generation] --out reports/
    rag-lab runs --collection handbook
    rag-lab report --collection handbook --run <run-id> --out reports/

Uses the same composition root as the API (``.env`` settings, PostgreSQL), so a
benchmark run from the CLI is stored and visible in the UI as well. Each run writes a
Markdown report and the versioned JSON of every per-question result.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from rag_lab.application.errors import RagLabError
from rag_lab.application.reporting import best_strategy, markdown_report
from rag_lab.application.run_codec import run_to_payload
from rag_lab.bootstrap import Container, build_container
from rag_lab.config import get_settings
from rag_lab.domain.collections import Collection
from rag_lab.domain.evaluation import BenchmarkConfig, BenchmarkRun
from rag_lab.domain.retrieval import SearchMethod


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-lab", description="RAG Benchmark Lab CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser("benchmark", help="run the chunking x strategy benchmark")
    bench.add_argument("--collection", required=True, help="collection name")
    bench.add_argument("--chunk-set", dest="chunk_sets", action="append",
                       help="chunk set key (repeatable; default: all)")  # fmt: skip
    bench.add_argument("--method", dest="methods", action="append",
                       choices=[m.value for m in SearchMethod],
                       help="search method (repeatable; default: all four)")  # fmt: skip
    bench.add_argument("--k", dest="ks", action="append", type=int,
                       help="Recall@k cut-off (repeatable; default: 1, 5, 10)")  # fmt: skip
    bench.add_argument("--generation", action="store_true", help="also evaluate answers")
    bench.add_argument("--judge", action="store_true", help="LLM judge (needs --generation)")
    bench.add_argument("--answer-top-k", type=int, default=5, help="sources per answer")
    bench.add_argument("--out", type=Path, default=Path("reports"), help="report directory")

    runs = commands.add_parser("runs", help="list stored benchmark runs")
    runs.add_argument("--collection", required=True)

    report = commands.add_parser("report", help="write the report of a stored run")
    report.add_argument("--collection", required=True)
    report.add_argument("--run", required=True, type=UUID, help="run id")
    report.add_argument("--out", type=Path, default=Path("reports"))
    return parser


def _collection(container: Container, name: str) -> Collection:
    for collection in container.ingestion.list_collections():
        if collection.name == name:
            return collection
    raise RagLabError(f"collection {name!r} not found")


def _write(run: BenchmarkRun, collection: Collection, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark-{run.created_at:%Y%m%d-%H%M%S}-{str(run.id)[:8]}"
    markdown, data = out / f"{stem}.md", out / f"{stem}.json"
    markdown.write_text(markdown_report(run, collection.name), encoding="utf-8")
    payload = {"id": str(run.id), "created_at": run.created_at.isoformat(), **run_to_payload(run)}
    data.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return markdown, data


def _benchmark(args: argparse.Namespace, container: Container) -> int:
    collection = _collection(container, args.collection)
    try:
        config = BenchmarkConfig(
            chunk_sets=tuple(args.chunk_sets or [c.key for c in collection.chunking_configs]),
            methods=tuple(SearchMethod(m) for m in args.methods or [m.value for m in SearchMethod]),
            ks=tuple(sorted(set(args.ks or [1, 5, 10]))),
            generation=args.generation,
            judge=args.judge,
            answer_top_k=args.answer_top_k,
        )
    except ValueError as error:
        raise RagLabError(str(error)) from error

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total} queries", end="", file=sys.stderr, flush=True)

    run = container.benchmark.run(collection.id, config, progress)
    print(file=sys.stderr)
    markdown, data = _write(run, collection, args.out)
    best = best_strategy(run.summaries)
    print(f"run {run.id}: best {best.chunk_set} + {best.method} "
          f"(Recall@5 {best.recall.get(5, 0.0):.1%}, MRR {best.mrr:.3f})")  # fmt: skip
    print(f"report: {markdown}\ndata:   {data}")
    return 0


def _runs(args: argparse.Namespace, container: Container) -> int:
    collection = _collection(container, args.collection)
    for run in container.benchmark.runs(collection.id):
        methods = ",".join(m.value for m in run.config.methods)
        print(f"{run.id}  {run.created_at:%Y-%m-%d %H:%M}  {run.question_count} questions  "
              f"{len(run.config.chunk_sets)} chunk sets  {methods}")  # fmt: skip
    return 0


def _report(args: argparse.Namespace, container: Container) -> int:
    collection = _collection(container, args.collection)
    run = container.benchmark.get(collection.id, args.run)
    markdown, data = _write(run, collection, args.out)
    print(f"report: {markdown}\ndata:   {data}")
    return 0


def main(argv: Sequence[str] | None = None, container: Container | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"benchmark": _benchmark, "runs": _runs, "report": _report}
    try:
        return handlers[args.command](args, container or build_container(get_settings()))
    except RagLabError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
