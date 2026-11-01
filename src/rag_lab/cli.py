"""Command line: run benchmarks and export reports without the UI.

    rag-lab benchmark --collection handbook [--method hybrid ...] [--generation] --out reports/
    rag-lab runs --collection handbook
    rag-lab report --collection handbook --run <run-id> --out reports/
    rag-lab demo                     # sample corpus: ingest, index, golden import, benchmark

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
from rag_lab.application.reporting import best_strategy, headline_k, markdown_report
from rag_lab.application.run_codec import run_document
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
    demo = commands.add_parser(
        "demo", help="ingest, index and benchmark the sample corpus end to end"
    )
    demo.add_argument("--collection", default="sample-handbook")
    demo.add_argument("--corpus", type=Path, default=Path("examples/corpus"))
    demo.add_argument("--golden", type=Path, default=Path("examples/golden.json"))
    demo.add_argument("--generation", action="store_true", help="also evaluate answers")
    demo.add_argument("--judge", action="store_true", help="LLM judge (needs --generation)")
    demo.add_argument("--out", type=Path, default=Path("reports"))
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
    data.write_text(json.dumps(run_document(run), indent=2, ensure_ascii=False), encoding="utf-8")
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

    interactive = sys.stderr.isatty()

    def progress(done: int, total: int) -> None:
        # A live counter on a terminal; one line per 10% in logs and pipes.
        if interactive:
            print(f"\r{done}/{total} queries", end="", file=sys.stderr, flush=True)
        elif done == total or done % max(1, total // 10) == 0:
            print(f"{done}/{total} queries", file=sys.stderr, flush=True)

    run = container.benchmark.run(collection.id, config, progress)
    if interactive:
        print(file=sys.stderr)
    markdown, data = _write(run, collection, args.out)
    k = headline_k(run.config)
    best = best_strategy(run.summaries, k)
    print(f"run {run.id}: best {best.chunk_set} + {best.method} "
          f"(Recall@{k} {best.recall.get(k, 0.0):.1%}, MRR {best.mrr:.3f})")  # fmt: skip
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


def _demo(args: argparse.Namespace, container: Container) -> int:
    """Idempotent: re-running reuses the collection, skips duplicate files and indexed
    chunks, imports the golden dataset only once, then runs a new benchmark."""
    files = sorted(p for p in args.corpus.iterdir() if p.is_file()) if args.corpus.is_dir() else []
    if not files:
        raise RagLabError(f"no documents found in {args.corpus}")
    existing = [c for c in container.ingestion.list_collections() if c.name == args.collection]
    collection = (
        existing[0]
        if existing
        else container.ingestion.create_collection(
            args.collection, container.default_chunking_configs
        )
    )
    print(f"collection {collection.name}: {len(collection.chunking_configs)} chunk sets")
    for path in files:
        result = container.ingestion.ingest_document(collection.id, path.name, path.read_bytes())
        print(f"  {'kept' if result.duplicate else 'ingested'} {path.name}")
    for config in collection.chunking_configs:
        report = container.indexing.index_chunk_set(collection.id, config.key)
        print(f"  indexed {config.key}: {report.embedded} new of {report.total} chunks")
    if not container.golden.questions(collection.id):
        imported = container.golden.import_(
            collection.id, json.loads(args.golden.read_text(encoding="utf-8"))
        )
        print(f"  golden dataset: {imported.imported} questions ({imported.skipped} skipped)")
    benchmark = argparse.Namespace(
        collection=collection.name, chunk_sets=None, methods=None, ks=None,
        generation=args.generation, judge=args.judge, answer_top_k=5, out=args.out,
    )  # fmt: skip
    return _benchmark(benchmark, container)


def main(argv: Sequence[str] | None = None, container: Container | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"benchmark": _benchmark, "runs": _runs, "report": _report, "demo": _demo}
    try:
        return handlers[args.command](args, container or build_container(get_settings()))
    except RagLabError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
