"""Benchmarks: run the chunking x strategy matrix and read stored runs and reports."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import PlainTextResponse

from rag_lab.adapters.inbound.api.routes.collections import ContainerDep
from rag_lab.adapters.inbound.api.schemas import BenchmarkIn, BenchmarkRunOut
from rag_lab.application.errors import InvalidRequestError
from rag_lab.application.reporting import markdown_report
from rag_lab.application.run_codec import run_to_payload

router = APIRouter(prefix="/collections/{collection_id}/benchmarks", tags=["benchmarks"])


@router.post("", status_code=status.HTTP_201_CREATED)
def run_benchmark(
    collection_id: UUID, body: BenchmarkIn, container: ContainerDep
) -> BenchmarkRunOut:
    """Run every approved golden question with every chunk set x method (synchronous).

    Retrieval-only runs take seconds; with ``generation`` (and ``judge``) every
    combination calls the LLM, so expect minutes. The CLI ``rag-lab benchmark`` runs
    the same thing and writes the report files.
    """
    collection = container.ingestion.get_collection(collection_id)
    try:
        config = body.to_config([c.key for c in collection.chunking_configs])
    except ValueError as error:
        raise InvalidRequestError(str(error)) from error
    return BenchmarkRunOut.from_run(container.benchmark.run(collection_id, config))


@router.get("")
def list_runs(collection_id: UUID, container: ContainerDep) -> list[BenchmarkRunOut]:
    """Stored runs, newest first (summaries only)."""
    return [BenchmarkRunOut.from_run(run) for run in container.benchmark.runs(collection_id)]


@router.get("/{run_id}")
def get_run(collection_id: UUID, run_id: UUID, container: ContainerDep) -> dict[str, Any]:
    """The full run, per-question results included, in the versioned JSON format."""
    run = container.benchmark.get(collection_id, run_id)
    return {"id": str(run.id), "created_at": run.created_at.isoformat(), **run_to_payload(run)}


@router.get("/{run_id}/report", response_class=PlainTextResponse)
def get_report(collection_id: UUID, run_id: UUID, container: ContainerDep) -> PlainTextResponse:
    """The Markdown report of a run."""
    collection = container.ingestion.get_collection(collection_id)
    run = container.benchmark.get(collection_id, run_id)
    return PlainTextResponse(markdown_report(run, collection.name), media_type="text/markdown")
