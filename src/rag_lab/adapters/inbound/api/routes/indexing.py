"""Indexing: embed chunk sets and report progress. Scoped by ``collection_id``."""

from uuid import UUID

from fastapi import APIRouter

from rag_lab.adapters.inbound.api.routes.collections import ContainerDep
from rag_lab.adapters.inbound.api.schemas import IndexingReportOut, IndexStatusOut

router = APIRouter(prefix="/collections", tags=["indexing"])


@router.post("/{collection_id}/chunk-sets/{config_key}/index")
def index_chunk_set(
    collection_id: UUID, config_key: str, container: ContainerDep, rebuild: bool = False
) -> IndexingReportOut:
    """Embed every chunk of the chunk set that has no vector yet.

    Runs synchronously and is resumable: if the provider fails, call it again and
    only the remaining chunks are embedded. ``rebuild=true`` drops every vector of
    the collection and rebinds it to the currently configured embedding model.
    """
    report = container.indexing.index_chunk_set(collection_id, config_key, rebuild=rebuild)
    return IndexingReportOut.from_report(report)


@router.get("/{collection_id}/index")
def index_status(collection_id: UUID, container: ContainerDep) -> IndexStatusOut:
    return IndexStatusOut.from_status(container.indexing.status(collection_id))
