"""Collections, documents and chunk sets. Every path is scoped by ``collection_id``."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, UploadFile, status

from rag_lab.adapters.inbound.api.schemas import (
    ChunkingConfigIn,
    ChunkOut,
    CollectionIn,
    CollectionOut,
    CollectionRenameIn,
    DocumentDetailOut,
    DocumentSummaryOut,
    IngestionOut,
)
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.bootstrap import Container

router = APIRouter(prefix="/collections", tags=["collections"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def _service(container: Container) -> IngestionService:
    return container.ingestion


@router.post("", status_code=status.HTTP_201_CREATED)
def create_collection(body: CollectionIn, container: ContainerDep) -> CollectionOut:
    configs = (
        tuple(c.to_domain() for c in body.chunking_configs)
        if body.chunking_configs is not None
        else container.default_chunking_configs
    )
    language = body.fulltext_language or container.settings.default_fulltext_language
    return CollectionOut.from_domain(
        _service(container).create_collection(body.name, configs, language)
    )


@router.get("")
def list_collections(container: ContainerDep) -> list[CollectionOut]:
    return [CollectionOut.from_domain(c) for c in _service(container).list_collections()]


@router.get("/{collection_id}")
def get_collection(collection_id: UUID, container: ContainerDep) -> CollectionOut:
    return CollectionOut.from_domain(_service(container).get_collection(collection_id))


@router.patch("/{collection_id}")
def rename_collection(
    collection_id: UUID, body: CollectionRenameIn, container: ContainerDep
) -> CollectionOut:
    return CollectionOut.from_domain(
        _service(container).rename_collection(collection_id, body.name)
    )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: UUID, container: ContainerDep) -> Response:
    """Delete the collection and everything in it. Irreversible."""
    _service(container).delete_collection(collection_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{collection_id}/chunking-configs", status_code=status.HTTP_201_CREATED)
def add_chunking_config(
    collection_id: UUID, body: ChunkingConfigIn, container: ContainerDep
) -> CollectionOut:
    updated = _service(container).add_chunking_config(collection_id, body.to_domain())
    return CollectionOut.from_domain(updated)


@router.post("/{collection_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: UUID, file: UploadFile, container: ContainerDep
) -> IngestionOut:
    # Read at most one byte past the limit: enough to detect oversize files without
    # buffering arbitrarily large uploads in memory.
    limit = container.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    result = _service(container).ingest_document(collection_id, file.filename or "upload", data)
    return IngestionOut.from_result(result)


@router.get("/{collection_id}/documents")
def list_documents(collection_id: UUID, container: ContainerDep) -> list[DocumentSummaryOut]:
    return [
        DocumentSummaryOut.from_domain(d) for d in _service(container).list_documents(collection_id)
    ]


@router.get("/{collection_id}/documents/{document_id}")
def get_document(
    collection_id: UUID, document_id: UUID, container: ContainerDep
) -> DocumentDetailOut:
    document = _service(container).get_document(collection_id, document_id)
    return DocumentDetailOut.from_document(document)


@router.get("/{collection_id}/chunk-sets/{config_key}/chunks")
def list_chunks(
    collection_id: UUID,
    config_key: str,
    container: ContainerDep,
    document_id: UUID | None = None,
) -> list[ChunkOut]:
    chunks = _service(container).list_chunks(collection_id, config_key, document_id)
    return [ChunkOut.from_domain(c) for c in chunks]
