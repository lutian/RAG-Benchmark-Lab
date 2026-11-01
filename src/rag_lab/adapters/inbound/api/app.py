"""FastAPI application factory and error mapping."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from rag_lab.adapters.inbound.api.routes import collections, indexing, retrieval
from rag_lab.application.errors import (
    ConflictError,
    DocumentParsingError,
    EmbeddingProviderError,
    InvalidRequestError,
    NotFoundError,
    RagLabError,
    RerankerError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)
from rag_lab.bootstrap import Container, build_container
from rag_lab.config import get_settings

# Most specific first: UploadTooLargeError is also a DocumentParsingError.
_STATUS_BY_ERROR: tuple[tuple[type[RagLabError], int], ...] = (
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ConflictError, status.HTTP_409_CONFLICT),
    (UploadTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (UnsupportedDocumentError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (DocumentParsingError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (InvalidRequestError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (EmbeddingProviderError, status.HTTP_502_BAD_GATEWAY),
    (RerankerError, status.HTTP_502_BAD_GATEWAY),
)


def _status_for(error: RagLabError) -> int:
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return code
    return status.HTTP_400_BAD_REQUEST


def create_app(container: Container | None = None) -> FastAPI:
    """Build the HTTP application.

    Without an explicit container the production wiring (PostgreSQL) is used; tests
    pass an in-memory container instead.
    """
    app = FastAPI(
        title="RAG Benchmark Lab",
        version="0.1.0",
        description="Ingest documents, compare retrieval strategies and evaluate RAG answers.",
    )
    app.state.container = container or build_container(get_settings())

    @app.exception_handler(RagLabError)
    async def handle_domain_error(_: Request, error: RagLabError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(error), content={"detail": str(error)})

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(collections.router)
    app.include_router(indexing.router)
    app.include_router(retrieval.router)
    return app
