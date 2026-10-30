"""FastAPI application factory."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the HTTP application. Routers are mounted here, never at import time."""
    app = FastAPI(
        title="RAG Benchmark Lab",
        version="0.1.0",
        description="Ingest documents, compare retrieval strategies and evaluate RAG answers.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
