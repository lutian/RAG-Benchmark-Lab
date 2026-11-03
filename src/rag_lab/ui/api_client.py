"""HTTP client used by the UI. The UI talks to the system only through the public API."""

from typing import Any

import httpx
import streamlit as st

from rag_lab.config import get_settings

INDEXING_TIMEOUT_SECONDS = 1800.0
# A full matrix with reranking and generation can take tens of minutes; for longer
# runs use the CLI (`rag-lab benchmark`), which has no HTTP timeout at all.
BENCHMARK_TIMEOUT_SECONDS = 3600.0
# Margin for retrieval and reranking on top of the LLM attempts.
GENERATION_MARGIN_SECONDS = 120.0


class ApiError(Exception):
    """The API answered with an error status; ``str(error)`` is its ``detail``."""


class ApiClient:
    def __init__(
        self, base_url: str, timeout: float = 120.0, generation_timeout: float = 600.0
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._generation_timeout = generation_timeout

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send a request; any error status becomes ApiError, JSON body or not (a proxy's
        HTML 502 page included)."""
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text[:300]
            raise ApiError(str(detail))
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._send(method, path, **kwargs).json()

    def list_collections(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request("GET", "/collections")
        return result

    def create_collection(
        self,
        name: str,
        chunking_configs: list[dict[str, Any]] | None = None,
        fulltext_language: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if chunking_configs is not None:
            body["chunking_configs"] = chunking_configs
        if fulltext_language is not None:
            body["fulltext_language"] = fulltext_language
        result: dict[str, Any] = self._request("POST", "/collections", json=body)
        return result

    def rename_collection(self, collection_id: str, name: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "PATCH", f"/collections/{collection_id}", json={"name": name}
        )
        return result

    def delete_collection(self, collection_id: str) -> None:
        self._send("DELETE", f"/collections/{collection_id}")

    def add_chunking_config(self, collection_id: str, config: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", f"/collections/{collection_id}/chunking-configs", json=config
        )
        return result

    def upload(self, collection_id: str, filename: str, data: bytes) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", f"/collections/{collection_id}/documents", files={"file": (filename, data)}
        )
        return result

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "GET", f"/collections/{collection_id}/documents"
        )
        return result

    def get_document(self, collection_id: str, document_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "GET", f"/collections/{collection_id}/documents/{document_id}"
        )
        return result

    def list_chunks(
        self, collection_id: str, config_key: str, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        params = {"document_id": document_id} if document_id else None
        result: list[dict[str, Any]] = self._request(
            "GET", f"/collections/{collection_id}/chunk-sets/{config_key}/chunks", params=params
        )
        return result

    def index_chunk_set(
        self, collection_id: str, config_key: str, rebuild: bool = False
    ) -> dict[str, Any]:
        # Indexing calls the embeddings provider for every chunk: allow a long timeout.
        result: dict[str, Any] = self._request(
            "POST",
            f"/collections/{collection_id}/chunk-sets/{config_key}/index",
            params={"rebuild": rebuild},
            timeout=INDEXING_TIMEOUT_SECONDS,
        )
        return result

    def index_status(self, collection_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", f"/collections/{collection_id}/index")
        return result

    def search(  # method: vector | fulltext | hybrid | hybrid_rerank
        self,
        collection_id: str,
        query: str,
        chunk_set: str,
        method: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "chunk_set": chunk_set,
            "method": method,
            "top_k": top_k,
            "document_ids": document_ids,
        }
        result: dict[str, Any] = self._request(
            "POST", f"/collections/{collection_id}/search", json=body
        )
        return result

    def answer(
        self,
        collection_id: str,
        question: str,
        chunk_set: str,
        method: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "question": question,
            "chunk_set": chunk_set,
            "method": method,
            "top_k": top_k,
            "document_ids": document_ids,
        }
        result: dict[str, Any] = self._request(
            "POST",
            f"/collections/{collection_id}/answer",
            json=body,
            timeout=self._generation_timeout,
        )
        return result

    # --- golden dataset -----------------------------------------------------------
    def golden_questions(
        self, collection_id: str, status: str | None = None
    ) -> list[dict[str, Any]]:
        params = {"status": status} if status else None
        result: list[dict[str, Any]] = self._request(
            "GET", f"/collections/{collection_id}/golden", params=params
        )
        return result

    def golden_add(
        self, collection_id: str, question: str, document_id: str, excerpt: str
    ) -> dict[str, Any]:
        body = {"question": question, "document_id": document_id, "excerpt": excerpt}
        result: dict[str, Any] = self._request(
            "POST", f"/collections/{collection_id}/golden", json=body
        )
        return result

    def golden_generate(self, collection_id: str, count: int, seed: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "POST",
            f"/collections/{collection_id}/golden/generate",
            json={"count": count, "seed": seed},
            timeout=self._generation_timeout * max(1, count),
        )
        return result

    def golden_update(
        self, collection_id: str, question_id: str, text: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        body = {key: value for key, value in {"text": text, "status": status}.items() if value}
        result: dict[str, Any] = self._request(
            "PATCH", f"/collections/{collection_id}/golden/{question_id}", json=body
        )
        return result

    def golden_delete(self, collection_id: str, question_id: str) -> None:
        self._send("DELETE", f"/collections/{collection_id}/golden/{question_id}")

    def golden_export(self, collection_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", f"/collections/{collection_id}/golden/export")
        return result

    def golden_import(self, collection_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", f"/collections/{collection_id}/golden/import", json=payload
        )
        return result

    # --- benchmarks ---------------------------------------------------------------
    def benchmark_runs(self, collection_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "GET", f"/collections/{collection_id}/benchmarks"
        )
        return result

    def benchmark_run(self, collection_id: str, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "GET", f"/collections/{collection_id}/benchmarks/{run_id}"
        )
        return result

    def benchmark_report(self, collection_id: str, run_id: str) -> str:
        return self._send("GET", f"/collections/{collection_id}/benchmarks/{run_id}/report").text

    def run_benchmark(self, collection_id: str, body: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST",
            f"/collections/{collection_id}/benchmarks",
            json=body,
            timeout=BENCHMARK_TIMEOUT_SECONDS,
        )
        return result


@st.cache_resource
def _cached_client(base_url: str, generation_timeout: float) -> ApiClient:
    return ApiClient(base_url, generation_timeout=generation_timeout)


def get_client() -> ApiClient:
    settings = get_settings()
    # The API tries the LLM twice (one retry), each up to LLM_TIMEOUT_SECONDS.
    generation_timeout = 2 * settings.llm_timeout_seconds + GENERATION_MARGIN_SECONDS
    return _cached_client(settings.api_url, generation_timeout)
