"""HTTP client used by the UI. The UI talks to the system only through the public API."""

from typing import Any

import httpx
import streamlit as st

from rag_lab.config import get_settings

INDEXING_TIMEOUT_SECONDS = 1800.0


class ApiError(Exception):
    """The API answered with an error status; ``str(error)`` is its ``detail``."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(str(detail))
        return response.json()

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
        # Generation waits for the LLM: allow the same long timeout as indexing.
        result: dict[str, Any] = self._request(
            "POST",
            f"/collections/{collection_id}/answer",
            json=body,
            timeout=INDEXING_TIMEOUT_SECONDS,
        )
        return result


@st.cache_resource
def _cached_client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


def get_client() -> ApiClient:
    return _cached_client(get_settings().api_url)
