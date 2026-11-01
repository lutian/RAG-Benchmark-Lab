"""Reranker for HTTP ``/rerank`` APIs (Jina, Cohere v2, llama.cpp server, vLLM, ...).

Request: ``POST {base_url}/rerank`` with ``{"model", "query", "documents", "top_n"}``.
Response: ``{"results": [{"index", "relevance_score"}]}`` (Jina/Cohere) or a bare list
of ``{"index", "score"}`` (Hugging Face TEI style). Results may come back sorted by
score, so they are mapped back to input order by ``index``.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from rag_lab.application.errors import RerankerError


class HttpReranker:
    def __init__(
        self, base_url: str, model: str, api_key: str = "", timeout_seconds: float = 60.0
    ) -> None:
        if not base_url.strip():
            raise ValueError("RERANKER_BASE_URL is required when RERANKER_PROVIDER=api")
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=timeout_seconds)

    @property
    def name(self) -> str:
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        body = {"model": self._model, "query": query, "documents": list(texts), "top_n": len(texts)}
        try:
            response = self._http.post(self._url, json=body)
        except httpx.HTTPError as error:
            raise RerankerError(f"reranker at {self._url} failed: {error}") from error
        if response.status_code >= 400:
            raise RerankerError(
                f"reranker at {self._url} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return _scores_in_input_order(response.json(), len(texts))


def _scores_in_input_order(payload: Any, count: int) -> list[float]:
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise RerankerError(f"unexpected reranker response: {str(payload)[:200]}")
    scores: dict[int, float] = {}
    for item in results:
        score = item.get("relevance_score", item.get("score"))
        if not isinstance(item.get("index"), int) or score is None:
            raise RerankerError(f"unexpected reranker result: {str(item)[:200]}")
        scores[item["index"]] = float(score)
    missing = [i for i in range(count) if i not in scores]
    if missing:
        raise RerankerError(f"reranker response is missing scores for texts {missing}")
    return [scores[i] for i in range(count)]
