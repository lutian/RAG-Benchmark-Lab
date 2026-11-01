"""Local cross-encoder reranker (Sentence Transformers), running in the API process.

The model is loaded on first use, not at startup: the API starts fast, and a lab that
never reranks never downloads it. ``sentence-transformers`` (and torch) is an optional
dependency: ``uv sync --extra local-rerank`` (installed by default in the Docker image).
"""

import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from rag_lab.application.errors import RerankerError


class _Model(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


def _load_sentence_transformers(name: str) -> _Model:
    from sentence_transformers import CrossEncoder  # optional dependency

    model: _Model = CrossEncoder(name)
    return model


class CrossEncoderReranker:
    def __init__(
        self,
        model: str,
        loader: Callable[[str], _Model] = _load_sentence_transformers,
        batch_size: int = 32,
    ) -> None:
        self._name = model
        self._loader = loader
        self._batch_size = batch_size
        self._model: _Model | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        try:
            scores = model.predict([(query, text) for text in texts], batch_size=self._batch_size)
        except Exception as error:  # model runtime failures (OOM, bad input)
            raise RerankerError(f"cross-encoder {self._name!r} failed: {error}") from error
        return [float(score) for score in scores]

    def _load(self) -> _Model:
        # FastAPI runs sync routes in a thread pool: load the model only once.
        with self._lock:
            if self._model is None:
                try:
                    self._model = self._loader(self._name)
                except Exception as error:
                    raise RerankerError(
                        f"cannot load cross-encoder {self._name!r}: {error}. Install the "
                        "'local-rerank' extra or set RERANKER_PROVIDER=api"
                    ) from error
            return self._model
