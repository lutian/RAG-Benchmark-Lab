# One image, two processes (API and UI) selected by the Compose `command`.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

# Local cross-encoder reranker (sentence-transformers + CPU torch, ~1 GB).
# Build with --build-arg INSTALL_LOCAL_RERANKER=false when RERANKER_PROVIDER=api.
ARG INSTALL_LOCAL_RERANKER=true
# Hugging Face model cache, writable by the non-root user (a Compose volume keeps it).
ENV HF_HOME=/app/data/hf

WORKDIR /app

# Dependencies first so they are cached independently of source changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    $([ "$INSTALL_LOCAL_RERANKER" = "true" ] && echo "--extra local-rerank")

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
COPY .streamlit ./.streamlit
# Sample corpus and golden dataset for `rag-lab demo`.
COPY examples ./examples
RUN uv sync --frozen --no-dev \
    $([ "$INSTALL_LOCAL_RERANKER" = "true" ] && echo "--extra local-rerank")

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data/hf && chown -R app /app/data
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "rag_lab.adapters.inbound.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
