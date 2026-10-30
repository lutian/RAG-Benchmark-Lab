.PHONY: install test test-unit test-integration lint typecheck format up down logs migrate api ui

install:            ## Install all dependencies (including dev) with uv
	uv sync

test:               ## Run the whole test suite (integration tests skip without TEST_DATABASE_URL)
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:   ## Requires TEST_DATABASE_URL pointing to a disposable database
	uv run pytest -m integration

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy

migrate:
	uv run alembic upgrade head

api:
	uv run uvicorn rag_lab.adapters.inbound.api.app:create_app --factory --reload

ui:
	uv run streamlit run src/rag_lab/ui/app.py

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api ui
