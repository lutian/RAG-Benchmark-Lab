.PHONY: install test test-unit test-integration lint typecheck format up down logs migrate api ui demo demo-docker compose-smoke golden

install:            ## Install all dependencies (including dev) with uv
	uv sync

test:               ## Run the whole test suite (integration tests skip without TEST_DATABASE_URL)
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:   ## Requires TEST_DATABASE_URL pointing to a disposable database
	uv run pytest -m integration

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

typecheck:
	uv run mypy

migrate:
	uv run alembic upgrade head

api:
	uv run uvicorn rag_lab.adapters.inbound.api.app:create_app --factory --reload

ui:
	uv run streamlit run src/rag_lab/ui/app.py

demo:               ## Sample corpus end to end: ingest, index, import golden, benchmark -> reports/
	uv run rag-lab demo

demo-docker:        ## Same, inside the running Compose stack
	docker compose exec api rag-lab demo --out /app/data/reports
	mkdir -p reports && docker compose cp api:/app/data/reports/. ./reports/

compose-smoke:      ## Build and start db/api/ui, then smoke-test the stack (no provider key needed)
	docker compose up --detach --build --wait --wait-timeout 600 db api ui
	python3 scripts/compose_smoke.py

golden:             ## Rebuild examples/golden.json after editing the sample corpus or questions
	uv run python examples/build_golden.py

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api ui
