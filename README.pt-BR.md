# RAG Benchmark Lab

**Um laboratório que mostra *por que* uma estratégia de recuperação supera outra, com experimentos reproduzíveis sobre os seus próprios documentos.**

Não é um chatbot que lê PDFs. Você envia qualquer corpus. O laboratório o divide em chunks de várias formas e recupera com busca lexical, vetorial, híbrida e com reranking. Depois mede qualidade, latência e custo contra um golden dataset, para você diagnosticar se uma resposta ruim vem da **recuperação** ou da **geração**.

> 🇺🇸 [Read in English](README.md)

## Status

| Fase | Entregável | Status |
|------|------------|--------|
| 1. Ingestão | Upload, parsing (PDF/MD/TXT), metadados, chunking configurável, camada de referência independente do chunking | ✅ Concluída |
| 2. Indexação | Embeddings de qualquer API compatível com OpenAI, persistidos no pgvector; retomável, coleções vinculadas ao modelo | ✅ Concluída |
| 3. Recuperação | Busca vetorial, full-text e híbrida | ⏳ Próxima |
| 4. Reranking | Cross-encoder local e API `/rerank` | Planejada |
| 5. Geração | Respostas fundamentadas com citações e abstenção | Planejada |
| 6. Avaliação | Golden dataset, Recall@k, MRR, validade das citações | Planejada |
| 7. Apresentação | Dashboard comparativo | Planejada |

## Início rápido (Docker)

Requisitos: Docker com Compose v2.

```bash
git clone https://github.com/lutian/RAG-Benchmark-Lab.git
cd RAG-Benchmark-Lab
cp .env.example .env            # defina POSTGRES_PASSWORD e as chaves do seu provedor
docker compose up --build -d
```

| Serviço | URL |
|---------|-----|
| UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 · documentação interativa em http://localhost:8000/docs |
| PostgreSQL + pgvector | `localhost:5432` (apenas loopback) |

Abra a UI, crie uma coleção na barra lateral, envie arquivos PDF, Markdown ou TXT e inspecione os limites dos chunks de cada chunk set. Clique em **Index chunk set** para gerar os embeddings com o provedor configurado no `.env`.

Modelos locais opcionais com Ollama:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
# no .env: LLM_BASE_URL=http://ollama:11434/v1, EMBEDDING_BASE_URL=http://ollama:11434/v1,
#          EMBEDDING_MODEL=nomic-embed-text, EMBEDDING_DIMENSIONS=768
```

## Teste com curl

```bash
# 1. Criar uma coleção (por padrão, um chunk set por estratégia)
curl -s -X POST localhost:8000/collections -H 'content-type: application/json' \
     -d '{"name": "meus-docs"}'

# 2. Enviar um documento
curl -s -X POST localhost:8000/collections/<collection_id>/documents -F "file=@relatorio.pdf"

# 3. Comparar como duas estratégias cortam o mesmo documento
curl -s "localhost:8000/collections/<collection_id>/chunk-sets/fixed-800-120/chunks?document_id=<id>"
curl -s "localhost:8000/collections/<collection_id>/chunk-sets/structure-800-120/chunks?document_id=<id>"

# 4. Gerar os embeddings de um chunk set (usa EMBEDDING_* do .env) e ver o progresso
curl -s -X POST localhost:8000/collections/<collection_id>/chunk-sets/structure-800-120/index
curl -s localhost:8000/collections/<collection_id>/index
```

## Desenvolvimento local (sem Docker para a aplicação)

| Ferramenta | Versão | Instalação |
|------------|--------|------------|
| Python | ≥ 3.12 | https://www.python.org/downloads/ |
| uv | ≥ 0.13 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (ou `pipx install uv`) |
| PostgreSQL + pgvector | 17 | `docker compose up -d db` (o mais simples) |

```bash
uv sync                          # cria o .venv com dependências da app e de dev (a partir do uv.lock)
cp .env.example .env             # use localhost como host no DATABASE_URL
docker compose up -d db          # apenas o banco
make migrate                     # alembic upgrade head
make api                         # http://localhost:8000 (auto-reload)
make ui                          # http://localhost:8501 (defina API_URL=http://localhost:8000)
```

### Bibliotecas e por que estão aqui

| Biblioteca | Papel |
|------------|-------|
| FastAPI, Uvicorn, python-multipart | API HTTP, servidor ASGI, upload de arquivos |
| Pydantic, pydantic-settings | Validação de requisições; configuração tipada do `.env` |
| SQLAlchemy 2 (Core), psycopg 3, Alembic | Acesso ao PostgreSQL e migrações versionadas |
| pypdf | Extração de texto de PDF (licença BSD) |
| langchain-openai (+ SDK openai) | Cliente para qualquer API de embeddings compatível com OpenAI |
| pgvector (Python) + extensão `vector` | Tipo de coluna vetorial para SQLAlchemy e PostgreSQL |
| Streamlit, httpx | UI e seu cliente HTTP |
| pytest, Hypothesis, fpdf2 | Testes, testes baseados em propriedades, fixtures de PDF |
| ruff, mypy | Lint/formatação e checagem estrita de tipos |

As próximas fases adicionam o Sentence Transformers (reranker local). Cada fase documenta aqui suas novas dependências.

### Comandos comuns

| Comando | O que faz |
|---------|-----------|
| `make test` | Suíte completa (os testes de contrato com PostgreSQL são pulados sem `TEST_DATABASE_URL`) |
| `make test-integration` | Testes de contrato contra um PostgreSQL descartável |
| `make lint` / `make format` | Ruff: checar / corrigir |
| `make typecheck` | mypy `--strict` |
| `make up` / `make down` / `make logs` | Ciclo de vida do Docker Compose |

## Configuração

Tudo (modelos, endpoints, API keys, limites) fica no `.env`. Qualquer provedor **compatível com OpenAI** funciona para o LLM e para os embeddings, configurados de forma independente: OpenAI, OpenRouter, Ollama, llama.cpp, vLLM… Veja [docs/pt-BR/configuration.md](docs/pt-BR/configuration.md).

## Arquitetura em uma imagem

```text
 Streamlit UI ──HTTP──▶ FastAPI ──▶ Aplicação (casos de uso, ports) ──▶ Domínio (camada de referência, chunking)
                                          │
                                          ├─▶ Parsers (PDF / Markdown / TXT)
                                          └─▶ PostgreSQL (SQLAlchemy Core)  |  Store em memória (testes)
```

Arquitetura hexagonal: o domínio não importa frameworks, e toda dependência externa fica atrás de um port. A ideia central é a **camada de referência**. Cada documento é processado uma única vez em um texto canônico, e chunks e rótulos do golden dataset são intervalos de caracteres sobre ele. O chunking pode mudar livremente sem invalidar o dataset de avaliação.

## Documentação

| Tema | Português | English |
|------|-----------|---------|
| Visão geral da arquitetura | [arquitetura](docs/pt-BR/architecture.md) | [architecture](docs/en/architecture.md) |
| Camada de domínio | [domínio](docs/pt-BR/domain.md) | [domain](docs/en/domain.md) |
| Camada de aplicação | [aplicação](docs/pt-BR/application.md) | [application](docs/en/application.md) |
| Adaptadores (parsers, banco, API, UI) | [adaptadores](docs/pt-BR/adapters.md) | [adapters](docs/en/adapters.md) |
| Configuração | [configuração](docs/pt-BR/configuration.md) | [configuration](docs/en/configuration.md) |
| Testes | [testes](docs/pt-BR/testing.md) | [testing](docs/en/testing.md) |
| Decisões (ADRs) | [decisões](docs/pt-BR/decisions.md) | [decisions](docs/en/decisions.md) |

## Metas de validação (propostas, não resultados)

| Indicador | Meta |
|-----------|------|
| Perguntas de avaliação | 50 |
| Estratégias comparadas | 4 (vetorial, full-text, híbrida, híbrida + rerank) |
| Recall@5 | ≥ 85% |
| MRR@10 | ≥ 0,75 |
| Respostas com citações válidas | ≥ 95% |
| Falhas críticas de isolamento | 0 |

As metas podem ser ajustadas conforme a dificuldade do corpus e a qualidade dos rótulos de referência.

## Estrutura do projeto

```text
src/rag_lab/
  domain/            spans, documentos, chunks, coleções, estratégias de chunking
  application/       casos de uso (services/), ports/, erros
  adapters/
    inbound/api/     app FastAPI, rotas, schemas
    outbound/        parsers/, persistence/ (memória, postgres)
  ui/                páginas Streamlit, tema, cliente da API
  bootstrap.py       composition root
  config.py          configurações do .env
migrations/          Alembic
tests/               unit/, contracts/ (memória + PostgreSQL), utilitários de integração
docs/                en/ e pt-BR/
```
