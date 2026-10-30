# Configuração — `.env`

Todos os modelos, endpoints, API keys e parâmetros ficam no `.env`. Nada fica fixo no código, então trocar a OpenAI por OpenRouter, Ollama ou llama.cpp não exige mudar uma linha de código.

> Outros idiomas: [English](../en/configuration.md)

## Caminho rápido

```bash
cp .env.example .env
# edite LLM_* e EMBEDDING_* para o seu provedor e depois:
docker compose up --build
```

## Como é carregado

`rag_lab/config.py` define a classe tipada `Settings` (pydantic-settings):

- cada campo corresponde a uma variável em maiúsculas (`llm_model` → `LLM_MODEL`);
- variáveis de ambiente reais **sobrescrevem** o `.env`;
- as API keys são `SecretStr`: nunca aparecem em `repr()`, logs ou `model_dump()` (um teste verifica isso);
- valores inválidos falham logo na inicialização (por exemplo, `RERANKER_PROVIDER=magic`).

**Por que pydantic-settings e não `os.getenv`?** A configuração fica tipada, validada e documentada em um só lugar, e os segredos vêm mascarados por padrão.

## Variáveis

| Variável | Padrão | Usada por | Observações |
|----------|--------|-----------|-------------|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `rag` / — / `rag_lab` | Serviço `db` do Compose | A senha é **obrigatória**; o Compose não sobe sem ela |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag_lab` | API, Alembic | Host `db` dentro do Compose, `localhost` a partir da sua máquina |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Geração (fase 5) | Qualquer endpoint `/chat/completions` compatível com OpenAI |
| `LLM_API_KEY` | vazio | Geração | Para o Ollama, qualquer valor não vazio |
| `LLM_MODEL` | `gpt-4o-mini` | Geração | Id do modelo no provedor |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT_SECONDS` | `0.0` / `60` | Geração | `0.0` para experimentos reproduzíveis |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | OpenAI | Indexação (fase 2) | Qualquer endpoint `/embeddings` compatível com OpenAI |
| `EMBEDDING_DIMENSIONS` | `1536` | Indexação | Deve corresponder ao modelo (ex.: `768` para `nomic-embed-text`) |
| `EMBEDDING_BATCH_SIZE` | `64` | Indexação | Textos por requisição de embeddings |
| `RERANKER_PROVIDER` | `local` | Reranking (fase 4) | `local` (cross-encoder no processo) ou `api` (HTTP `/rerank`) |
| `RERANKER_MODEL` / `RERANKER_BASE_URL` / `RERANKER_API_KEY` | Cross-encoder MiniLM | Reranking | URL e chave só para `api` |
| `MAX_UPLOAD_MB` | `25` | Ingestão | Arquivos maiores recebem HTTP 413 |
| `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | `800` / `120` | Ingestão | Em caracteres; padrões para novas coleções |
| `API_URL` | `http://localhost:8000` | UI | `http://api:8000` dentro do Compose |

## Receitas por provedor

| Provedor | `*_BASE_URL` | `*_API_KEY` | Modelos de exemplo |
|----------|--------------|-------------|--------------------|
| OpenAI | `https://api.openai.com/v1` | sua chave | `gpt-4o-mini`, `text-embedding-3-small` (1536) |
| OpenRouter | `https://openrouter.ai/api/v1` | sua chave | `meta-llama/llama-3.1-8b-instruct` |
| Ollama (perfil do Compose) | `http://ollama:11434/v1` | `ollama` | `llama3.2`, `nomic-embed-text` (768) |
| Ollama no host | `http://host.docker.internal:11434/v1` | `ollama` | os mesmos |
| Servidor llama.cpp | `http://<host>:8080/v1` | qualquer | o modelo GGUF carregado |

LLM e embeddings são configurados de forma **independente**: por exemplo, gerar com o OpenRouter e gerar embeddings localmente com o Ollama.

> **Vínculo de embeddings:** os vetores de uma coleção só são comparáveis com vetores do mesmo modelo e dimensão. Mudar `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` exige reindexar ou criar uma nova coleção (garantido a partir da fase 2).
