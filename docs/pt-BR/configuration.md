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
| `LLM_REASONING_EFFORT` | não definido | Geração | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; os valores aceitos variam por modelo. Enviado como `reasoning_effort` só quando definido |
| `LLM_TEMPERATURE` | não definido | Geração | Enviado só quando definido. A inicialização falha se for combinado com um reasoning effort diferente de `none` (veja abaixo) |
| `LLM_TIMEOUT_SECONDS` | `60` | Geração | Timeout das requisições |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | OpenAI | Indexação (fase 2) | Qualquer endpoint `/embeddings` compatível com OpenAI |
| `EMBEDDING_DIMENSIONS` | `1536` | Indexação | Deve corresponder ao modelo (ex.: `768` para `nomic-embed-text`). Não é enviado ao provedor: serve para validar os vetores retornados e vincular a coleção |
| `EMBEDDING_BATCH_SIZE` | `64` | Indexação | Textos por requisição de embeddings |
| `RERANKER_PROVIDER` | `local` | Reranking | `local` (cross-encoder no processo, exige o extra `local-rerank`) ou `api` (HTTP `/rerank`) |
| `RERANKER_MODEL` / `RERANKER_BASE_URL` / `RERANKER_API_KEY` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | Id do modelo no Hugging Face para `local`, nome do modelo enviado à API para `api`; URL (ex.: `http://host:8080/v1`) e chave só para `api` |
| `RERANK_CANDIDATES` | `30` | Reranking | Resultados da híbrida entregues ao reranker; mais = mais chance de resgatar um chunk enterrado, porém mais lento |
| `INSTALL_LOCAL_RERANKER` | `true` | Build do Docker | Argumento de build do Compose; `false` gera uma imagem sem torch (use com `RERANKER_PROVIDER=api`) |
| `MAX_UPLOAD_MB` | `25` | Ingestão | Arquivos maiores recebem HTTP 413 |
| `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | `800` / `120` | Ingestão | Em caracteres; padrões para novas coleções |
| `DEFAULT_FULLTEXT_LANGUAGE` | `english` | Coleções | Configuração de busca textual do PostgreSQL para novas coleções (`portuguese`, `spanish`, `simple`…). Validada na inicialização |
| `RETRIEVAL_CANDIDATES` | `50` | Recuperação | Chunks que cada método contribui antes da fusão híbrida |
| `HYBRID_RRF_K` | `60` | Recuperação | Constante do Reciprocal Rank Fusion; valores maiores achatam a vantagem das primeiras posições |
| `API_URL` | `http://localhost:8000` | UI | `http://api:8000` dentro do Compose |

## Receitas por provedor

| Provedor | `*_BASE_URL` | `*_API_KEY` | Modelos de exemplo |
|----------|--------------|-------------|--------------------|
| OpenAI | `https://api.openai.com/v1` | sua chave | `gpt-4o-mini`, `text-embedding-3-small` (1536) |
| OpenRouter | `https://openrouter.ai/api/v1` | sua chave | `meta-llama/llama-3.1-8b-instruct` |
| Ollama (perfil do Compose) | `http://ollama:11434/v1` | `ollama` | `llama3.2`, `nomic-embed-text` (768) |
| Ollama no host | `http://host.docker.internal:11434/v1` | `ollama` | os mesmos |
| Servidor llama.cpp | `http://<host>:8080/v1` | qualquer | o modelo GGUF carregado |

### Temperatura vs reasoning effort

Os modelos de raciocínio da OpenAI rejeitam parâmetros de amostragem: *"When reasoning effort is not `none`, remove `temperature`, `top_p`, and `top_logprobs`"* ([Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model)). O GPT-6.1 Sol e o GPT-6 Astra não suportam `none`, então **nunca** aceitam `temperature`.

| Tipo de modelo | Configure | Deixe sem definir |
|----------------|-----------|-------------------|
| Modelo de raciocínio (ex.: `gpt-6.1-sol`) | `LLM_REASONING_EFFORT=low` (ou maior) | `LLM_TEMPERATURE` |
| GPT-6 Sol / Luna com raciocínio desligado | `LLM_REASONING_EFFORT=none` e, opcionalmente, `LLM_TEMPERATURE` | — |
| Modelo de chat clássico (ex.: `gpt-4o-mini`, a maioria dos modelos Ollama/llama.cpp) | `LLM_TEMPERATURE=0.0` para execuções reproduzíveis | `LLM_REASONING_EFFORT` |

**Por que não há uma lista fixa de modelos?** A regra depende do reasoning effort, não só do nome do modelo, e novos modelos saem com frequência. A aplicação envia apenas o que você configurou. Ela recusa na inicialização a única combinação que a API documenta como inválida (temperatura com raciocínio ativo), em vez de falhar na primeira requisição.

LLM e embeddings são configurados de forma **independente**: por exemplo, gerar com o OpenRouter e gerar embeddings localmente com o Ollama.

> **Vínculo de embeddings:** os vetores de uma coleção só são comparáveis com vetores do mesmo modelo e dimensão. Mudar `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` exige reindexar ou criar uma nova coleção (garantido a partir da fase 2).
