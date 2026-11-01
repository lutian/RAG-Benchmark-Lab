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

## Configuração do reranker

A quarta estratégia, `hybrid_rerank`, reordena os `RERANK_CANDIDATES` melhores resultados da busca híbrida com um cross-encoder: um modelo que lê a pergunta e cada chunk **juntos** e dá uma nota para o quanto o chunk responde à pergunta. Isso é mais preciso do que comparar dois embeddings separados, mas lento demais para rodar sobre a coleção inteira, e por isso ele só reordena uma lista curta.

### Opção A: cross-encoder local (padrão)

```dotenv
RERANKER_PROVIDER=local
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_BASE_URL=
RERANKER_API_KEY=
```

`RERANKER_BASE_URL` e `RERANKER_API_KEY` são ignorados com `local`; deixe-os vazios.

1. **Instale o runtime.** No Docker, não há nada a fazer: a imagem já instala (`INSTALL_LOCAL_RERANKER=true` é o padrão). No desenvolvimento local: `uv sync --extra local-rerank` (sentence-transformers + torch para CPU, cerca de 1 GB).
2. **Obtenha o modelo.** Também não há nada a fazer: o modelo é baixado do Hugging Face na **primeira** busca com reranking e fica em cache. No Docker, o cache é o volume `models` (`HF_HOME=/app/data/hf`), então ele sobrevive a reinícios; localmente, fica em `~/.cache/huggingface` (ou `$HF_HOME`). Para baixá-lo antes (por exemplo, antes de uma demo ou numa máquina que vai ficar offline):

   ```bash
   docker compose exec api python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   # localmente: uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

3. **Confira.** Faça uma busca com `"method": "hybrid_rerank"` (ou use a página Retrieval). A primeira chamada leva alguns segundos (carregamento); as seguintes são rápidas. Cada resultado mostra sua posição na busca híbrida antes do reranking.

Escolhendo o `RERANKER_MODEL` (qualquer id de cross-encoder do Sentence Transformers no Hugging Face funciona):

| Modelo | Download | Idiomas | Quando usar |
|--------|----------|---------|-------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (padrão) | ~90 MB | Inglês | Corpus em inglês; rápido em CPU. Usado no relatório de exemplo |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130 MB | Inglês | Um pouco mais preciso, cerca de duas vezes mais lento |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~470 MB | Multilíngue (inclui português) | Corpus em outros idiomas; ainda leve para CPU |
| `BAAI/bge-reranker-base` | ~1,1 GB | Inglês, chinês | Qualidade maior, mais lento |
| `BAAI/bge-reranker-v2-m3` | ~2,3 GB | Multilíngue | A melhor qualidade desta lista; lento em CPU, prefira GPU ou a opção via API |

Escolha o reranker de acordo com o idioma do corpus: um modelo só de inglês ainda ordena texto em português, mas separa muito pior os chunks relevantes dos irrelevantes. Depois de trocar o modelo, recrie a API para ela ler o novo `.env` (`docker compose up -d api`; `docker compose restart` **não** relê o `.env`). O relatório do benchmark registra o reranker usado (`local: <modelo>`), então execuções com modelos diferentes continuam comparáveis.

### Opção B: uma API `/rerank`

```dotenv
RERANKER_PROVIDER=api
RERANKER_MODEL=<nome do modelo enviado à API>
RERANKER_BASE_URL=<URL base; a aplicação chama {RERANKER_BASE_URL}/rerank>
RERANKER_API_KEY=<chave, enviada como "Authorization: Bearer ..."; vazia se não houver>
```

A aplicação envia `{model, query, documents, top_n}` e lê `{"results": [{"index", "relevance_score"}]}`, o formato que estes provedores compartilham:

| Provedor | `RERANKER_BASE_URL` | Exemplo de `RERANKER_MODEL` | `RERANKER_API_KEY` |
|----------|---------------------|-----------------------------|--------------------|
| Jina AI | `https://api.jina.ai/v1` | `jina-reranker-v2-base-multilingual` | sua chave |
| Cohere | `https://api.cohere.com/v2` | `rerank-v3.5` | sua chave |
| vLLM (`vllm serve BAAI/bge-reranker-v2-m3`) | `http://<host>:8000/v1` | `BAAI/bge-reranker-v2-m3` | vazia, ou a chave do servidor |
| llama.cpp (`llama-server --reranking -m <reranker>.gguf`) | `http://<host>:8080/v1` | qualquer um (responde o modelo carregado) | vazia |

O Text Embeddings Inference (TEI) do Hugging Face **não** é compatível: o `/rerank` dele espera `texts` em vez de `documents`.

Com a opção via API você não precisa de torch na imagem: monte com `INSTALL_LOCAL_RERANKER=false docker compose up --build` para ter uma imagem bem menor. De dentro de um container, acesse um servidor na sua máquina com `http://host.docker.internal:<porta>/v1` (Docker Desktop; no Linux, adicione `extra_hosts: ["host.docker.internal:host-gateway"]` ao serviço `api`).

### Quando falha

Uma busca com reranking que não consegue reordenar devolve **HTTP 502** com o motivo (os outros três métodos continuam funcionando):

| Mensagem | Correção |
|----------|----------|
| `cannot load cross-encoder ... Install the 'local-rerank' extra or set RERANKER_PROVIDER=api` | Rode `uv sync --extra local-rerank`, ou remonte a imagem com `INSTALL_LOCAL_RERANKER=true`; confira também o id do modelo e se o Hugging Face está acessível no primeiro uso |
| `RERANKER_BASE_URL is required when RERANKER_PROVIDER=api` | Defina a URL base |
| `reranker at ... returned HTTP 401` (ou 4xx/5xx) | Confira a chave, o nome do modelo e a URL (ela não deve terminar em `/rerank`) |
