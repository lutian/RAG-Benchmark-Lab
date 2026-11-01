# Adaptadores — `rag_lab.adapters` e `rag_lab.ui`

Os adaptadores são o único código que toca arquivos, bancos de dados, HTTP e frameworks. Os adaptadores de **saída** implementam ports usados pela aplicação. Os de **entrada** recebem requisições e chamam casos de uso.

> Outros idiomas: [English](../en/adapters.md)

| Adaptador | Direção | Implementa / chama | Tecnologia |
|-----------|---------|--------------------|------------|
| Parsers | Saída | `DocumentParser` | pypdf, Markdown via regex |
| Persistência PostgreSQL | Saída | `UnitOfWork` + repositórios | SQLAlchemy Core 2, psycopg 3, Alembic |
| Persistência em memória | Saída | `UnitOfWork` + repositórios | Dicionários simples |
| Modelo de chat | Saída | `ChatModel` | `ChatOpenAI` do LangChain → qualquer API `/chat/completions` compatível com OpenAI |
| Reranking | Saída | `Reranker` | Cross-encoder do Sentence Transformers, ou `httpx` → qualquer API `/rerank` |
| Embeddings | Saída | `Embedder` | `OpenAIEmbeddings` do LangChain → qualquer API compatível com OpenAI |
| API HTTP | Entrada | `IngestionService` | FastAPI, Pydantic v2 |
| UI Streamlit | Entrada (cliente) | API HTTP | Streamlit, httpx |

---

## Parsers — `adapters/outbound/parsers/`

| Módulo | O que produz |
|--------|--------------|
| `registry.py` | `ParserRegistry`: escolhe o parser pela extensão; rejeita arquivos vazios, tipos desconhecidos e `.pdf` sem a assinatura `%PDF-` |
| `text.py` | Decodifica UTF-8 (tolera BOM) e usa Latin-1 como alternativa; parágrafos viram elementos `PARAGRAPH` |
| `markdown.py` | Títulos ATX (`#`…`######`) viram elementos `SECTION` com caminhos de títulos. Títulos dentro de blocos de código são ignorados. Front matter no estilo YAML (`chave: valor`) alimenta os metadados. Título = front matter → primeiro H1 → nome do arquivo |
| `pdf.py` | Um elemento `PAGE` por página com texto; as páginas são unidas por uma linha em branco. Páginas em branco são puladas **mas a numeração é mantida**, então as citações mostram a página real. Título e autor do PDF vão para os metadados |
| `normalize.py` | Normalização compartilhada: `\r\n`/`\r` → `\n`, remoção de NUL (o PostgreSQL não armazena esse caractere em texto) |

**Por que pypdf e não PyMuPDF?** O PyMuPDF extrai o layout melhor, mas tem licença AGPL-3.0, o que é um problema para quem quiser reutilizar este código. O pypdf é BSD, Python puro e suficiente para PDFs com camada de texto. O parser fica atrás de um port, então trocá-lo depois é uma classe. **OCR está fora do escopo:** um PDF escaneado falha com uma mensagem explícita em vez de ser indexado vazio em silêncio.

**Por que regex para Markdown em vez de um parser Markdown completo?** Só são necessários títulos, blocos de código e front matter, e os offsets precisam ser exatos sobre o texto *bruto*. Uma AST CommonMark completa traria uma dependência e um passo extra de mapeamento de volta aos offsets, sem ganho.

## Embeddings — `adapters/outbound/embeddings/`

`OpenAICompatibleEmbedder` chama `POST {EMBEDDING_BASE_URL}/embeddings` pelo `OpenAIEmbeddings` do LangChain. Funciona com OpenAI, OpenRouter, Ollama, llama.cpp, vLLM e outros.

| Parâmetro | Valor | Por quê |
|-----------|-------|---------|
| `check_embedding_ctx_length` | `False` | O padrão do LangChain tokeniza com o tiktoken e envia **ids de tokens**, que só a OpenAI aceita. Texto puro funciona com qualquer provedor |
| `dimensions` | nunca enviado | A maioria dos provedores não reduz vetores e alguns rejeitam o campo; `EMBEDDING_DIMENSIONS` valida a resposta |
| `chunk_size` | `EMBEDDING_BATCH_SIZE` | Textos por requisição HTTP |
| `max_retries` / `request_timeout` | `2` / `LLM_TIMEOUT_SECONDS` | Erros transitórios são repetidos pelo SDK da OpenAI |
| API key vazia | trocada por um placeholder | O SDK recusa chaves vazias; servidores locais ignoram o valor |

Todo `openai.OpenAIError` (erro HTTP, timeout, conexão recusada) vira `EmbeddingProviderError`, com a URL e o modelo na mensagem → HTTP 502.

**Por que LangChain aqui e não o SDK `openai` direto?** A stack do projeto inclui o LangChain, e o valor dele está na borda dos provedores: uma interface (`Embeddings`) sobre muitos provedores, que as próximas fases reutilizam para os modelos de chat. Ele fica restrito a este adaptador; o resto do código só enxerga o port `Embedder`.

## Modelo de chat — `adapters/outbound/generation/`

`OpenAICompatibleChatModel` chama `POST {LLM_BASE_URL}/chat/completions` pelo `ChatOpenAI` do LangChain.

| Parâmetro | Valor | Por quê |
|-----------|-------|---------|
| `use_responses_api` | `False` | A Responses API existe só na OpenAI; Chat Completions é o que todo servidor compatível implementa |
| `temperature` / `reasoning_effort` | só quando definidos no `.env` | Modelos de raciocínio rejeitam `temperature` (veja [configuração](configuration.md#temperatura-vs-reasoning-effort)); nada é enviado por padrão |
| `max_retries` / `timeout` | `1` / `LLM_TIMEOUT_SECONDS` | Uma única nova tentativa: uma geração que estourou o timeout ainda pode ser cobrada pelo provedor |
| API key vazia | trocada por um placeholder | O SDK recusa chaves vazias; servidores locais ignoram o valor |

Todo `openai.OpenAIError`, e também um HTTP 200 com um objeto de erro no lugar das respostas (é assim que alguns roteadores informam falhas do provedor final), vira `GenerationProviderError` com o modelo e a URL (sem o corpo da resposta) → HTTP 502. O uso de tokens vem do campo `usage` da resposta quando o provedor o envia.

## Reranking — `adapters/outbound/reranking/`

| Adaptador | `RERANKER_PROVIDER` | Como funciona |
|-----------|---------------------|---------------|
| `CrossEncoderReranker` | `local` (padrão) | `CrossEncoder` do Sentence Transformers no processo da API. O modelo (`RERANKER_MODEL`, padrão `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) é baixado e carregado na **primeira** busca com reranking, protegido por um lock para que requisições concorrentes o carreguem uma única vez |
| `HttpReranker` | `api` | `POST {RERANKER_BASE_URL}/rerank` com `{model, query, documents, top_n}` e `Authorization: Bearer RERANKER_API_KEY`. Lê `{"results": [{"index", "relevance_score"}]}` (Jina, Cohere v2, servidor do llama.cpp, vLLM) ou uma lista simples `[{"index", "score"}]`, e devolve os scores na ordem de entrada pelo `index` |

- `sentence-transformers` e torch são um **extra opcional** (`uv sync --extra local-rerank`). No Linux, o torch vem do índice CPU do PyTorch (`[tool.uv.sources]`), o que evita gigabytes de bibliotecas CUDA. Sem o extra, uma busca com reranking retorna 502 com "Install the 'local-rerank' extra or set RERANKER_PROVIDER=api".
- A primeira busca com reranking paga o carregamento do modelo (segundos). Os benchmarks de latência aquecem o reranker antes de medir.
- Toda falha (carregamento, execução, status HTTP, corpo malformado, host inacessível) vira `RerankerError` → HTTP 502.

**Por que não fazer o reranking pelo LangChain?** Os rerankers do LangChain encapsulam fornecedores específicos; um adaptador de 40 linhas com `httpx` cobre qualquer servidor `/rerank` que fale o esquema Jina/Cohere (o padrão de fato), e o adaptador local é uma camada fina sobre a biblioteca que de fato roda o modelo.

## Persistência — `adapters/outbound/persistence/`

### PostgreSQL (`postgres/`)

| Arquivo | Papel |
|---------|-------|
| `tables.py` | Definições das tabelas com SQLAlchemy Core (o schema) |
| `repositories.py` | Implementações dos repositórios; violações de unicidade viram `ConflictError` |
| `unit_of_work.py` | Uma conexão + uma transação por bloco `with` |
| `migrations/versions/0001_reference_layer.py` | Migração Alembic que cria o schema da camada de referência |
| `migrations/versions/0002_embeddings.py` | Habilita `vector`, adiciona as colunas de vínculo da coleção e `chunk_embeddings` |
| `migrations/versions/0004_evaluation.py` | Adiciona `golden_questions`, `golden_evidence` e `benchmark_runs` |
| `migrations/versions/0003_retrieval.py` | Adiciona `collections.fulltext_language` e `chunks.search_vector` (preenchido para dados existentes, índice GIN) |

```mermaid
erDiagram
    collections ||--o{ chunking_configs : "define chunk sets"
    collections ||--o{ documents : contém
    documents ||--o{ document_elements : "páginas / seções / parágrafos"
    documents ||--o{ chunks : "dividido em"
    chunking_configs ||--o{ chunks : "agrupa"
    chunks ||--o| chunk_embeddings : "vetor"
```

Destaques do schema:

- `UNIQUE (collection_id, content_hash)` em `documents` torna a deduplicação segura mesmo com uploads concorrentes.
- `chunks` tem uma FK composta `(collection_id, config_key) → chunking_configs`: um chunk não existe em um chunk set não declarado.
- `CHECK (start_offset >= 0 AND end_offset > start_offset)` replica no banco a invariante do `Span`.
- Apagar uma coleção remove em cascata tudo o que pertence a ela.
- `alembic check` confirma que a migração corresponde ao `tables.py` (sem divergência).
- `collections.embedding_model` / `embedding_dimensions` guardam o vínculo, com um CHECK garantindo que ambos sejam NULL ou ambos preenchidos.
- `chunk_embeddings(chunk_id → chunks.id ON DELETE CASCADE, collection_id, config_key, model, dimensions, embedding vector)`. A coluna `embedding` **não tem dimensão fixa**, porque as coleções podem usar modelos diferentes. `CHECK (vector_dims(embedding) = dimensions)` garante a consistência de cada linha. Refazer o chunking de um documento apaga seus chunks, e os vetores vão junto. Veja o [ADR-012](decisions.md#adr-012-uma-tabela-de-vetores-sem-dimensão-fixa).
- A migração 0002 executa `CREATE EXTENSION IF NOT EXISTS vector`, o que exige um usuário com permissão para criar extensões (o dono do banco no Compose tem).
- `collections.fulltext_language` e `chunks.search_vector` (migração 0003) sustentam a busca full-text.
- `golden_evidence.document_id → documents.id` **não tem cascade** e é `DEFERRABLE INITIALLY DEFERRED`: um documento rotulado não pode ser apagado sozinho (sua pergunta perderia a evidência em silêncio), enquanto apagar uma coleção, que se propaga para perguntas e documentos em qualquer ordem, passa na verificação feita no commit.
- `benchmark_runs.payload` é um documento JSONB por execução (`run_codec`, `format_version` 1): execuções são escritas uma vez e lidas inteiras, então normalizar as linhas por pergunta acrescentaria joins sem nenhuma consulta que precise deles.

### Busca (`PostgresSearchRepository`)

| Método | SQL | Score |
|--------|-----|-------|
| Vetorial | `embedding <=> :query` sobre os `chunk_embeddings` do chunk set, ordenado pela distância | `1 - distância de cosseno` (similaridade de cosseno) |
| Full-text | `search_vector @@ :tsquery`, ordenado por `ts_rank_cd(search_vector, tsquery, 1)` | Densidade de cobertura, dividida por `1 + log(tamanho)` |

- **Busca vetorial exata, sem índice ANN.** Cada consulta percorre os vetores de um chunk set. Assim os números do benchmark medem a *estratégia* de recuperação, e não o recall perdido por um índice aproximado. Veja o [ADR-014](decisions.md#adr-014-busca-vetorial-exata-para-o-benchmark).
- **`search_vector`** é `to_tsvector(<idioma da coleção>, text)`, calculado pelo repositório na inserção dos chunks e indexado com **GIN**. A migração 0003 o preenche para os chunks existentes.
- **Consultas com OU.** O `plainto_tsquery` une as palavras com E, o que rejeita a maioria dos chunks para uma pergunta em linguagem natural. O repositório mantém a normalização dele (stemming, stop words) e troca `&` por `|`, então um chunk é encontrado quando compartilha **qualquer** termo, e chunks com mais termos em comum ficam mais acima. Veja o [ADR-013](decisions.md#adr-013-full-text-do-postgresql-como-base-lexical).
- Empates são resolvidos pela ordem de leitura (documento, depois índice do chunk), então os resultados são determinísticos.

**Por que SQLAlchemy Core e não o ORM?** As entidades de domínio são dataclasses congeladas. O mapeamento do ORM exigiria classes mutáveis e instrumentadas ou um segundo conjunto de modelos. O Core dá SQL explícito com tabelas tipadas, e as funções de mapeamento são curtas e visíveis. **Por que não SQL em strings?** O Core compõe consultas com segurança (sem concatenação de strings) e serve de base para o Alembic. **Por que PostgreSQL?** O pgvector (fase 2) e a busca full-text nativa (fase 3) permitem recuperação vetorial, lexical e híbrida em **um único** banco, então as comparações de latência são justas.

### Em memória (`memory.py`)

O mesmo contrato sobre dicionários, com rollback real (cada transação trabalha sobre uma cópia até o `commit`). Ele sustenta os testes unitários e de API rápidos. Os **mesmos testes de contrato** rodam contra os dois adaptadores, então o store em memória não pode divergir do comportamento do PostgreSQL.

## API HTTP — `adapters/inbound/api/`

| Método | Caminho | Propósito |
|--------|---------|-----------|
| `GET` | `/health` | Liveness |
| `POST` | `/collections` | Criar coleção; sem `chunking_configs`, cria uma por estratégia com os padrões do `.env` |
| `GET` | `/collections` · `/collections/{id}` | Listar / obter |
| `POST` | `/collections/{id}/chunking-configs` | Adicionar um chunk set e dividir os documentos existentes |
| `POST` | `/collections/{id}/documents` | Enviar um arquivo (`multipart/form-data`, campo `file`) |
| `GET` | `/collections/{id}/documents` | Resumos dos documentos |
| `GET` | `/collections/{id}/documents/{doc}` | Texto canônico, estrutura, metadados |
| `GET` | `/collections/{id}/chunk-sets/{key}/chunks?document_id=` | Chunks com spans, páginas e caminhos de títulos |
| `POST` | `/collections/{id}/chunk-sets/{key}/index?rebuild=false` | Gera os embeddings dos chunks pendentes; retorna total / já indexados / gerados / segundos |
| `GET` | `/collections/{id}/index` | Vínculo de embeddings e `embedded / total` por chunk set |
| `GET` `POST` | `/collections/{id}/golden` | Listar perguntas (`?status=`) com os trechos de evidência; adicionar uma pergunta manual `{question, document_id, excerpt}` |
| `POST` | `/collections/{id}/golden/generate` | `{count, seed}`: rascunhos do LLM a partir de trechos sorteados |
| `PATCH` `DELETE` | `/collections/{id}/golden/{qid}` | Editar `{text, status}`; apagar |
| `GET` `POST` | `/collections/{id}/golden/export` · `/import` | JSON portável por hash de conteúdo; a importação informa `{imported, skipped}` |
| `POST` `GET` | `/collections/{id}/benchmarks` | Executar `{chunk_sets, methods, ks, mrr_cutoff, generation, judge, answer_top_k}` (padrão: tudo, só recuperação); listar execuções |
| `GET` | `/collections/{id}/benchmarks/{run}` · `/report` | JSON completo da execução (resultados por pergunta); relatório Markdown |
| `POST` | `/collections/{id}/answer` | Corpo `{question, chunk_set, method, top_k, document_ids}`; retorna a resposta, `abstained` + `abstention_reason`, `citations` (número → fonte), `citations_valid`, `invalid_citations`, todas as `sources`, modelo, uso de tokens e tempos (etapas de `retrieval`, `generation_ms`, `total_ms`) |
| `POST` | `/collections/{id}/search` | Corpo `{query, chunk_set, method: vector\|fulltext\|hybrid\|hybrid_rerank, top_k, document_ids}`; retorna resultados ordenados com spans, score, `ranks` por etapa e tempos |

A documentação interativa fica em `http://localhost:8000/docs` (OpenAPI).

- `app.py` contém a factory `create_app(container=None)` e o mapeamento dos erros da aplicação para códigos HTTP. Sem container, ela conecta o PostgreSQL; os testes injetam o em memória.
- `schemas.py` contém os modelos Pydantic de requisição/resposta. Eles ficam separados dos tipos de domínio, então o contrato HTTP pode evoluir sem mudar o núcleo.
- A rota de upload lê **no máximo `MAX_UPLOAD_MB` + 1 byte**, então arquivos grandes demais são rejeitados sem serem carregados inteiros na memória.

**Por que FastAPI?** Validação tipada das requisições, documentação OpenAPI automática e injeção de dependências, com pouquíssima cerimônia.

## CLI — `rag_lab/cli.py`

O `rag-lab` (instalado como script de console) usa o mesmo composition root e o mesmo `.env` da API:

```bash
rag-lab benchmark --collection handbook --out reports/                 # todos os chunk sets x 4 métodos
rag-lab benchmark --collection handbook --method hybrid --method hybrid_rerank --generation --judge
rag-lab runs --collection handbook                                     # execuções gravadas
rag-lab report --collection handbook --run <run-id> --out reports/    # reexportar uma execução
```

Cada execução escreve `benchmark-<timestamp>-<id>.md` e `.json` e também fica gravada, então aparece na UI. Erros imprimem `error: …` e saem com código 1.

**Por que um CLI além da API?** Reprodutibilidade: um benchmark vira um comando que cabe num README, num Makefile ou no CI, e os arquivos gerados podem ser versionados ao lado do código que os produziu.

## Composition root — `bootstrap.py`

`build_container(settings)` conecta o PostgreSQL. `build_in_memory_container(settings)` conecta o store em memória. Este é o **único** módulo que conhece os adaptadores concretos, então trocar uma implementação é mudar uma linha aqui.

## UI Streamlit — `rag_lab/ui/`

| Módulo | Papel |
|--------|-------|
| `app.py` | Ponto de entrada e navegação |
| `pages/home.py` | Visão geral dos dois pipelines |
| `pages/ingestion.py` | Coleções (barra lateral), uploads, lista de documentos, **inspetor de chunks**, progresso da indexação e a ação **Index chunk set** (com rebuild opcional) |
| `pages/golden.py` | **Revisão do golden dataset**: progresso até 50 perguntas aprovadas, geração pelo LLM e perguntas manuais, cada card com a evidência destacada e editar / aprovar / rejeitar / apagar, exportação e importação em JSON |
| `pages/ask.py` | **Respostas fundamentadas** em três painéis: escopo e estratégia (barra lateral), a resposta com as citações em destaque (fontes desconhecidas sinalizadas) e o motivo da abstenção, e todas as fontes, com as citadas marcadas |
| `pages/retrieval.py` | **Comparação de métodos**: uma pergunta, quatro colunas (vetorial, full-text, híbrida, híbrida + rerank; uma chave desliga o reranking), cada resultado com score, posição por etapa (inclusive a posição na híbrida antes do reranking), localização e termos da pergunta destacados; filtro por documento e `top_k` na barra lateral |
| `highlight.py` | Funções puras que desenham os spans dos chunks com destaques alternados (sobreposições ganham uma cor própria) e marcam os termos da pergunta como evidência |
| `api_client.py` | Cliente `httpx` tipado para a API |
| `theme.py` + `.streamlit/config.toml` | Identidade visual: fundo branco, destaque azul `#2563EB`, fonte Inter, cards e badges |

O inspetor de chunks mostra o mesmo documento sob diferentes chunk sets. Dá para *ver* por que o `fixed` corta frases enquanto o `structure` segue os títulos, que é a resposta visual para "por que o chunking afeta a recuperação?".

**Por que a UI chama a API em vez de importar os serviços?** Isso mantém a UI como um cliente substituível (um front-end React poderia usar a mesma API) e faz da API o único ponto de entrada testado. **Por que Streamlit?** Ele constrói dashboards cheios de dados rapidamente, na mesma linguagem do resto do projeto. O CSS customizado aproxima a UI do design de referência. Veja o [ADR-007](decisions.md#adr-007-streamlit-com-css-customizado).
