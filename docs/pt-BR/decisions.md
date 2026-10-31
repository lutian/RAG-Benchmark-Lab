# Registros de decisões de arquitetura (ADRs)

Cada registro apresenta primeiro a decisão e depois as alternativas descartadas e o motivo. Status: **Aceita** (implementada), **Planejada** (acordada, implementada em uma fase futura).

> Outros idiomas: [English](../en/decisions.md)

| ADR | Decisão | Status |
|-----|---------|--------|
| [001](#adr-001-arquitetura-hexagonal) | Arquitetura hexagonal | Aceita |
| [002](#adr-002-camada-de-referência-independente-do-chunking) | Camada de referência independente do chunking (spans de caracteres) | Aceita |
| [003](#adr-003-tamanho-do-chunk-em-caracteres) | Tamanho do chunk medido em caracteres | Aceita |
| [004](#adr-004-chunkers-próprios-langchain-na-borda-dos-provedores) | Chunkers próprios; LangChain só na borda dos provedores | Aceita |
| [005](#adr-005-postgresql-para-tudo) | PostgreSQL + pgvector para documentos, vetores e full-text | Aceita |
| [006](#adr-006-provedores-compatíveis-com-openai-configurados-no-env) | Provedores compatíveis com OpenAI configurados no `.env` | Aceita (embeddings prontos, chat planejado) |
| [007](#adr-007-streamlit-com-css-customizado) | Streamlit com CSS customizado | Aceita |
| [008](#adr-008-coleções-como-fronteira-de-isolamento) | Coleções como fronteira de isolamento | Aceita |
| [009](#adr-009-cross-encoder-local-e-adaptador-de-api-de-rerank) | Cross-encoder local e adaptador de API `/rerank` | Planejada |
| [010](#adr-010-perguntas-golden-sintéticas-com-revisão-humana) | Perguntas golden sintéticas com revisão humana | Planejada |
| [011](#adr-011-indexação-explícita-e-retomável) | Indexação explícita e retomável, com coleções vinculadas ao modelo | Aceita |
| [012](#adr-012-uma-tabela-de-vetores-sem-dimensão-fixa) | Uma tabela de vetores sem dimensão fixa | Aceita |
| [013](#adr-013-full-text-do-postgresql-como-base-lexical) | Full-text do PostgreSQL (consulta com OU, `ts_rank_cd`) como método lexical | Aceita |
| [014](#adr-014-busca-vetorial-exata-para-o-benchmark) | Busca vetorial exata, sem índice ANN | Aceita |

---

## ADR-001: Arquitetura hexagonal

**Decisão:** domínio → aplicação (casos de uso + ports) → adaptadores, conectados em um único composition root.

**Por quê:** o projeto existe para *comparar* estratégias e provedores. Os ports transformam cada um deles em uma implementação substituível, enquanto a lógica central que está sendo medida fica fixa. Isso também permite que a maioria dos testes rode sem infraestrutura.

| Alternativa | Por que não |
|-------------|-------------|
| MVC em camadas ao redor do FastAPI | As regras de negócio acabam nas rotas e nos modelos do ORM; trocar o armazenamento ou o provedor afeta tudo |
| App centrada no LangChain (chains em tudo) | Começa rápido, mas o framework vira a arquitetura; difícil de testar e de explicar cada etapa |

## ADR-002: Camada de referência independente do chunking

**Decisão:** processar cada documento uma vez em um texto canônico. Armazenar páginas, seções, parágrafos, chunks e (depois) rótulos golden como spans `[início, fim)` sobre ele, e medir a relevância pela sobreposição dos spans.

**Por quê:** rótulos golden presos a ids de chunk quebram sempre que o chunking muda, o que torna impossível uma matriz chunking × recuperação. Com spans, os rótulos são permanentes e os chunk sets são descartáveis.

| Alternativa | Por que não |
|-------------|-------------|
| Rotular ids de chunk | Os rótulos precisam ser refeitos para cada configuração de chunking |
| Rotular apenas documentos | Grosso demais: um PDF de 50 páginas é "relevante" para qualquer pergunta sobre ele |
| Associar por similaridade de texto | Impreciso e não reproduzível; os limiares viram uma variável oculta |

## ADR-003: Tamanho do chunk em caracteres

**Decisão:** `chunk_size` e `chunk_overlap` contam caracteres.

**Por quê:** a contagem de tokens depende do tokenizer de cada modelo de embeddings. Caracteres mantêm os chunk sets idênticos entre provedores, então comparar provedores de embeddings compara apenas os embeddings. **Trade-off:** o tamanho em tokens varia um pouco entre modelos. Isso é aceitável, porque os limites ficam bem acima dos tamanhos típicos.

## ADR-004: Chunkers próprios, LangChain na borda dos provedores

**Decisão:** os algoritmos de chunking são escritos no domínio e são exatos nos offsets por construção. O LangChain é usado nos adaptadores, para os clientes de LLM e de embeddings (fases 2 e 5).

**Por quê:** offsets exatos são a base do ADR-002. Os splitters de bibliotecas retornam strings e recuperam os offsets depois. Escrever cerca de 150 linhas testadas sai mais barato do que contornar isso.

## ADR-005: PostgreSQL para tudo

**Decisão:** um único PostgreSQL guarda a camada de referência, os embeddings com pgvector (fase 2) e os índices full-text (fase 3). O acesso é via SQLAlchemy Core, e as migrações via Alembic.

**Por quê:** recuperação vetorial, lexical e híbrida rodam no mesmo motor, então as comparações de latência são justas, e uma consulta híbrida pode ser um único comando SQL. Um único serviço também mantém o `docker compose up` simples.

| Alternativa | Por que não |
|-------------|-------------|
| Banco vetorial dedicado (Qdrant, Weaviate) + Elasticsearch | Dois serviços a mais; a latência entre motores distorce a comparação |
| ORM do SQLAlchemy | As entidades de domínio são dataclasses congeladas; o ORM exige classes mapeadas mutáveis ou modelos duplicados |
| `Base.metadata.create_all()` | Sem histórico de schema; o Alembic oferece migrações versionadas e revisáveis |

## ADR-006: Provedores compatíveis com OpenAI configurados no `.env`

**Decisão:** LLM e embeddings são chamados pela API compatível com OpenAI, com `*_BASE_URL`, `*_API_KEY` e `*_MODEL` vindos do `.env` e configurados de forma independente.

**Por quê:** um único protocolo cobre OpenAI, OpenRouter, Ollama, llama.cpp, vLLM e outros. Trocar de provedor é mudar configuração, exatamente o que um laboratório de benchmark precisa. **Consequência:** as coleções ficam vinculadas ao modelo e à dimensão de embeddings usados na indexação.

## ADR-007: Streamlit com CSS customizado

**Decisão:** Streamlit para a UI, estilizado conforme o design de referência (fundo branco, destaque `#2563EB`, Inter, cards, badges, evidências destacadas). A UI é um cliente HTTP da API.

| Alternativa | Por que não |
|-------------|-------------|
| React / Next.js | Mais fiel ao design pixel a pixel, mas traz uma segunda linguagem e toolchain para um projeto cujo núcleo é recuperação |
| Streamlit importando os serviços diretamente | Acopla UI e núcleo; a API deixaria de ser o único ponto de entrada testado |

## ADR-008: Coleções como fronteira de isolamento

**Decisão:** toda leitura recebe um `collection_id` no nível do port. Acesso entre coleções retorna 404. Testes de contrato provam isso em todos os adaptadores.

**Por quê:** "zero falhas críticas de isolamento" é uma meta do projeto. Colocar o escopo na *assinatura* torna o vazamento impossível por construção, em vez de depender de lembrar de um filtro. Autenticação multi-tenant ficou fora do MVP de propósito.

## ADR-009: Cross-encoder local e adaptador de API de rerank

**Decisão (planejada, fase 4):** o reranker padrão é um cross-encoder do Sentence Transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rodando no processo. Um segundo adaptador chama endpoints HTTP `/rerank` (estilo Jina/Cohere, TEI, llama.cpp, vLLM). A escolha vem de `RERANKER_PROVIDER`.

**Por quê:** a API da OpenAI não tem endpoint de rerank. Um modelo local pequeno funciona offline em CPU, e o adaptador de API evita levar o torch para a imagem quando já existe um serviço de rerank.

## ADR-010: Perguntas golden sintéticas com revisão humana

**Decisão (planejada, fase 6):** o LLM propõe perguntas a partir de trechos amostrados; o trecho de origem vira o span rotulado; uma tela no Streamlit permite que uma pessoa edite, aprove ou rejeite as perguntas e adicione as suas.

**Por quê:** o corpus é arbitrário, então não existem rótulos de antemão. Perguntas totalmente sintéticas copiam a redação do trecho e inflam as pontuações lexicais (BM25). A revisão humana mantém o benchmark honesto.

## ADR-011: Indexação explícita e retomável

**Decisão:** gerar embeddings é uma ação explícita por chunk set (`POST …/index`), fora do upload. O trabalho pendente são os "chunks sem vetor", processados em lotes com um commit por lote. A primeira execução vincula a coleção ao modelo e à dimensão de embeddings; um modelo diferente retorna 409, a menos que `rebuild=true`.

**Por quê:** o upload continua rápido e funciona offline. O custo e o tempo da indexação podem ser medidos por chunk set para o benchmark. Uma falha do provedor no meio do caminho (rate limit, timeout) perde no máximo um lote, e a próxima chamada retoma sem precisar de uma tabela de jobs.

| Alternativa | Por que não |
|-------------|-------------|
| Gerar embeddings durante o upload | A latência do upload passaria a depender de uma API remota; uma falha rejeitaria um documento perfeitamente válido |
| Fila de jobs em background (Celery, RQ) | Um serviço e um broker a mais para um MVP; o design retomável entrega a maior parte do benefício |
| Uma única transação para o chunk set inteiro | Uma falha em 95% jogaria fora todos os embeddings já pagos |
| Sem vínculo, guardando só o modelo por vetor | Coleções com modelos misturados tornariam as similaridades silenciosamente sem sentido |

## ADR-012: Uma tabela de vetores sem dimensão fixa

**Decisão:** uma única tabela `chunk_embeddings` com uma coluna `vector` sem tipo de dimensão, mais uma coluna `dimensions` protegida por `CHECK (vector_dims(embedding) = dimensions)`.

**Por quê:** as coleções ficam vinculadas a modelos diferentes (768, 1024, 1536… dimensões), e uma coluna tipada `vector(n)` obrigaria um único modelo para o banco inteiro. **Trade-off:** índices HNSW exigem dimensão fixa. A recuperação não precisa de um (veja o ADR-014); se uma instalação grande precisar, ele pode ser criado por coleção como índice parcial de expressão (`(embedding::vector(n)) … WHERE collection_id = …`).

| Alternativa | Por que não |
|-------------|-------------|
| Coluna `vector(1536)` | Prende todas as coleções a um único modelo |
| Uma tabela por coleção ou por dimensão | DDL dinâmico na aplicação, migrações e isolamento mais difíceis de raciocinar |
| Coluna de vetor em `chunks` | Uma reconstrução reescreveria a tabela da camada de referência; vetores são dados derivados e ficam separados |

## ADR-013: Full-text do PostgreSQL como base lexical

**Decisão:** o método lexical é a busca full-text do PostgreSQL. Cada chunk guarda `to_tsvector(<idioma da coleção>, text)` (índice GIN). A pergunta vira `plainto_tsquery(idioma, pergunta)` com `&` trocado por `|`, e os resultados são ordenados com `ts_rank_cd(..., 1)` (densidade de cobertura dividida por `1 + log(tamanho)`). O idioma é uma configuração por coleção (`fulltext_language`).

**Por quê:** roda no mesmo banco dos vetores (latência justa, um único serviço), trata stemming e stop words para cerca de 30 idiomas e é uma base bem conhecida. A troca por OU importa: com E, uma pergunta em linguagem natural só encontra chunks que contêm *todas* as palavras de conteúdo, e o recall lexical despencaria por motivos que nada têm a ver com o método.

**Trade-off:** `ts_rank_cd` não é BM25. Não tem frequência inversa de documento, então um termo comum pesa o mesmo que um raro. O laboratório o chama de "full-text", e não de "BM25".

| Alternativa | Por que não |
|-------------|-------------|
| Extensão BM25 (ParadeDB `pg_search`, VectorChord-bm25) | Não vem na imagem oficial do pgvector; exigiria uma imagem própria e uma extensão nativa para um MVP |
| BM25 em Python (`rank_bm25`) | Carrega todos os chunks na memória a cada consulta; a latência não seria comparável com a do método vetorial rodando em SQL |
| Elasticsearch / OpenSearch | Um segundo motor; a latência entre motores distorce a comparação (ADR-005) |
| Um único idioma global | Corpora mistos são comuns; um corpus em português com stemming em inglês perde recall em silêncio |

## ADR-014: Busca vetorial exata para o benchmark

**Decisão:** a busca vetorial é uma varredura exata (`ORDER BY embedding <=> :query`) sobre os vetores de um chunk set. Nenhum índice HNSW ou IVFFlat é criado.

**Por quê:** o laboratório mede *estratégias de recuperação*. Um índice aproximado acrescenta uma segunda fonte de erro, sem controle: um Recall@5 baixo poderia vir dos embeddings ou do índice ANN pulando vizinhos. Os corpora de benchmark são pequenos (milhares de chunks), e nesse tamanho uma varredura exata leva milissegundos. HNSW com filtros (chunk set, documentos) também perde resultados, a menos que as varreduras iterativas estejam ativas (pgvector ≥ 0.8).

**Trade-off:** a latência cresce linearmente com o chunk set. Para um corpus de milhões de chunks, crie um índice HNSW parcial de expressão por coleção (ADR-012) e reporte o recall dele separadamente.

| Alternativa | Por que não |
|-------------|-------------|
| HNSW por coleção (como planejado originalmente) | DDL dinâmico por coleção e dimensão; resultados aproximados contaminam a comparação; consultas filtradas podem retornar menos que `top_k` |
| IVFFlat | Precisa de dados de treino antes de ser criado e tem o mesmo problema de aproximação |
