# Camada de aplicação — `rag_lab.application`

A camada de aplicação contém os **casos de uso** (o que o sistema sabe fazer) e os **ports** (o que ele precisa do mundo externo). Ela orquestra objetos de domínio e nunca sabe se o parser é o pypdf ou se o banco é o PostgreSQL.

> Outros idiomas: [English](../en/application.md)

## Módulos

| Módulo | Propósito |
|--------|-----------|
| `services/ingestion.py` | `IngestionService`: coleções, ingestão de documentos, chunk sets |
| `services/indexing.py` | `IndexingService`: embeddings dos chunk sets, vínculo, progresso |
| `services/answering.py` | `AnswerService`: recuperação → contexto numerado → LLM → citações verificadas ou abstenção |
| `prompts.py` | O prompt de resposta fundamentada (regras, fontes numeradas) |
| `services/golden.py` | `GoldenDatasetService`: perguntas sintéticas e manuais, revisão, export/import |
| `services/benchmark.py` | `BenchmarkService`: a matriz chunking × estratégia sobre as perguntas aprovadas |
| `reporting.py` / `run_codec.py` | Relatório Markdown de uma execução; codificação JSON versionada das execuções |
| `services/retrieval.py` | `RetrievalService`: busca vetorial, full-text, híbrida e híbrida com reranking, com filtros e tempos |
| `ports/parsing.py` | Port `DocumentParser` e sua saída `ParsedDocument` |
| `ports/embeddings.py` | Ports `Embedder` (o modelo) e `EmbeddingRepository` (o armazenamento de vetores) |
| `ports/generation.py` | Port `ChatModel`, `ChatMessage`, `Completion` |
| `ports/reranking.py` | `Reranker`: pontua pares (pergunta, texto) |
| `ports/search.py` | `SearchRepository`: buscas vetoriais e full-text ordenadas sobre um chunk set |
| `ports/repositories.py` | Ports de repositório e a fronteira transacional `UnitOfWork` |
| `errors.py` | Falhas esperadas, traduzidas em códigos HTTP pela API |

## `IngestionService`

| Caso de uso | Comportamento |
|-------------|---------------|
| `create_collection(name, configs)` | Normaliza e valida o nome; nome duplicado lança `ConflictError` |
| `add_chunking_config(id, config)` | Registra um novo chunk set **e divide todos os documentos existentes** com ele, em uma transação |
| `ingest_document(id, filename, bytes)` | Limite de tamanho → SHA-256 → checagem de duplicado → parsing → cria o `Document` → chunking com cada configuração → persistência atômica |
| `list_documents`, `get_document`, `list_chunks` | Leituras, sempre com escopo de `collection_id` |

**Ingestão idempotente:** enviar os mesmos bytes duas vezes para uma coleção retorna o documento existente com `duplicate=True`. Reenvios nunca inflam o corpus, o que distorceria os resultados do benchmark.

**Atomicidade:** o documento, sua estrutura e todos os seus chunk sets são gravados em uma única unit of work. Uma falha (por exemplo, um PDF corrompido) não deixa nada para trás. Isso é verificado nos dois adaptadores por `tests/contracts/test_transactions.py`.

**Por que uma classe de serviço e não uma classe por caso de uso?** Os casos de uso de ingestão compartilham as mesmas dependências (unit of work, parser, limites) e são pequenos. Um serviço coeso mantém o composition root simples. Quando um caso de uso crescer (recuperação, avaliação), ganha seu próprio serviço.

## `IndexingService`

| Caso de uso | Comportamento |
|-------------|---------------|
| `index_chunk_set(id, key, rebuild=False)` | Vincula a coleção à spec do embedder e gera os embeddings dos chunks pendentes lote a lote (`EMBEDDING_BATCH_SIZE`), com um commit por lote. Retorna um `IndexingReport` (total, já indexados, gerados agora, segundos) |
| `index_chunk_set(..., rebuild=True)` | Apaga todos os vetores da coleção, desfaz o vínculo e indexa do zero com o modelo atual |
| `status(id)` | `IndexStatus`: o vínculo e `embedded / total` de cada chunk set |

Pontos de design:

- **O provedor é chamado fora de qualquer transação.** Uma API de embeddings lenta nunca segura uma conexão nem locks no banco.
- **Retomável por construção.** "Pendente" significa "chunk sem vetor", então rodar de novo após uma falha continua de onde parou. Não é preciso uma tabela de jobs.
- **Os vetores são validados antes de gravar.** A quantidade precisa bater com o lote, e todo vetor precisa ter a dimensão vinculada. Caso contrário, `EmbeddingProviderError` indica qual configuração revisar (`EMBEDDING_DIMENSIONS`).
- **Ação explícita, não no upload.** O upload continua rápido e funciona sem provedor. A indexação pode ser repetida por chunk set e cronometrada para o benchmark. Veja o [ADR-011](decisions.md#adr-011-indexação-explícita-e-retomável).
- **Loop limitado.** O loop de lotes roda no máximo `ceil(pendentes / lote) + 1` vezes, então um armazenamento com defeito nunca prende o processo em um loop infinito.

## `RetrievalService`

| Caso de uso | Comportamento |
|-------------|---------------|
| `search(id, key, query, method, top_k=10, document_ids=None)` | Ordena os chunks de um chunk set e retorna um `SearchResult`: resultados (chunk, score, posição, posição por método) e tempos (`embed_ms`, `search_ms`, `rerank_ms`, `total_ms`) |

Fluxo: validar (pergunta não vazia, `1 ≤ top_k ≤ 100`) → verificar se o chunk set existe → para `vector`/`hybrid`, verificar o vínculo de embeddings e a cobertura → gerar o embedding da pergunta → buscar → fundir (híbrida) → reordenar (`hybrid_rerank`).

Pontos de design:

- **Scores vetoriais só fazem sentido contra um índice completo do mesmo modelo.** `vector` e `hybrid` retornam 409 quando a coleção não está indexada, está vinculada a um modelo diferente do embedder configurado, ou tem chunks sem vetor (`3/4 chunks`). Um índice parcial silencioso distorceria todos os números do benchmark. `fulltext` funciona logo após o upload.
- **A híbrida busca mais fundo que o `top_k`.** Cada método retorna `RETRIEVAL_CANDIDATES` (padrão 50) chunks antes da fusão, porque um chunk em 12º lugar nos dois métodos pode merecer uma posição no top 5 fundido.
- **Tempo por etapa.** Gerar o embedding da pergunta (uma chamada de rede) e buscar (o banco) são medidos separadamente, para o benchmark mostrar de onde vem a latência. O full-text nunca chama o embedder (`embed_ms = 0`).
- **Filtros.** `document_ids` restringe todos os métodos a esses documentos (a etapa "Filtros" do fluxo da pergunta).
- **Reranking (`hybrid_rerank`).** Os `RERANK_CANDIDATES` (padrão 30) melhores resultados da híbrida vão para o `Reranker`, que pontua cada par (pergunta, chunk); os resultados são reordenados por esse score (empates mantêm a ordem da híbrida) e cortados em `top_k`. Cada resultado guarda `ranks["hybrid"]`, sua posição antes do reranking, então a UI mostra movimentos como "hybrid #7 → #1". O tempo do reranking é reportado em `rerank_ms`. Um reranker que retorna um número diferente de scores gera `RerankerError`.
- **Os vetores da pergunta são validados** contra a dimensão do embedder; um tamanho errado é `EmbeddingProviderError` (502), nunca um erro do banco.
- **Resultados explicáveis.** Cada resultado traz `ranks`, por exemplo `{"vector": 9, "fulltext": 1}` para um chunk que só o método lexical encontrou perto do topo.

## `AnswerService`

| Caso de uso | Comportamento |
|-------------|---------------|
| `answer(id, key, question, method=hybrid, top_k=5, document_ids=None)` | Recupera com o `RetrievalService`, numera os resultados como fontes, chama o `ChatModel` e depois faz o parsing e a checagem das citações. Retorna um `Answer`: texto, `abstained` + `abstention_reason`, fontes, citações válidas (número → resultado), `CitationCheck`, modelo, uso de tokens e tempos (etapas da recuperação + `generation_ms`) |

Pontos de design:

- **Dois motivos de abstenção, dois culpados.** `no_evidence_retrieved`: a recuperação não retornou nada, então o LLM **não é chamado** (sem custo e sem chance de alucinar). `model_abstained`: havia fontes, mas o modelo respondeu `NO_ANSWER`. Ao diagnosticar uma resposta ruim, o primeiro aponta para a recuperação; o segundo, para a *qualidade* da recuperação ou para a geração.
- **Todas as fontes são retornadas**, citadas ou não, para você ver se o chunk certo foi recuperado e ignorado (problema de geração) ou nunca recuperado (problema de recuperação).
- **O prompt fica em `application/prompts.py`**: um único lugar versionado, com testes unitários para as suas regras (fontes numeradas, sentinela, resposta no idioma da pergunta). As fontes são rotuladas com página e seção para o modelo conseguir distingui-las.
- **Sem limiar de score.** Os scores vetoriais e de RRF não são calibrados entre corpora e modelos, então um corte fixo faria o sistema se abster arbitrariamente. O modelo decide pelo conteúdo; o benchmark mede com que frequência ele acerta ao fazer isso.

## `GoldenDatasetService`

| Caso de uso | Comportamento |
|-------------|---------------|
| `generate(id, count, seed)` | Divide cada documento em trechos de ~400 caracteres com o chunker recursivo (independente dos chunk sets da coleção), descarta trechos com menos de 100 caracteres ou já usados como evidência, sorteia com `random.Random(seed)` alternando entre documentos e pede ao LLM uma pergunta por trecho (`SKIP` = sem conteúdo factual). Cada pergunta nasce como **rascunho** e recebe commit assim que existe |
| `add_manual(id, text, document_id, excerpt)` | Localiza o trecho colado no texto canônico, ignorando diferenças de espaço (quebras de linha de PDF); 422 quando ele não está lá |
| `questions` / `views` / `update` / `delete` | Listar (opcionalmente por status), listar com trechos e nomes de arquivo para revisão, editar texto ou status, apagar |
| `export(id)` / `import_(id, payload)` | JSON portável cuja evidência leva o **hash do conteúdo** do documento, o span e o trecho; a importação mapeia os hashes para os documentos da coleção de destino e pula as perguntas cujos arquivos não existem lá |

**Por que trechos sorteados com semente a partir da camada de referência?** A mesma semente e os mesmos documentos geram os mesmos trechos, então o dataset pode ser regenerado; usar o texto de referência (e não um chunk set) mantém os rótulos independentes do chunking que está sendo testado.

## `BenchmarkService`

Executa um `BenchmarkConfig` (chunk sets × métodos, `ks`, corte do MRR, geração e juiz opcionais) sobre as perguntas **aprovadas**:

1. Validar: os chunk sets existem (404) e há perguntas aprovadas (422).
2. **Aquecer** cada chunk set × método uma vez, sem medir. A primeira consulta com reranking carrega o cross-encoder; sem aquecimento, esse custo cairia sobre uma pergunta qualquer. O aquecimento também falha cedo (ex.: 409 para um chunk set não indexado) antes de qualquer medição.
3. Para cada chunk set × método × pergunta: recuperar `max(ks, mrr_cutoff)` chunks, medindo tempo de relógio e de CPU, e pontuar com `score_ranking`.
4. Com `generation`: responder com `answer_top_k` fontes (abstenção, validade das citações, tokens); com `judge`: mais uma chamada ao LLM retorna `SUPPORTED` / `UNSUPPORTED` (`parse_verdict`; qualquer outra coisa conta como desconhecido).
5. `summarize` por estratégia e gravar a execução.

## Relatórios — `reporting.py` e `run_codec.py`

`markdown_report(run, collection_name)` gera: a configuração (modelos e parâmetros em uso), a matriz de recuperação com ✅/❌ em relação às metas do projeto, os confrontos diretos (lexical vs vetorial, híbrida vs vetorial, rerank vs híbrida), a qualidade da geração e o diagnóstico recuperação vs geração. `best_strategy` (maior Recall@5, depois MRR, depois menor p50) é a mesma no relatório, na API e no CLI. `run_codec` é o formato JSON único e versionado de uma execução (gravado no PostgreSQL, servido pela API, escrito pelo CLI).

## Ports

### `ChatModel`

```python
name: str
def complete(self, messages: Sequence[ChatMessage]) -> Completion  # texto, modelo, tokens de prompt/completion
```

As implementações lançam `GenerationProviderError`. As contagens de tokens são `None` quando o provedor não informa o uso.

### `Reranker`

```python
name: str
def score(self, query: str, texts: Sequence[str]) -> list[float]  # um score por texto, na ordem de entrada
```

Um cross-encoder lê a pergunta e o chunk **juntos**, então julga a relevância melhor do que comparar dois embeddings calculados separadamente, ao custo de uma passada do modelo por candidato. Por isso ele só reordena uma lista curta de candidatos. As implementações lançam `RerankerError`.


### `DocumentParser`

```python
def parse(self, filename: str, data: bytes) -> ParsedDocument
```

Retorna o texto canônico, os elementos estruturais e os metadados, ou lança `UnsupportedDocumentError` / `DocumentParsingError`. O serviço atribui a identidade (id, hash, coleção) depois, então os parsers continuam sendo transformações puras.

### Repositórios e `UnitOfWork`

| Port | Responsabilidade |
|------|------------------|
| `CollectionRepository` | Adicionar, atualizar, buscar por id ou nome, listar |
| `DocumentRepository` | Adicionar, buscar **(collection_id, document_id)**, listar por coleção, buscar por hash |
| `ChunkRepository` | Substituir os chunks de um documento em um chunk set; listar um chunk set (opcionalmente de um documento) |
| `EmbeddingRepository` | Chunks pendentes de um chunk set, gravar vetores, contar, apagar todos os vetores de uma coleção |
| `SearchRepository` | `vector(...)`: vizinhos mais próximos exatos por similaridade de cosseno; `fulltext(...)`: chunks que compartilham ao menos um termo normalizado com a pergunta, ordenados por cobertura. Ambos com escopo de coleção e chunk set, e filtro opcional por documento |
| `GoldenQuestionRepository` | Adicionar (a evidência precisa ser da coleção), atualizar só texto/status, buscar, listar por status, apagar |
| `BenchmarkRunRepository` | Adicionar, buscar, listar da mais nova para a mais antiga |
| `UnitOfWork` | Context manager que expõe os repositórios; `commit()` torna as mudanças visíveis, e sair do bloco sem commit desfaz tudo |

### `Embedder`

```python
spec: EmbeddingSpec
def embed_documents(self, texts: Sequence[str]) -> list[list[float]]
def embed_query(self, text: str) -> list[float]
```

As implementações encapsulam as falhas do provedor em `EmbeddingProviderError`. `embed_query` é usado pela recuperação para a pergunta.

**O isolamento está nas assinaturas.** Nenhum método de leitura retorna documentos ou chunks sem um `collection_id`. Quem chama não tem como "esquecer o filtro", porque não existe método sem filtro.

**Por que Unit of Work + Repository em vez de passar a sessão do banco adiante?**

| Opção | Problema neste contexto |
|-------|-------------------------|
| `Session` do SQLAlchemy nos serviços | Acopla os casos de uso ao SQLAlchemy; o adaptador em memória fica impossível |
| Repositórios que fazem commit a cada chamada | Um documento poderia ser salvo sem os chunks, quebrando a atomicidade |
| **Unit of Work (escolhida)** | Uma fronteira transacional explícita que funciona igual no PostgreSQL e em memória |

## Erros

| Erro | Significado | HTTP |
|------|-------------|------|
| `NotFoundError` | A entidade não existe **ou pertence a outra coleção** | 404 |
| `ConflictError` | Nome de coleção ou chunk set duplicado; coleção vinculada a outro modelo de embeddings; busca vetorial/híbrida em um chunk set que não está totalmente indexado | 409 |
| `InvalidRequestError` | Valores que não podem ser usados: `hybrid_rerank` sem reranker, pergunta vazia, `top_k` fora do intervalo, idioma de full-text desconhecido, nome de coleção em branco | 422 |
| `EmbeddingProviderError` | A API de embeddings falhou, estava inacessível ou retornou vetores do tamanho errado | 502 |
| `GenerationProviderError` | O LLM de chat falhou, estourou o timeout ou estava inacessível (o corpo da resposta não é repassado) | 502 |
| `RerankerError` | O reranker não carregou (falta o extra `local-rerank`), a API dele falhou ou retornou scores malformados | 502 |
| `UploadTooLargeError` | Arquivo maior que `MAX_UPLOAD_MB` | 413 |
| `UnsupportedDocumentError` | A extensão não é `.pdf`, `.md`, `.markdown` ou `.txt` | 415 |
| `DocumentParsingError` | Tipo suportado, mas ilegível (corrompido, vazio, PDF escaneado sem texto) | 422 |

Um documento de outra coleção retorna **404 e não 403**, então a API não revela nem que aquele id existe em outro lugar.
