# Camada de domínio — `rag_lab.domain`

O domínio é o vocabulário do laboratório: **spans, documentos, chunks, coleções e estratégias de chunking**. É Python puro, sem I/O e sem frameworks. Cada regra aqui é testada em milissegundos e se comporta igual atrás da API, de uma CLI ou de um executor de benchmark.

> Outros idiomas: [English](../en/domain.md)

## Módulos

| Módulo | Propósito | Tipos principais |
|--------|-----------|------------------|
| `spans.py` | Intervalo de caracteres sobre o texto canônico | `Span` |
| `documents.py` | Um documento processado e sua estrutura | `Document`, `StructuralElement`, `ElementKind`, `MediaType` |
| `chunks.py` | Unidade recuperável derivada de um documento | `Chunk` |
| `collections.py` | Fronteira de isolamento, chunk sets a construir, vínculo de embeddings | `Collection`, `EmbeddingMismatchError` |
| `answers.py` | Parsing de citações, validade das citações, sentinela de abstenção | `extract_citations()`, `check_citations()`, `CitationCheck`, `ABSTENTION` |
| `retrieval.py` | Métodos de busca, resultados com score e fusão de rankings | `SearchMethod`, `ScoredChunk`, `reciprocal_rank_fusion()` |
| `embeddings.py` | Qual modelo gerou um vetor e seu tamanho | `EmbeddingSpec` |
| `chunking/config.py` | Como fazer o chunking | `ChunkingConfig`, `ChunkingStrategy` |
| `chunking/strategies.py` | Os algoritmos de chunking | `chunk_document()` |

---

## `Span`

**O que é:** um intervalo semiaberto `[início, fim)` de offsets de caracteres. A regra `início < fim` é garantida, então spans vazios não existem.

**O que faz:** `length`, `overlap(other)` (quantidade de caracteres em comum, `0` quando os spans apenas se tocam) e `contains(offset)`.

**Por que existe:** chunks, citações e rótulos do golden dataset são todos spans do mesmo texto, então todos podem ser comparados com uma única operação, `overlap`. As métricas da fase 6 (Recall@k, MRR) consideram um chunk recuperado relevante quando ele se sobrepõe a um span rotulado.

**Por que semiaberto?** Spans adjacentes `[0,10)` e `[10,20)` não compartilham nenhum caractere e seus tamanhos somam exatamente. O fatiamento do Python (`text[início:fim]`) usa a mesma convenção, então os erros de "off-by-one" desaparecem.

## `Document` e `StructuralElement`

**O que é:** a **camada de referência**. Um documento guarda:

- `text`: o texto canônico, normalizado uma vez (quebras de linha unificadas, sem caracteres NUL);
- `elements`: páginas (PDF), seções (títulos Markdown) ou parágrafos (TXT), cada um com um `Span`, rótulo, nível e `path` de títulos;
- `metadata`: título, autor, número de páginas, encoding… como strings;
- `content_hash`: SHA-256 dos bytes originais, usado para deduplicação.

**Invariantes:** o texto nunca é vazio, e todo elemento fica dentro do texto.

**Auxiliares usados por chunks e citações:**

| Método | Retorna |
|--------|---------|
| `excerpt(span)` | O trecho do texto |
| `section_path_for(span)` | Caminho de títulos da seção mais profunda que contém o início do span, ex.: `("Install", "Docker")` |
| `page_numbers_for(span)` | Todas as páginas (base 1) que o span toca, ex.: `(3, 4)` para um chunk que cruza uma quebra de página |

**Seções cobrem apenas o próprio conteúdo.** Uma seção vai até o próximo título de *qualquer* nível, e seu `path` carrega a cadeia de títulos pais. Assim os elementos nunca se sobrepõem, mas os chunks continuam conhecendo o contexto completo de títulos.

**Por que não guardar só os chunks?** Os chunks mudam a cada configuração. O texto canônico não. Guardar a estrutura como spans significa que o documento é processado uma vez, e quantos chunk sets forem necessários podem ser reconstruídos a partir do banco, sem o arquivo original. Veja o [ADR-002](decisions.md#adr-002-camada-de-referência-independente-do-chunking).

## `Chunk`

Um span de um documento com metadados de localização desnormalizados (`page_numbers`, `section_path`) para citações. Crie chunks com `Chunk.from_document(document, span, index)`, para que `chunk.text == document.text[span.start:span.end]` seja sempre verdade.

## `Collection`

Um conjunto nomeado de documentos, com a lista de `ChunkingConfig` com as quais cada documento é dividido (um **chunk set** por configuração). É a **fronteira de isolamento**: as consultas sempre têm escopo de uma coleção. As chaves de configuração precisam ser únicas, e `with_chunking_config()` retorna uma nova coleção (as entidades são imutáveis).

Ela também guarda o seu **idioma de full-text** (`fulltext_language`, padrão `english`): a configuração de busca textual do PostgreSQL usada para reduzir palavras ao radical (stemming) e descartar stop words. Precisa ser uma das `FULLTEXT_LANGUAGES` (as configurações que vêm com o PostgreSQL, mais `simple`, sem stemming). É escolhido na criação e nunca muda, porque todos os chunks armazenados foram normalizados com ele.

## `EmbeddingSpec` e o vínculo de embeddings

**O que é:** `EmbeddingSpec(model, dimensions)`, um value object validado (modelo não vazio, dimensão positiva).

**O que faz:** uma coleção começa **sem vínculo** (`embedding is None`). A primeira indexação chama `bind_embedding(spec)`:

| Estado da coleção | `bind_embedding(spec)` |
|-------------------|------------------------|
| Sem vínculo | Vincula a `spec` |
| Vinculada à mesma spec | Nada muda |
| Vinculada a outra spec | `EmbeddingMismatchError` ("rebuild the index or create a new collection") |

`unbind_embedding()` é usado na reconstrução (rebuild).

**Por quê:** vetores de modelos diferentes, ou com dimensões diferentes, vivem em espaços diferentes. Comparar o vetor de uma pergunta gerado pelo `nomic-embed-text` com chunks gerados pelo `text-embedding-3-small` produz números, mas sem significado, e nada falharia de forma visível. O vínculo transforma essa corrupção silenciosa em um erro explícito.

## `ChunkingConfig`

`strategy` + `chunk_size` + `chunk_overlap`, ambos **em caracteres**, validados (`size > 0`, `0 ≤ overlap < size`). Sua `key` estável (por exemplo, `recursive-800-120`) nomeia um chunk set no armazenamento, na API e nos relatórios de benchmark.

**Por que caracteres e não tokens?** A contagem de tokens depende do tokenizer do modelo de embeddings. Contar caracteres mantém a configuração idêntica entre provedores, então um benchmark que compara embeddings da OpenAI e do Ollama usa exatamente os mesmos chunks. Veja o [ADR-003](decisions.md#adr-003-tamanho-do-chunk-em-caracteres).

## Estratégias de chunking

`chunk_document(document, config)` delega para uma de três estratégias. Todas compartilham estas garantias, verificadas por testes baseados em propriedades (Hypothesis, 300 casos aleatórios por execução):

1. um chunk é uma **fatia exata** do texto canônico;
2. nenhum chunk passa de `chunk_size`, e nenhum contém apenas espaços em branco;
3. todo caractere que não é espaço pertence a pelo menos um chunk (**nenhum conteúdo se perde**);
4. os chunks vêm em ordem de leitura.

| Estratégia | Como corta | Ponto forte | Ponto fraco | Papel no benchmark |
|------------|------------|-------------|-------------|--------------------|
| `fixed` | Janela deslizante de `size` caracteres, passo `size - overlap` | Simples, previsível | Corta palavras e frases | Linha de base |
| `recursive` | Divide por `\n\n` → `\n` → `. ` → ` ` → caracteres e depois agrupa as partes até `size`, levando ≤ `overlap` caracteres das partes finais | Respeita limites naturais | Ignora a estrutura do documento | Uso geral |
| `structure` | Seções e páginas são limites **rígidos** (nunca cruzados); unidades grandes demais são divididas recursivamente; parágrafos são limites **flexíveis** (os pequenos são unidos) | Chunks alinhados a títulos e páginas, então as citações são precisas | Tamanhos irregulares | Consciente da estrutura |

**Por que implementar os chunkers em vez de usar os splitters do LangChain?** Os text splitters do LangChain retornam strings. O `start_index` opcional é reconstruído depois, procurando o texto de cada chunk no documento, ou seja, os offsets são *recuperados*, não *rastreados*. Offsets exatos são a base da camada de referência, então os algoritmos aqui são exatos por construção (cerca de 150 linhas, totalmente testadas). O LangChain é usado onde agrega valor: nos clientes de LLM e de embeddings das próximas fases. Veja o [ADR-004](decisions.md#adr-004-chunkers-próprios-langchain-na-borda-dos-provedores).

## Recuperação — `SearchMethod` e Reciprocal Rank Fusion

**O que é:** as quatro estratégias que o laboratório compara (`vector`, `fulltext`, `hybrid`, `hybrid_rerank`), o `ScoredChunk` que um método retorna (score maior é melhor) e `reciprocal_rank_fusion()`.

**O que a fusão faz:** a busca híbrida recebe um ranking por método e pontua cada chunk com

```
score(chunk) = Σ sobre os rankings  1 / (k + posição do chunk naquele ranking)
```

Um chunk bem posicionado pelos **dois** métodos vence um chunk que ficou em primeiro em apenas um. Cada resultado fundido guarda a posição que cada método lhe deu (`{"vector": 7, "fulltext": 1}`), e é assim que a UI explica *por que* um chunk foi recuperado. Empates mantêm a ordem do primeiro ranking, então os resultados são determinísticos.

**Por que RRF e não uma soma ponderada de scores?**

| Opção | Problema |
|-------|----------|
| `α·cosseno + (1-α)·ts_rank` | Os scores estão em escalas sem relação (cosseno ∈ [-1, 1], `ts_rank_cd` sem limite e dependente do corpus); `α` precisaria de ajuste por corpus e por pergunta |
| Normalização min-max e soma | Sensível a outliers; um chunk muito longo desloca o score normalizado de todos |
| **RRF (escolhida)** | Usa só as posições, então não precisa de calibração; um único parâmetro (`k`, padrão 60 como no artigo original, `HYBRID_RRF_K`) |

**Por que no domínio?** A fusão é aritmética pura sobre rankings. Mantê-la fora do SQL a torna idêntica para todos os adaptadores de armazenamento e testável em milissegundos.

## Respostas — citações e abstenção

**O que é:** as regras que tornam uma resposta *verificável*. As fontes são numeradas `1..n` no prompt; a resposta as cita como `[n]` ou `[n, m]`; quando as fontes não bastam, o modelo responde com a sentinela `NO_ANSWER`.

| Função | Comportamento |
|--------|---------------|
| `extract_citations(text)` | Números citados, sem repetição, na ordem do primeiro uso. Colchetes colados a uma palavra (`list[0]`) ou com não inteiros (`[1.5]`, `[a]`) não são citações |
| `check_citations(text, n)` | `CitationCheck(cited, invalid)`; `valid` significa ao menos uma citação **e** todos os números em `1..n` |
| `is_abstention(text)` | A resposta começa com `NO_ANSWER` (sem diferenciar maiúsculas, ignorando espaços iniciais e caracteres de markdown ou aspas: asterisco, sublinhado, `>`, `#`, crase, aspas) |

**Por que determinístico?** A meta do projeto "citações válidas ≥ 95%" precisa ser barata e reproduzível. O parsing mede *se as citações apontam para fontes reais*; se a fonte citada de fato sustenta a frase é uma questão semântica, deixada para o juiz LLM opcional (fase de avaliação).

**Por que uma sentinela em vez de saída JSON?** Saída estruturada (`response_format`) não é implementada por todos os servidores compatíveis com OpenAI (llama.cpp, Ollama mais antigo, algumas rotas do OpenRouter). Uma primeira palavra fixa funciona em qualquer um e é trivial de detectar.
