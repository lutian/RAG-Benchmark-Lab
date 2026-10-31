# Arquitetura

O RAG Benchmark Lab usa **arquitetura hexagonal** (ports and adapters). As regras de negócio ficam em um núcleo puro. Frameworks, bancos de dados e provedores de modelos se conectam por fora. Assim, estratégias de recuperação, provedores de LLM e armazenamento podem ser trocados e comparados sem tocar nas regras que estão sendo medidas.

> Outros idiomas: [English](../en/architecture.md)

## As camadas de relance

```mermaid
flowchart LR
    subgraph Inbound["Adaptadores de entrada"]
        API["FastAPI<br/>adapters/inbound/api"]
        UI["Streamlit UI<br/>ui/ (cliente HTTP)"]
    end
    subgraph Core["Núcleo"]
        APP["Aplicação<br/>casos de uso + ports"]
        DOM["Domínio<br/>camada de referência, chunking"]
    end
    subgraph Outbound["Adaptadores de saída"]
        PAR["Parsers<br/>PDF / Markdown / TXT"]
        PG["PostgreSQL + pgvector<br/>SQLAlchemy Core + Alembic"]
        MEM["Store em memória"]
        EMB["Provedor de embeddings<br/>API compatível com OpenAI (LangChain)"]
    end
    UI -- HTTP --> API
    API --> APP
    APP --> DOM
    APP -. port .-> PAR
    APP -. port .-> PG
    APP -. port .-> MEM
    APP -. port .-> EMB
```

| Camada | Pacote | Propósito | Pode importar |
|--------|--------|-----------|---------------|
| Domínio | `rag_lab.domain` | Conceitos de negócio e algoritmos puros (spans, documentos, chunking) | Apenas a biblioteca padrão |
| Aplicação | `rag_lab.application` | Casos de uso e os ports (interfaces) de que precisam | Domínio |
| Adaptadores | `rag_lab.adapters` | Implementam ports (saída) ou chamam casos de uso (entrada) | Aplicação, domínio, frameworks |
| Composition root | `rag_lab.bootstrap` | Decide qual adaptador implementa cada port | Tudo |
| UI | `rag_lab.ui` | Páginas Streamlit; fala com o sistema **somente pela API HTTP** | `config` e seus próprios módulos |

**A regra de dependência:** as setas apontam apenas para dentro. O domínio nunca importa FastAPI, SQLAlchemy, pypdf ou LangChain. O verificador de tipos (`mypy --strict`) e a suíte de testes rodam em todo commit, então uma violação aparece imediatamente.

## Fluxos de dados

### Ingestão (implementada)

```mermaid
sequenceDiagram
    participant C as Cliente (UI / curl)
    participant A as Rota da API
    participant S as IngestionService
    participant P as ParserRegistry
    participant D as Domínio (chunking)
    participant U as UnitOfWork
    C->>A: POST /collections/{id}/documents (arquivo)
    A->>S: ingest_document(collection_id, filename, bytes)
    S->>S: limite de tamanho, hash sha256
    S->>U: duplicado? (mesmo hash na coleção)
    S->>P: parse(filename, bytes)
    P-->>S: texto canônico + páginas/seções/parágrafos + metadados
    S->>D: chunk_document(document, config) para cada chunk set
    S->>U: grava documento + chunks, commit (uma transação)
    A-->>C: 201 resumo do documento + quantidade de chunks por set
```

### Indexação (implementada)

```mermaid
sequenceDiagram
    participant C as Cliente
    participant S as IndexingService
    participant U as UnitOfWork
    participant E as Embedder (compatível com OpenAI)
    C->>S: index_chunk_set(coleção, chunk set, rebuild?)
    S->>U: vincula a coleção a modelo + dimensão (ou 409 se já vinculada a outro), commit
    loop até não haver chunks pendentes
        S->>U: próximo lote de chunks sem vetor
        S->>E: embed_documents(textos)  (sem transação aberta)
        S->>S: confere quantidade e dimensão dos vetores
        S->>U: grava os vetores, commit
    end
    S-->>C: relatório (total, já indexados, gerados agora, segundos)
```

Cada lote recebe seu próprio commit, então uma falha do provedor no meio do caminho não perde nada: a próxima execução gera só o que ainda está pendente.

### Recuperação (implementada)

```mermaid
sequenceDiagram
    participant C as Cliente
    participant R as RetrievalService
    participant E as Embedder
    participant DB as SearchRepository
    C->>R: search(coleção, chunk set, pergunta, método)
    R->>DB: verifica o chunk set (+ vínculo e cobertura para vector/hybrid)
    alt vector ou hybrid
        R->>E: embed_query(pergunta)
        R->>DB: vector(melhores candidatos)
    end
    alt fulltext ou hybrid
        R->>DB: fulltext(melhores candidatos)
    end
    R->>R: reciprocal_rank_fusion (hybrid)
    R-->>C: resultados + posição por método + tempos
```

### Perguntas e respostas (planejado, fases 4–5)

`Pergunta → estratégia de recuperação (vetorial / full-text / híbrida) → filtros → reranking → contexto → LLM → resposta com citações`, ou abstenção quando a evidência é insuficiente.

## A camada de referência: por que o chunking pode variar livremente

Esta é a decisão de design mais importante do projeto. Veja o [ADR-002](decisions.md#adr-002-camada-de-referência-independente-do-chunking).

- Cada documento é processado **uma única vez** em um **texto canônico**.
- Páginas, seções e parágrafos são armazenados como **spans** (offsets de caracteres `[início, fim)`) sobre esse texto.
- Um **chunk** é apenas um span com metadados derivados (números de página, caminho de títulos).
- Os rótulos do golden dataset (fase 6) também serão spans sobre o mesmo texto.

```text
texto canônico:  |# Install\nRun it.\n\n## Docker\ndocker compose up\n|
seções:          [0 ─────────── 19)[19 ───────────────────────── 47)
fixed-20-0:      [0 ──── 20)[20 ──── 40)[40 ─ 47)
structure-800-0: [0 ─────── 17)  [19 ────────────────────── 46)
rótulo golden:                         [29 ──────────── 46)
```

A relevância é medida pela **sobreposição entre spans**, e não pela correspondência de ids de chunk. Um rótulo escrito uma vez continua válido para qualquer configuração de chunking, e é isso que torna possível uma matriz de benchmark chunking × recuperação.

## Isolamento

A **coleção** é a fronteira de isolamento. Todo método de repositório que retorna documentos ou chunks recebe um `collection_id`, então a própria assinatura do port garante o escopo. Os testes de contrato em `tests/contracts/test_isolation.py` rodam contra os adaptadores em memória e PostgreSQL. Eles provam que documentos idênticos em duas coleções ficam separados e que o id de um documento de outra coleção retorna *não encontrado*.

## Próximos passos

| Quero entender… | Leia |
|-----------------|------|
| Spans, documentos, algoritmos de chunking | [domain.md](domain.md) |
| Casos de uso, ports, transações | [application.md](application.md) |
| Parsers, PostgreSQL, API, UI | [adapters.md](adapters.md) |
| Configurações do `.env` e provedores | [configuration.md](configuration.md) |
| Estratégia de testes e como rodá-los | [testing.md](testing.md) |
| Por que cada tecnologia foi escolhida | [decisions.md](decisions.md) |
