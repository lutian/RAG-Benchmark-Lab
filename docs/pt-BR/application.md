# Camada de aplicação — `rag_lab.application`

A camada de aplicação contém os **casos de uso** (o que o sistema sabe fazer) e os **ports** (o que ele precisa do mundo externo). Ela orquestra objetos de domínio e nunca sabe se o parser é o pypdf ou se o banco é o PostgreSQL.

> Outros idiomas: [English](../en/application.md)

## Módulos

| Módulo | Propósito |
|--------|-----------|
| `services/ingestion.py` | `IngestionService`: coleções, ingestão de documentos, chunk sets |
| `ports/parsing.py` | Port `DocumentParser` e sua saída `ParsedDocument` |
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

## Ports

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
| `UnitOfWork` | Context manager que expõe os três repositórios; `commit()` torna as mudanças visíveis, e sair do bloco sem commit desfaz tudo |

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
| `ConflictError` | Nome de coleção ou chunk set duplicado | 409 |
| `UploadTooLargeError` | Arquivo maior que `MAX_UPLOAD_MB` | 413 |
| `UnsupportedDocumentError` | A extensão não é `.pdf`, `.md`, `.markdown` ou `.txt` | 415 |
| `DocumentParsingError` | Tipo suportado, mas ilegível (corrompido, vazio, PDF escaneado sem texto) | 422 |

Um documento de outra coleção retorna **404 e não 403**, então a API não revela nem que aquele id existe em outro lugar.
