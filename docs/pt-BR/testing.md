# Estratégia de testes

O projeto é construído com **testes primeiro**: cada regra ganha um teste que falha (RED), depois o código que o faz passar (GREEN), depois uma refatoração com os testes continuando verdes.

> Outros idiomas: [English](../en/testing.md)

## Caminho rápido

```bash
uv sync                      # instala as dependências de desenvolvimento
uv run pytest                # testes unitários + de contrato (os de PostgreSQL são pulados sem banco)
make lint typecheck          # ruff + mypy --strict
```

Para rodar os testes de contrato com PostgreSQL usando um banco descartável:

```bash
docker compose up -d db
TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_lab uv run pytest -m integration
```

> Os testes **truncam todas as tabelas** desse banco. Nunca aponte `TEST_DATABASE_URL` para dados importantes.

## Pirâmide de testes

| Nível | Local | O que prova | Velocidade |
|-------|-------|-------------|------------|
| Unitário de domínio | `tests/unit/domain/` | Matemática de spans, invariantes de documento, algoritmos de chunking | ms |
| Baseado em propriedades | `tests/unit/domain/test_chunkers.py` | Invariantes do chunking para **qualquer** texto e configuração (Hypothesis) | < 1 s |
| Unitário de adaptadores | `tests/unit/adapters/` | Parsers sobre bytes reais de PDF (gerados com fpdf2), contrato da API via `TestClient` | ms |
| Aplicação | `tests/unit/application/` | Regras dos casos de uso (deduplicação, limite de tamanho, chunking retroativo, atomicidade) | ms |
| Contrato | `tests/contracts/` | **Os mesmos testes contra memória e PostgreSQL**: isolamento, transações, ida e volta da camada de referência | s |
| UI | `tests/unit/ui/` | Páginas renderizam com uma API falsa (`streamlit.testing.AppTest`); renderização dos destaques | ms |

## Testando o adaptador de embeddings

`tests/unit/adapters/embeddings/` sobe um servidor HTTP real em `127.0.0.1` que fala o protocolo `/v1/embeddings` e registra cada requisição. Os testes verificam o que de fato trafega: texto puro (e não ids de tokens), os limites dos lotes, o nome do modelo, o header `Authorization` e a ausência do campo `dimensions`. Também cobrem como respostas 500 e hosts inacessíveis são reportados. Um mock do SDK não pegaria um padrão do LangChain que muda o payload.

Tudo acima do adaptador usa `tests/fakes.py::FakeEmbedder`: vetores determinísticos, com opções para falhar na n-ésima chamada ou retornar a dimensão errada. Os testes de recuperação usam o `ConceptEmbedder`, que coloca sinônimos na mesma dimensão, para que um teste mostre a busca vetorial encontrando uma paráfrase ("car" → "automobile") que a busca full-text não encontra.

## Testando os rerankers

`tests/unit/adapters/reranking/` roda o `HttpReranker` contra um stub `/rerank` local real (corpo da requisição, header de autenticação, scores devolvidos fora de ordem, listas no estilo TEI, erros HTTP, índices faltando, host inacessível) e o `CrossEncoderReranker` com um modelo falso injetado (carregamento preguiçoso, uma única carga, biblioteca ausente). Os testes nunca baixam um modelo; acima do adaptador, quem atua é o `tests/fakes.py::FakeReranker`.

## Testando o modelo de chat

`tests/unit/adapters/generation/` roda o `OpenAICompatibleChatModel` contra um stub local de `/v1/chat/completions`: mensagens e modelo na requisição, `temperature` / `reasoning_effort` enviados só quando configurados, uso de tokens presente ou ausente, erros HTTP e hosts inacessíveis. Acima do adaptador, o `tests/fakes.py::FakeChatModel` devolve respostas roteirizadas e registra os prompts.

## Testando a avaliação

As funções de métrica são testadas com rankings calculados à mão (`tests/unit/domain/test_evaluation.py`). O `BenchmarkService` roda de ponta a ponta em memória com `ConceptEmbedder`, `FakeReranker` e um `FakeChatModel` que responde e julga, verificando o formato da matriz, o aquecimento (uma chamada extra ao reranker por método) e as taxas de geração e do juiz. O CLI é testado com um container em memória e um diretório de saída temporário. O armazenamento do golden dataset tem testes de contrato nos dois adaptadores, e `tests/integration/test_golden_constraints.py` prova no PostgreSQL que um documento rotulado não pode ser apagado sozinho, enquanto sua coleção pode.

## Testando a camada de apresentação

`tests/unit/ui/test_dashboard_data.py` verifica os dados por trás de cada gráfico do dashboard; `test_benchmark_page.py` renderiza a página com uma API falsa (indicadores contra as metas, início de uma execução). O comando de demo roda o corpus e o golden dataset reais de `examples/` com um container em memória, duas vezes, para provar que é idempotente. O `examples/build_golden.py` falha se algum trecho de evidência não for encontrado no texto processado, então o dataset distribuído sempre corresponde ao corpus.

## Comportamento exclusivo do PostgreSQL

`tests/integration/test_postgres_fulltext.py` cobre o que o store em memória deliberadamente não imita: stemming e stop words por idioma da coleção (`rates` encontra `rate` em `english`, mas não em `simple`; `banco` encontra `bancos` em `portuguese`). `tests/integration/test_migrations.py` volta para a 0002, insere um chunk, sobe de novo e prova que a migração 0003 o tornou pesquisável.

## Por que testes de contrato?

O store em memória deixa a maioria dos testes rápida. Mas testes rápidos só são confiáveis se o dublê se comporta como o real, então a suíte de contrato é parametrizada para os dois adaptadores. Se o PostgreSQL e a memória divergirem (ordenação, rollback, isolamento), a suíte falha.

## Por que testes baseados em propriedades para o chunking?

Testes por exemplos cobrem os casos em que pensamos. Os bugs de chunking se escondem nos casos em que ninguém pensou: sequências de espaços, chunk de tamanho 1, overlap quase igual ao tamanho, caracteres acentuados. O Hypothesis gera centenas dessas entradas e verifica as quatro invariantes (fatias exatas, limite de tamanho, nenhum conteúdo perdido, ordem de leitura) em toda execução.

## CI

`.github/workflows/ci.yml` roda em todo push e PR: `uv sync --frozen` → ruff (lint + checagem de formatação) → mypy strict → pytest com cobertura, contra um container de serviço `pgvector/pgvector:pg17`. Por isso, no CI, os testes de contrato sempre rodam contra um PostgreSQL real.
