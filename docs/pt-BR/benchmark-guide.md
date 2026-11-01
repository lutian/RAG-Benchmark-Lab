# Guia do benchmark — como executar e ler o experimento

Este guia mostra como rodar a comparação e transformar os números em respostas: por que o chunking importa, quando a busca lexical supera os embeddings, quando o reranking ajuda e se uma resposta ruim vem da recuperação ou da geração.

> Outros idiomas: [English](../en/benchmark-guide.md)

## Caminho rápido

```bash
make demo              # local: ingestão + indexação + importação do golden + benchmark de examples/
make demo-docker       # o mesmo dentro do Compose; o relatório é copiado para ./reports
```

Depois abra **Benchmark** na UI (http://localhost:8501) para explorar a execução. [`examples/sample-report.md`](../../examples/sample-report.md) é uma execução de referência.

## O que uma execução mede

| Eixo | Valores |
|------|---------|
| Chunking | todos os chunk sets da coleção (padrão: `fixed`, `recursive`, `structure` com 800/120) |
| Recuperação | `vector`, `fulltext`, `hybrid` (RRF), `hybrid_rerank` |
| Perguntas | todas as perguntas golden **aprovadas** (meta: 50) |
| Qualidade da recuperação | Recall@1/5/10, MRR@10, por sobreposição com os spans de evidência |
| Custo | latência p50/p95 e tempo de CPU por consulta (depois de um aquecimento) |
| Respostas (opcional) | taxa de abstenção, citações válidas, suporte segundo o juiz LLM, tokens |

## Lendo os resultados — as quatro perguntas

Os números abaixo vêm da execução de referência (52 perguntas, embeddings MiniLM, cross-encoder local, só CPU).

### 1. Por que o chunking afeta a recuperação?

As mesmas perguntas com o mesmo método pontuam diferente em cada chunk set. Na execução de referência, o Recall@5 do full-text vai de 96,2% (`fixed`) a 92,3% (`structure`), enquanto a busca vetorial é melhor em `recursive` (100%). Chunks menores e alinhados à estrutura (42 em vez de 21) dão à busca lexical menos palavras por chunk, então uma pergunta que corresponde a uma frase com palavras *parafraseadas* deixa de compartilhar termos com o seu chunk. Veja a tabela **Rank of the first relevant chunk** no dashboard: uma pergunta cuja posição muda muito entre chunk sets é um efeito do chunking, não do modelo.

### 2. Quando a busca lexical (full-text) supera os embeddings?

Veja **Head to head → Does lexical beat vectors?** Na execução de referência dá 12 vitórias, 12 derrotas e 28 empates em `fixed`, e as posições por pergunta mostram um padrão claro:

- **O full-text vence** quando a pergunta repete um termo marcante do trecho e o embedding confunde temas vizinhos: "Can I return something with a red **Final Sale** tag?", "Do I earn **points** when I buy a **gift card**?", "When do loyalty points **expire**?", "How long does a shop **keep** an order…?" (posição 7 no vetorial, 1 no full-text).
- **Os vetores vencem** nas paráfrases, quando pergunta e trecho usam palavras diferentes para a mesma ideia: "sent by **plane**" vs "by **air**", "a box that arrived **crushed**" vs "**damaged** in transit", "**membership level**" vs "**tier**" (o full-text não encontrou nada), "**split my payment**" vs "**instalments**".

O full-text também é cerca de 10 vezes mais rápido (2–3 ms contra 23–25 ms no p50), porque não precisa chamar o modelo de embeddings. A busca híbrida aproveita os dois: vence 8 e perde 3 contra os vetores sozinhos.

### 3. Quando o reranking ajuda?

**Does reranking help hybrid?** dá 14 vitórias e 0 derrotas em `fixed`. O Recall@5 quase não muda (já estava em 98–100%), mas **o Recall@1 salta de ~71% para ~96%** e o MRR de 0,84 para 0,98. O reranking ajuda quando a evidência já está entre os candidatos, mas não no topo. Ele não resgata um chunk que a primeira etapa nunca recuperou. O custo aparece nas colunas de latência e CPU: ~1,8 s por consulta para 30 candidatos em uma CPU, e por isso ele só reordena uma lista curta.

### 4. O problema está na recuperação ou na geração?

Rode com **Evaluate answers** (e, se quiser, o juiz). A seção *Retrieval or generation?* do relatório separa as respostas ruins:

- **Falha de recuperação**: a evidência não estava entre as `answer_top_k` fontes que o LLM viu. Ajuste o chunking, o método ou o `top_k`; um LLM melhor não vai resolver.
- **Erro de geração**: a evidência estava lá e mesmo assim a resposta se absteve, citou errado ou não foi sustentada. Ajuste o prompt, o modelo ou seus parâmetros.

A página **Ask** mostra o mesmo diagnóstico para uma pergunta: todas as fontes entregues ao LLM, com as citadas marcadas.

## Usando o seu próprio corpus

1. Crie uma coleção em **Ingestion** (escolha o idioma de full-text dos seus documentos) e envie os arquivos.
2. Clique em **Index chunk set** para cada chunk set.
3. Em **Golden dataset**, gere rascunhos com o LLM e **revise-os**: edite perguntas que copiam o trecho palavra por palavra (elas inflam os scores do full-text), rejeite as vagas e adicione as suas. Aprove pelo menos 50.
4. Rode o benchmark em **Benchmark** ou com `rag-lab benchmark --collection <nome>`.
5. Exporte o golden dataset (JSON) para repetir o mesmo benchmark depois ou em outro banco.

## Armadilhas

| Armadilha | Por que importa | O que o laboratório faz |
|-----------|-----------------|-------------------------|
| Perguntas sintéticas copiam o trecho | Inflam o recall lexical | Instrução de paráfrase no prompt + revisão humana |
| Modelo frio na primeira consulta | Uma pergunta absorve segundos de carregamento | Aquecimento não medido por estratégia |
| Índice vetorial aproximado | A perda de recall é atribuída aos embeddings | Busca exata (ADR-014) |
| Comparar modelos com vetores diferentes | Similaridades de espaços diferentes | Coleções vinculadas a um único modelo de embeddings |
| Golden dataset pequeno | Com 50 perguntas, uma pergunta = 2% do Recall@5 | O relatório mostra a quantidade de perguntas; meta de 50 |
| Corpus com idiomas misturados | Stemming errado prejudica o full-text | Idioma de full-text por coleção; `simple` para corpora mistos |
