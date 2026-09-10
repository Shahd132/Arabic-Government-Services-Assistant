# 03 — Retrieval (Embeddings + Hybrid RAG)

Owner: Person 3

## What this module does

Takes the chunks produced by **Person 2** (`02_knowledge_base_person2`) and
builds a retrieval system that, given a user question, returns the most
relevant chunks to hand to **Person 5**'s generation step.

```
Chunks ──► Transformer (multilingual-E5) ──► Embeddings ──► FAISS (dense index)
Chunks ──► BM25 (sparse index) ─────────────────────────────────────┘
                                     │
                              Hybrid Search (fusion)
                                     │
                              Top-K chunks  ──►  retrieve(query)
```

## Files

| File | Purpose |
|---|---|
| `src/embedder.py` | Wraps `intfloat/multilingual-e5-base` (sentence-transformers) with the E5 `query:`/`passage:` prefix convention. Also ships `MockEmbedder`, a dependency-free stand-in for offline/CI smoke tests. |
| `src/dense_index.py` | FAISS `IndexFlatIP` over L2-normalized embeddings (= cosine similarity). `DenseIndexRegistry` keeps one index per department + one over everything. |
| `src/sparse_index.py` | BM25 (`rank_bm25`) with Arabic normalization (strips diacritics, unifies alef/ta-marbuta/alef-maqsura variants, light "ال" stemming) before tokenizing. |
| `src/hybrid_search.py` | Fuses dense + sparse results. Default: **Reciprocal Rank Fusion (RRF)**. Alternative: min-max normalized weighted sum (`alpha` param). |
| `src/build_index.py` | CLI: loads Person 2's chunk JSON files, embeds, builds + saves dense & sparse indices to `vector_store/`. |
| `src/retrieve.py` | **Public API.** `retrieve(query, top_k=5, department=None) -> List[RetrievedChunk]`. Loads indices from `vector_store/` (lazy, cached). |
| `src/evaluate_retrieval.py` | Computes Recall@K, Precision@K, MRR against a labeled qrels file. Writes `results/retrieval_metrics.json`. |
| `data/eval/qrels.json` | Hand-labeled eval queries + relevant chunk ids (small starter set — extend this before reporting final numbers). |
| `data/sample_chunks/` | Small sample of Person 2's chunk format (civil_affairs / tax / traffic) used to test this module in isolation. |
| `tests_local/test_retrieval_smoke.py` | Fast smoke tests using `MockEmbedder` (no model download needed). |

## Setup

```bash
pip install sentence-transformers faiss-cpu rank_bm25
```

## Build the indices

Point `--chunks_dir` at Person 2's real output once it's available:

```bash
python src/build_index.py \
    --chunks_dir ../02_knowledge_base_person2/output/chunks \
    --out_dir vector_store \
    --model intfloat/multilingual-e5-base
```

For a quick local test against the bundled sample data (no GPU/model
download required):

```bash
python src/build_index.py --chunks_dir data/sample_chunks --out_dir vector_store --use_mock_embedder
```

## Use it

```python
from retrieve import retrieve

chunks = retrieve("ازاي اجدد رخصة القيادة؟", top_k=5, department="traffic")
for c in chunks:
    print(c.score, c.chunk.law_name, c.chunk.text[:100])
```

`department` should come from Person 4's router output
(`RouterOutput.department` in `shared/schemas.py`). Pass `None` to search
across all departments.

## Evaluate

```bash
python src/evaluate_retrieval.py \
    --chunks_dir data/sample_chunks \
    --qrels data/eval/qrels.json \
    --fusion rrf \
    --out results/retrieval_metrics.json
```

Reports **Recall@K**, **Precision@K** (K = 1, 3, 5, 10), and **MRR**, both
as an aggregate summary and per-query, so failure cases can be inspected
directly.

**Note on `data/eval/qrels.json`**: it currently has 9 hand-written queries
against the 12 sample chunks — enough to validate the pipeline mechanics,
not enough for a defensible report number. Before the final report, either:
- hand-label a larger set (ideally reusing/relabeling some of Person 4's
  router training questions, since those are already real citizen-style
  questions), or
- at minimum, use `build_synthetic_qrels()` in `evaluate_retrieval.py` as a
  stopgap and say so explicitly in the report.

## Design decisions & why

- **Multilingual-E5** over an Arabic-only model: citizens sometimes type
  legal terms in English/Franco-Arabic, and E5 has strong multilingual
  coverage plus retrieval-specific training (the query/passage prefix
  convention actually matters — don't skip it).
- **RRF as the default fusion**: doesn't require score normalization
  across dense/sparse (which live on very different scales), and is a
  well-established robust default. The weighted alternative (`alpha`
  tunable) is kept for a report comparison, and `evaluate_retrieval.py`
  supports running both.
- **BM25 kept as a first-class citizen, not an afterthought**: legal text
  is full of exact terms (article numbers, law numbers, named terms) that
  a general-purpose embedding model can under-weight. Hybrid > dense-only
  here.
- **Per-department indices**: since Person 4's router already classifies
  the department before retrieval runs, scoping search to one department's
  index is both faster and more precise than filtering a single big index
  post-hoc. An `__all__` index is still kept for cross-department fallback
  / router-uncertain cases.

## Known limitations / TODO

- `vector_store/` currently holds indices built from `data/sample_chunks/`
  (12 chunks) for wiring validation — rebuild against Person 2's full
  corpus before integration.
- Real embedding quality (multilingual-E5) hasn't been benchmarked yet in
  this environment (sandbox had no disk budget for `torch` — see
  `embedder.py`'s `MockEmbedder` used for local testing instead). Run
  `build_index.py` without `--use_mock_embedder` in an environment with
  GPU/disk to get real numbers before the final report.
- `notebooks/embedding_comparison.ipynb` (comparing multilingual-E5 vs
  Arabic-only alternatives) is a placeholder — fill in once real qrels
  exist.
