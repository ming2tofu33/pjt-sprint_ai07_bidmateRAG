# Hybrid Search Design

## Summary

This change adds lightweight hybrid retrieval to the existing RAG pipeline by
combining:

- dense vector search
- BM25 sparse search
- RRF rank fusion
- existing rule-based boost

The goal is to improve exact-match recall for agency names, project titles,
document IDs, and other keyword-heavy queries without adding a heavy reranker.

## Scope

In scope:
- chunk-level BM25 over existing `chunks.parquet`
- shared hybrid retrieval core for retriever and web API
- RRF fusion between dense and sparse candidates
- existing boost applied after hybrid retrieval
- runtime/config support for turning hybrid retrieval on or off

Out of scope:
- external search infrastructure
- persistent sparse index storage
- multi-turn retrieval changes
- replacing existing Chroma storage

## Data Source

Hybrid sparse retrieval will use the same chunk source as dense retrieval:

- `data/processed/chunks.parquet`
- `data/processed/{experiment_name}/chunks.parquet`

This keeps dense and sparse retrieval aligned to the same `chunk_id` universe
and avoids introducing a second document source.

## Proposed Retrieval Flow

1. Resolve metadata filters (`where`) as today
2. Run dense retrieval through `vector_store.query()`
3. Run sparse retrieval through a new in-memory BM25 store built from
   `chunks.parquet`
4. Fuse dense and sparse candidates with RRF
5. Apply Cross-Encoder reranking if configured
6. Apply existing rule-based boost
7. Return final `top_k`

## Components

### BM25 sparse store

A new in-memory sparse search component will:

- load chunk text and metadata from `chunks.parquet`
- build a lightweight BM25 index at runtime
- support the same metadata `where` filters needed by current retrieval paths

### Hybrid retrieval core

A shared helper will:

- run dense and sparse searches
- fuse candidates with RRF
- normalize fused scores into a stable `0..1` range for downstream boost logic

### Shared integration

Both retrieval entry points will use the same hybrid core:

- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/web_api/retrieval_helpers.py`

## Runtime and Config

Retrieval config will add a `hybrid` section with:

- `enabled`
- `dense_pool_multiplier`
- `sparse_pool_multiplier`
- `rrf_k`

Runtime assembly will also resolve the correct experiment-specific
`chunks.parquet` path so dense and sparse retrieval use the same chunk set.

## Risks

- In-memory BM25 adds startup overhead when building a runtime pipeline
- Korean tokenization remains lightweight, so sparse recall will still be
  approximate rather than morphology-aware
- RRF score normalization needs to remain stable enough that boost does not
  overwhelm retrieval ordering

## Verification

- unit tests for BM25 sparse filtering and RRF fusion
- retriever tests for hybrid candidate ordering and final `top_k`
- web retrieval helper tests to ensure web path also uses hybrid retrieval
- runtime tests to verify hybrid config and sparse store wiring
