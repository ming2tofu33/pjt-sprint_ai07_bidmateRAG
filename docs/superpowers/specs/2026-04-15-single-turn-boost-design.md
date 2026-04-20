# Single-Turn Boost Design

## Summary

This change improves single-turn retrieval quality without using a Cross-Encoder.
The goal is to make ranking more robust with lightweight rules while keeping the
current dense retrieval pipeline intact.

## Scope

In scope:
- Single-turn retrieval only
- Rule-based boost after dense retrieval
- No new model dependencies
- No multi-turn memory or session carry-over

Out of scope:
- Multi-turn retrieval logic
- Cross-Encoder changes
- Hybrid BM25 + dense search
- Reindexing or embedding regeneration

## Current Behavior

The current pipeline retrieves dense candidates, optionally applies a
Cross-Encoder, then applies a small rule-based boost.

Current boost signals:
- Section hint match: `+0.1`
- Table query + table chunk: `+0.1`

Current issues:
- Boost is too weak to noticeably change ranking when Cross-Encoder is off
- Section hint is used both as a boost signal and as a pre-retrieval
  `where_document` filter, which can cause zero-result retrieval for spacing or
  formatting variants such as `사업개요` vs `사업 개요`
- When Cross-Encoder is disabled, the candidate pool is small, so there is
  little room for lightweight reranking to help

## Proposed Design

### 1. Use section hints only for reranking

Do not apply `section_hint` as a `where_document` pre-filter during retrieval.
Use it only as a post-retrieval boost signal.

Expected effect:
- Reduce silent zero-result failures
- Keep candidate recall wide enough for lightweight reranking to work

### 2. Expand candidate pool when Cross-Encoder is off

When no Cross-Encoder is configured, retrieve more than `top_k` candidates
before boost-based reranking.

Recommended behavior:
- `rerank_pool_k = final_top_k * 3` when Cross-Encoder is off
- Keep current `final_top_k * 4` behavior when Cross-Encoder is on

Expected effect:
- Give the boost rules enough headroom to reorder candidates meaningfully

### 3. Strengthen boost with metadata-aware signals

Apply conservative, additive score boosts on top of dense similarity.

Recommended signals:
- Section match: `+0.15`
- Table query + table chunk: `+0.08`
- Agency metadata match: `+0.12`
- Project/file-title style match: `+0.12`

Apply a total boost cap:
- Maximum additive boost per chunk: `+0.30`

Expected effect:
- Better ranking for queries that mention issuing agency, project title, or
  section intent
- Maintain dense score as the dominant signal

## Matching Rules

### Section match

If `extract_section_hint(query)` returns a section and the chunk section matches
that hint, add section boost.

### Table match

If `should_boost_tables(query)` is true and the chunk `content_type` is
`table`, add table boost.

### Agency metadata match

Use chunk metadata fields already present in the parquet:
- `발주 기관`
- `resolved_agency`
- `original_agency`

If the query contains a normalized agency string that matches any of the above
for the chunk, add agency boost.

### Project/file-title style match

Use lightweight substring checks against:
- `사업명`
- `파일명`
- `doc_id`

This is meant for explicit project-title-like queries, not broad fuzzy matching.
Use conservative normalization to avoid overboosting generic words.

## Implementation Areas

Primary files:
- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/retrieval/reranker.py`

Possible helper updates:
- `src/bidmate_rag/retrieval/filters.py`

Implementation outline:
1. Remove `section_hint`-based `where_document` filtering from retrieval
2. Adjust candidate pool sizing for the no-reranker path
3. Extend `rerank_with_boost()` to include metadata-aware boosts
4. Keep boost logic isolated and easy to disable or tune

## Safety Constraints

- Do not change embedding or vector store schema
- Do not require reindexing
- Do not introduce heavy model inference
- Keep additive boosts small enough that dense retrieval remains the primary
  ranking signal

## Verification

### Functional checks

- Retrieval no longer returns zero results solely because of a section-spacing
  variant
- Queries that mention agency or project names show improved top-k ordering
- Table-oriented queries rank table chunks higher

### Regression checks

- Existing single-turn evaluation still runs without new dependencies
- No failures when `reranker_model` is `null`
- Rankings remain stable for generic queries without strong metadata signals

### Evaluation plan

Compare before/after on the existing single-turn eval set with Cross-Encoder off:
- Baseline: current boost behavior
- Variant: strengthened boost behavior

Track:
- Hit rate
- MRR
- NDCG
- MAP
- Qualitative failures on known section/agency/project queries

## Risks

- Overboosting generic project or agency tokens may cause false positives
- Simple substring matching can be noisy if normalization is too permissive
- Candidate expansion increases retrieval cost slightly, though still far below
  Cross-Encoder cost

## Recommendation

Start with the conservative score values above and evaluate them on the current
single-turn set. Only after single-turn behavior is stable should multi-turn
signals or hybrid retrieval be added.
