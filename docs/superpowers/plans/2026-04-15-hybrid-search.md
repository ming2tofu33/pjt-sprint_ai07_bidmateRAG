# Hybrid Search Plan

1. Add runtime/config support for hybrid retrieval and experiment-specific chunk
   path resolution.
2. Implement a lightweight BM25 sparse store over `chunks.parquet`.
3. Implement shared dense + sparse + RRF fusion helpers.
4. Connect the shared hybrid search path to both `RAGRetriever` and
   `web_api/retrieval_helpers.py`.
5. Keep existing Cross-Encoder and boost logic, but ensure the final result is
   clipped back to `top_k`.
6. Add and update unit tests for runtime wiring, retriever behavior, and web
   helper behavior.
