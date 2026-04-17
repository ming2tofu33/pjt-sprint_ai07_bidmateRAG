"""Retriever orchestration.

검색 흐름:
  1. 질문에서 메타데이터 필터 자동 추출
  2. 필요하면 멀티턴 rewrite / history agency 상속
  3. 비교/나열형 질문이면 기관별 fan-out 검색
  4. Dense 또는 Hybrid 검색으로 후보 청크 조회
  5. Cross-Encoder 리랭킹
  6. 섹션/테이블 부스팅
  7. shortlist / where_document 과적용 시 fallback
"""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_matched_agencies,
    extract_metadata_filters,
    extract_project_clues,
    extract_range_filters,
    extract_section_hint,
    should_fan_out_multi_source_query,
)
from bidmate_rag.retrieval.hybrid import hybrid_query, resolve_hybrid_pool_sizes
from bidmate_rag.retrieval.multiturn import (
    extract_recent_agency_filter,
    rewrite_query_with_history,
)
from bidmate_rag.retrieval.reranker import (
    _assign_ranks,
    cross_encoder_rerank,
    rerank_with_boost,
)


class RAGRetriever:
    """메타데이터 필터와 벡터 검색을 결합하는 RAG 리트리버."""

    def __init__(
        self,
        vector_store,
        embedder,
        metadata_store=None,
        sparse_store=None,
        reranker_model=None,
        enable_multiturn: bool = True,
        boost_config: dict | None = None,
        hybrid_config: dict | None = None,
        rewrite_llm=None,
        rewrite_mode: str = "llm_with_rule_fallback",
        rewrite_max_completion_tokens: int = 16000,
        rewrite_timeout_seconds: int = 30,
        memory=None,
        debug_trace_enabled: bool = True,
    ) -> None:
        """RAGRetriever를 초기화한다."""
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store
        self.sparse_store = sparse_store
        self.reranker = reranker_model
        self.enable_multiturn = enable_multiturn
        self.boost_config = boost_config
        self.hybrid_config = hybrid_config
        self.rewrite_llm = rewrite_llm
        self.rewrite_mode = rewrite_mode
        self.rewrite_max_completion_tokens = rewrite_max_completion_tokens
        self.rewrite_timeout_seconds = rewrite_timeout_seconds
        self.memory = memory
        self.debug_trace_enabled = debug_trace_enabled
        self._last_debug: dict = {}

    def _serialize_results(self, results: list) -> list[dict]:
        return [result.to_record() for result in results]

    def _extract_scope_key(self, where: dict | None) -> tuple[str, list[str]] | None:
        if not where:
            return None
        for key in ("발주 기관", "파일명", "사업명"):
            value = where.get(key)
            if not isinstance(value, dict):
                continue
            scoped_values = value.get("$in")
            if isinstance(scoped_values, list) and len(scoped_values) >= 2:
                return key, scoped_values
        return None

    def _should_run_scoped_queries(
        self,
        query: str,
        where: dict | None,
        *,
        force_scoped: bool = False,
    ) -> bool:
        if self._extract_scope_key(where) is None:
            return False
        return force_scoped or should_fan_out_multi_source_query(query)

    def _build_scoped_filters(self, where: dict) -> list[dict]:
        scoped_target = self._extract_scope_key(where)
        if scoped_target is None:
            return [where]
        scope_key, scoped_values = scoped_target
        shared_filters = {key: value for key, value in where.items() if key != scope_key}
        return [{**shared_filters, scope_key: value} for value in scoped_values]

    def _doc_identity(self, result) -> str:
        metadata = getattr(result.chunk, "metadata", {}) or {}
        return str(
            metadata.get("파일명")
            or getattr(result.chunk, "doc_id", "")
            or getattr(result.chunk, "chunk_id", "")
        )

    def _merge_round_robin(self, grouped_results: list[list], top_k: int) -> list:
        merged: list = []
        seen_chunk_ids: set[str] = set()
        seen_doc_ids: set[str] = set()
        max_group_len = max((len(results) for results in grouped_results), default=0)

        def collect(*, prefer_new_docs: bool) -> bool:
            for index in range(max_group_len):
                for results in grouped_results:
                    if index >= len(results):
                        continue
                    item = results[index]
                    chunk_id = item.chunk.chunk_id
                    if chunk_id in seen_chunk_ids:
                        continue
                    doc_id = self._doc_identity(item)
                    if prefer_new_docs and doc_id in seen_doc_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)
                    seen_doc_ids.add(doc_id)
                    merged.append(item)
                    if len(merged) >= top_k:
                        return True
            return False

        if collect(prefer_new_docs=True):
            return _assign_ranks(merged)
        collect(prefer_new_docs=False)
        return _assign_ranks(merged)

    def _has_doc_level_constraint(self, where: dict | None) -> bool:
        if not where:
            return False
        return "파일명" in where or "사업명" in where

    def _extract_history_text(self, message: dict) -> list[str]:
        texts: list[str] = []

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())

        for key in ("user", "assistant"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        return texts

    def _build_history_aware_query(self, query: str, chat_history: list[dict] | None) -> str:
        if not chat_history:
            return query
        texts: list[str] = []
        for message in chat_history[-4:]:
            texts.extend(self._extract_history_text(message))
        texts.append(query)
        return " ".join(texts).strip()

    def _augment_where_with_history_docs(
        self,
        query: str,
        where: dict | None,
        chat_history: list[dict] | None,
    ) -> dict | None:
        if self.metadata_store is None or not chat_history or self._has_doc_level_constraint(where):
            return where
        combined_query = self._build_history_aware_query(query, chat_history)
        relevant_docs = self.metadata_store.find_relevant_docs(combined_query, top_n=3)
        if not relevant_docs:
            return where
        return {**(where or {}), "파일명": {"$in": relevant_docs}}

    def _augment_where_with_project_docs(
        self,
        query: str,
        where: dict | None,
    ) -> dict | None:
        if self.metadata_store is None or self._has_doc_level_constraint(where):
            return where
        project_clues = extract_project_clues(query)
        if len(project_clues) < 2:
            return where

        candidate_docs: list[str] = []
        seen_docs: set[str] = set()
        for clue in project_clues[:5]:
            for doc in self.metadata_store.find_relevant_docs(clue, top_n=2):
                if doc and doc not in seen_docs:
                    candidate_docs.append(doc)
                    seen_docs.add(doc)

        if len(candidate_docs) < 2:
            return where
        return {**(where or {}), "파일명": {"$in": candidate_docs}}

    def _build_where_document(
        self,
        query: str,
        where: dict | None,
        section_hint: str | None,
        *,
        force_scoped: bool = False,
    ) -> dict | None:
        if not section_hint:
            return None
        if self._should_run_scoped_queries(query, where, force_scoped=force_scoped):
            return None
        return {"$contains": section_hint}

    def _query_vector_store(
        self,
        *,
        query_embedding: list[float],
        query: str,
        top_k: int,
        dense_pool_k: int,
        sparse_pool_k: int,
        where: dict | None,
        where_document: dict | None,
        force_scoped: bool = False,
    ) -> list:
        if self._should_run_scoped_queries(query, where, force_scoped=force_scoped):
            scoped_dense_top_k = max(top_k * 2, dense_pool_k if self.reranker is not None else 0)
            scoped_sparse_top_k = (
                max(top_k * 2, sparse_pool_k if self.reranker is not None else 0)
                if sparse_pool_k
                else 0
            )
            grouped_results = [
                hybrid_query(
                    query=query,
                    query_embedding=query_embedding,
                    vector_store=self.vector_store,
                    sparse_store=self.sparse_store,
                    dense_top_k=scoped_dense_top_k,
                    sparse_top_k=scoped_sparse_top_k,
                    where=scoped_where,
                    where_document=where_document,
                    hybrid_config=self.hybrid_config,
                )
                for scoped_where in self._build_scoped_filters(where)
            ]
            return self._merge_round_robin(
                grouped_results,
                max(scoped_dense_top_k, scoped_sparse_top_k or 0),
            )

        return hybrid_query(
            query=query,
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            sparse_store=self.sparse_store,
            dense_top_k=dense_pool_k,
            sparse_top_k=sparse_pool_k,
            where=where,
            where_document=where_document,
            hybrid_config=self.hybrid_config,
        )

    def retrieve(
        self,
        query: str,
        chat_history=None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        """쿼리에 대해 메타데이터 필터링 후 벡터 검색을 수행한다."""

        agency_list = getattr(self.metadata_store, "agency_list", [])
        rewrite_memory_state = (
            self.memory.build(
                chat_history or [],
                current_question=query,
                rewritten_query=None,
            )
            if self.enable_multiturn and self.memory is not None
            else {"summary_buffer": "", "slot_memory": {}}
        )
        rewrite_slot_memory = rewrite_memory_state.get("slot_memory", {})
        resolved_query, rewrite_trace = (
            rewrite_query_with_history(
                query,
                chat_history,
                agency_list,
                llm=self.rewrite_llm,
                mode=self.rewrite_mode,
                slot_memory=rewrite_slot_memory,
                max_completion_tokens=self.rewrite_max_completion_tokens,
                timeout_seconds=self.rewrite_timeout_seconds,
            )
            if self.enable_multiturn
            else (
                query,
                {
                    "original_query": query,
                    "rewritten_query": query,
                    "rewrite_applied": False,
                    "rewrite_reason": "original",
                    "rewrite_prompt_tokens": 0,
                    "rewrite_completion_tokens": 0,
                    "rewrite_total_tokens": 0,
                    "rewrite_cost_usd": 0.0,
                },
            )
        )

        # Generation용 memory state는 rewritten_query를 반영해 한 번 더 빌드.
        # rewrite LLM이 본 slot_memory와 generation LLM이 보는 slot_memory를
        # 소스 단일화 — chat 파이프라인이 이 state만 재사용한다.
        generation_memory_state = (
            self.memory.build(
                chat_history or [],
                current_question=query,
                rewritten_query=resolved_query,
            )
            if self.enable_multiturn and self.memory is not None
            else None
        )
        matched_agencies = extract_matched_agencies(resolved_query, agency_list)
        project_clues = extract_project_clues(resolved_query)
        force_scoped = len(matched_agencies) >= 2 or len(project_clues) >= 2

        base_where: dict | None = None
        if metadata_filter is not None:
            base_where = dict(metadata_filter) if metadata_filter else None
            where = dict(base_where) if base_where else None
            where = self._augment_where_with_history_docs(resolved_query, where, chat_history)
            where = self._augment_where_with_project_docs(resolved_query, where)
            where = self._augment_where_with_history_docs(resolved_query, where, chat_history)
        else:
            base_where = extract_metadata_filters(
                resolved_query,
                agency_list,
                chat_history=chat_history,
            )
            where = dict(base_where) if base_where else None

            if len(matched_agencies) >= 2:
                where = {**(where or {}), "발주 기관": {"$in": matched_agencies}}
                base_where = dict(where)

            if self.enable_multiturn and (where is None or "발주 기관" not in where):
                history_agency_filter = extract_recent_agency_filter(chat_history, agency_list)
                if history_agency_filter:
                    where = {**history_agency_filter, **(where or {})}
                    if base_where:
                        base_where = {**history_agency_filter, **base_where}

            range_filter = extract_range_filters(resolved_query)
            if range_filter:
                where = {**(where or {}), **range_filter}
                base_where = {**(base_where or {}), **range_filter}

            where = self._augment_where_with_history_docs(resolved_query, where, chat_history)
            where = self._augment_where_with_project_docs(resolved_query, where)
            where = self._augment_where_with_history_docs(resolved_query, where, chat_history)

            if where is None and self.metadata_store is not None:
                relevant_docs = self.metadata_store.find_relevant_docs(resolved_query, top_n=3)
                if relevant_docs:
                    where = {"파일명": {"$in": relevant_docs}}

            if base_where == {}:
                base_where = None

        final_top_k = top_k
        dense_pool_k, sparse_pool_k = resolve_hybrid_pool_sizes(
            final_top_k,
            reranker_present=self.reranker is not None,
            sparse_store=self.sparse_store,
            hybrid_config=self.hybrid_config,
        )
        section_hint = extract_section_hint(resolved_query)
        where_document = (
            self._build_where_document(
                resolved_query,
                where,
                section_hint,
                force_scoped=force_scoped,
            )
            if metadata_filter is not None
            else None
        )
        query_embedding = self.embedder.embed_query(resolved_query)

        results = self._query_vector_store(
            query_embedding=query_embedding,
            query=resolved_query,
            top_k=final_top_k,
            dense_pool_k=dense_pool_k,
            sparse_pool_k=sparse_pool_k,
            where=where,
            where_document=where_document,
            force_scoped=force_scoped,
        )

        if not results and where_document is not None:
            results = self._query_vector_store(
                query_embedding=query_embedding,
                query=resolved_query,
                top_k=final_top_k,
                dense_pool_k=dense_pool_k,
                sparse_pool_k=sparse_pool_k,
                where=where,
                where_document=None,
            )

        if (
            not results
            and base_where != where
            and base_where is not None
            and self._has_doc_level_constraint(where)
            and not self._has_doc_level_constraint(base_where)
        ):
            base_where_document = (
                self._build_where_document(
                    resolved_query,
                    base_where,
                    section_hint,
                    force_scoped=force_scoped,
                )
                if metadata_filter is not None
                else None
            )
            results = self._query_vector_store(
                query_embedding=query_embedding,
                query=resolved_query,
                top_k=final_top_k,
                dense_pool_k=dense_pool_k,
                sparse_pool_k=sparse_pool_k,
                where=base_where,
                where_document=base_where_document,
                force_scoped=force_scoped,
            )

        before_rerank = list(results)
        results = cross_encoder_rerank(self.reranker, resolved_query, results, final_top_k)
        reranked = rerank_with_boost(
            results,
            query=resolved_query,
            section_hint=section_hint,
            boost_config=self.boost_config,
        )
        final_results = _assign_ranks(reranked[:final_top_k])
        if self.debug_trace_enabled:
            self._last_debug = {
                **rewrite_trace,
                "rewrite_slot_memory": rewrite_slot_memory,
                "where": where,
                "where_document": where_document,
                "retrieved_chunks_before_rerank": self._serialize_results(before_rerank[:final_top_k]),
                "retrieved_chunks_after_rerank": self._serialize_results(final_results),
                "memory_state": generation_memory_state,
            }
        else:
            # Keep the minimum runtime contract even when verbose tracing is off.
            # chat still needs rewritten_query and rewrite token/cost values to keep
            # generation input and cost accounting aligned with the retrieval step.
            self._last_debug = {
                "rewritten_query": resolved_query,
                "rewrite_prompt_tokens": int(
                    rewrite_trace.get("rewrite_prompt_tokens", 0) or 0
                ),
                "rewrite_completion_tokens": int(
                    rewrite_trace.get("rewrite_completion_tokens", 0) or 0
                ),
                "rewrite_total_tokens": int(
                    rewrite_trace.get("rewrite_total_tokens", 0) or 0
                ),
                "rewrite_cost_usd": float(
                    rewrite_trace.get("rewrite_cost_usd", 0.0) or 0.0
                ),
                "memory_state": generation_memory_state,
            }
        return final_results
