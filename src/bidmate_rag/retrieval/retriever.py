"""Retriever orchestration."""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_matched_agencies,
    extract_metadata_filters,
    extract_project_clues,
    extract_range_filters,
    extract_section_hint,
    is_comparison_query,
    should_fan_out_multi_source_query,
    should_boost_tables,
)


class RAGRetriever:
    """메타데이터 필터와 벡터 검색을 결합하는 RAG 리트리버."""

    def __init__(self, vector_store, embedder, metadata_store=None) -> None:
        """RAGRetriever를 초기화

        Args:
            vector_store: 벡터 검색에 사용할 벡터 스토어.
            embedder: 쿼리 임베딩 생성기.
            metadata_store: 메타데이터 기반 문서 필터링 스토어.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store

    def _extract_scope_key(self, where: dict | None) -> tuple[str, list[str]] | None:
        if not where:
            return None
        value = where.get("발주 기관")
        if isinstance(value, dict):
            scoped_values = value.get("$in")
            if isinstance(scoped_values, list) and len(scoped_values) >= 2:
                return "발주 기관", scoped_values
        return None

    def _should_run_scoped_queries(self, query: str, where: dict | None) -> bool:
        return should_fan_out_multi_source_query(query) and self._extract_scope_key(where) is not None

    def _build_where_document(
        self,
        query: str,
        where: dict | None,
        section_hint: str | None,
    ) -> dict | None:
        if not section_hint:
            return None
        # 비교형 다기관 질의에서는 section hint를 hard filter로 쓰면
        # 한 기관 문서가 통째로 탈락할 수 있으므로 rerank에서만 활용한다.
        if self._should_run_scoped_queries(query, where):
            return None
        return {"$contains": section_hint}

    def _build_scoped_filters(self, where: dict) -> list[dict]:
        scoped_target = self._extract_scope_key(where)
        if scoped_target is None:
            return [where]
        scope_key, scoped_values = scoped_target
        shared_filters = {key: value for key, value in where.items() if key != scope_key}
        return [{**shared_filters, scope_key: value} for value in scoped_values]

    def _scoped_query_top_k(self, top_k: int) -> int:
        return max(top_k, top_k * 2)

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
            return self._assign_ranks(merged)
        collect(prefer_new_docs=False)
        return self._assign_ranks(merged)

    def _assign_ranks(self, results: list) -> list:
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results

    def _should_rerank_results(self, section_hint: str | None, table_boost: bool) -> bool:
        return bool(section_hint or table_boost)

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
            for doc in self.metadata_store.find_relevant_docs(clue, top_n=1):
                if doc and doc not in seen_docs:
                    candidate_docs.append(doc)
                    seen_docs.add(doc)

        if len(candidate_docs) < 2:
            return where
        return {**(where or {}), "파일명": {"$in": candidate_docs}}

    def _query_vector_store(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        query: str,
        where: dict | None,
        where_document: dict | None,
    ) -> list:
        if self._should_run_scoped_queries(query, where):
            scoped_top_k = self._scoped_query_top_k(top_k)
            grouped_results = [
                self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=scoped_top_k,
                    where=scoped_where,
                    where_document=where_document,
                )
                for scoped_where in self._build_scoped_filters(where)
            ]
            return self._merge_round_robin(grouped_results, top_k)

        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
            where_document=where_document,
        )

    def _rerank_results(self, results: list, query: str, section_hint: str | None) -> list:
        if not results:
            return results

        table_boost = should_boost_tables(query)
        if not self._should_rerank_results(section_hint, table_boost):
            return self._assign_ranks(results)

        def boosted_score(result) -> float:
            score = result.score
            if section_hint and section_hint in result.chunk.section:
                score += 0.1
            if table_boost and result.chunk.content_type == "table":
                score += 0.1
            return score

        ordered = sorted(
            enumerate(results),
            key=lambda item: (boosted_score(item[1]), item[1].score, -item[0]),
            reverse=True,
        )
        reranked = []
        for index, (_, result) in enumerate(ordered, start=1):
            result.rank = index
            reranked.append(result)
        return reranked

    def retrieve(
        self,
        query: str,
        chat_history=None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        """쿼리에 대해 메타데이터 필터링 후 벡터 검색을 수행

        Args:
            query: 사용자 질의 문자열.
            chat_history: 이전 대화 이력.
            top_k: 반환할 최대 결과 수.
            metadata_filter: ChromaDB ``where`` 절로 직접 사용할 필터.
                평가셋의 ``metadata_filter`` 또는 Streamlit UI의 수동 필터를
                위한 explicit override. 지정 시 query 기반 자동 추출은 무시되고
                이 값이 그대로 사용됩니다.

        Returns:
            RetrievedChunk 리스트.
        """
        # ``metadata_filter is None`` → 자동 추출 (legacy 기본 동작)
        # ``metadata_filter == {}``   → "필터 없음" 명시 (자동 추출도 비활성화)
        # ``metadata_filter == {...}`` → explicit override
        if metadata_filter is not None:
            base_where = dict(metadata_filter) if metadata_filter else None
            where = dict(base_where) if base_where else None
            where = self._augment_where_with_project_docs(query, where)
            where = self._augment_where_with_history_docs(query, where, chat_history)
        else:
            agency_list = getattr(self.metadata_store, "agency_list", [])
            base_where = extract_metadata_filters(query, agency_list, chat_history=chat_history)
            where = dict(base_where) if base_where else None
            matched_agencies = extract_matched_agencies(query, agency_list)
            if (
                where is None
                and len(matched_agencies) >= 2
                and should_fan_out_multi_source_query(query)
            ):
                where = {"발주 기관": {"$in": matched_agencies}}
                base_where = dict(where)
            range_filter = extract_range_filters(query)
            if range_filter:
                where = {**(where or {}), **range_filter}
                base_where = {**(base_where or {}), **range_filter}
            where = self._augment_where_with_project_docs(query, where)
            where = self._augment_where_with_history_docs(query, where, chat_history)
            if where is None and self.metadata_store is not None:
                relevant_docs = self.metadata_store.find_relevant_docs(query, top_n=3)
                if relevant_docs:
                    where = {"파일명": {"$in": relevant_docs}}
            if base_where == {}:
                base_where = None
        section_hint = extract_section_hint(query)
        where_document = self._build_where_document(query, where, section_hint)
        query_embedding = self.embedder.embed_query(query)
        results = self._query_vector_store(
            query_embedding=query_embedding,
            top_k=top_k,
            query=query,
            where=where,
            where_document=where_document,
        )
        if not results and where_document is not None:
            results = self._query_vector_store(
                query_embedding=query_embedding,
                top_k=top_k,
                query=query,
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
            results = self._query_vector_store(
                query_embedding=query_embedding,
                top_k=top_k,
                query=query,
                where=base_where,
                where_document=self._build_where_document(query, base_where, section_hint),
            )
        return self._rerank_results(results, query=query, section_hint=section_hint)
