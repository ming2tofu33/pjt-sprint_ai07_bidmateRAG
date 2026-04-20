"""Shared dense + sparse hybrid retrieval helpers."""

from __future__ import annotations

from collections import defaultdict


def hybrid_enabled(sparse_store, hybrid_config: dict | None) -> bool:
    """하이브리드 검색이 활성화 상태인지 확인한다.

    Args:
        sparse_store: BM25 sparse 스토어. None이면 비활성.
        hybrid_config: 하이브리드 설정 dict.

    Returns:
        sparse_store가 존재하고 enabled=True이면 True.
    """
    return bool(sparse_store is not None and (hybrid_config or {}).get("enabled"))


def resolve_hybrid_pool_sizes(
    final_top_k: int,
    *,
    reranker_present: bool,
    sparse_store=None,
    hybrid_config: dict | None = None,
) -> tuple[int, int]:
    """Dense/Sparse 각각의 후보 풀 크기를 결정한다.

    Args:
        final_top_k: 최종 반환할 청크 수.
        reranker_present: Cross-Encoder 리랭커 사용 여부.
        sparse_store: BM25 sparse 스토어.
        hybrid_config: 하이브리드 설정 dict.

    Returns:
        (dense_top_k, sparse_top_k) 튜플. 하이브리드 비활성이면 sparse=0.
    """
    # 리랭커가 있으면 4배, 없으면 3배로 후보 풀 확장
    dense_top_k = final_top_k * (4 if reranker_present else 3)
    if not hybrid_enabled(sparse_store, hybrid_config):
        return dense_top_k, 0

    cfg = hybrid_config or {}
    dense_multiplier = max(int(cfg.get("dense_pool_multiplier", 3)), 1)
    sparse_multiplier = max(int(cfg.get("sparse_pool_multiplier", 3)), 1)

    dense_top_k = max(dense_top_k, final_top_k * dense_multiplier)
    sparse_top_k = max(final_top_k, final_top_k * sparse_multiplier)
    return dense_top_k, sparse_top_k


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    *,
    limit: int,
    rrf_k: int = 60,
) -> list:
    """Dense + Sparse 결과를 RRF(Reciprocal Rank Fusion)로 융합한다.

    Args:
        dense_results: 벡터 검색 결과 리스트.
        sparse_results: BM25 검색 결과 리스트.
        limit: 최종 반환할 청크 수.
        rrf_k: RRF 상수 (기본 60). 값이 클수록 순위 차이 영향이 완만해진다.

    Returns:
        RRF 점수 기준 정렬된 RetrievedChunk 리스트.
    """
    if not dense_results:
        return sparse_results[:limit]
    if not sparse_results:
        return dense_results[:limit]

    # 각 청크의 RRF 점수 누적: 1/(k + rank)
    fused_scores: dict[str, float] = defaultdict(float)
    chosen_results: dict[str, object] = {}
    dense_lookup = {result.chunk.chunk_id: result for result in dense_results}
    sparse_lookup = {result.chunk.chunk_id: result for result in sparse_results}

    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
            if chunk_id not in chosen_results:
                chosen_results[chunk_id] = result.model_copy(deep=True)

    # RRF 점수 상위 limit개 선택 후 0~1 정규화
    ordered_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:limit]
    raw_scores = [fused_scores[chunk_id] for chunk_id in ordered_ids]
    max_score = max(raw_scores)
    min_score = min(raw_scores)

    fused_results: list = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        result = chosen_results[chunk_id]
        raw_score = fused_scores[chunk_id]
        # min-max 정규화
        normalized = 1.0 if max_score == min_score else (raw_score - min_score) / (
            max_score - min_score
        )
        result.rank = rank
        result.score = round(normalized, 4)
        # 디버깅용 메타데이터 기록
        result.chunk.metadata["hybrid_rrf_score"] = round(raw_score, 6)
        if chunk_id in dense_lookup:
            result.chunk.metadata["hybrid_dense_score"] = dense_lookup[chunk_id].score
        if chunk_id in sparse_lookup:
            result.chunk.metadata["hybrid_sparse_score"] = sparse_lookup[chunk_id].score
        result.chunk.metadata["retrieval_source"] = "hybrid"
        fused_results.append(result)
    return fused_results


def hybrid_query(
    *,
    query: str,
    query_embedding: list[float],
    vector_store,
    sparse_store=None,
    dense_top_k: int,
    sparse_top_k: int = 0,
    where: dict | None = None,
    where_document: dict | None = None,
    hybrid_config: dict | None = None,
) -> list:
    """Dense 벡터 검색 + Sparse BM25 검색을 실행하고 RRF로 융합한다.

    Args:
        query: 사용자 질의 문자열 (BM25용).
        query_embedding: 질의 임베딩 벡터 (Dense용).
        vector_store: 벡터 검색 스토어.
        sparse_store: BM25 sparse 스토어. None이면 Dense만 사용.
        dense_top_k: Dense 후보 풀 크기.
        sparse_top_k: Sparse 후보 풀 크기.
        where: ChromaDB where 필터.
        where_document: ChromaDB where_document 필터.
        hybrid_config: 하이브리드 설정 dict.

    Returns:
        하이브리드 활성이면 RRF 융합 결과, 아니면 Dense 결과만 반환.
    """
    dense_results = vector_store.query(
        query_embedding=query_embedding,
        top_k=dense_top_k,
        where=where,
        where_document=where_document,
    )
    if not hybrid_enabled(sparse_store, hybrid_config):
        return dense_results

    sparse_results = sparse_store.query(
        query=query,
        top_k=sparse_top_k or dense_top_k,
        where=where,
    )
    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        limit=max(dense_top_k, sparse_top_k or 0),
        rrf_k=int((hybrid_config or {}).get("rrf_k", 60)),
    )
