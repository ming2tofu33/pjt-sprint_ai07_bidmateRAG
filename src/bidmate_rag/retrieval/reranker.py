"""Reranking logic for RAG retrieval.

Cross-Encoder 리랭킹과 섹션/테이블 부스팅을 담당한다.
retriever.py의 오케스트레이션과 분리하여 독립적으로 테스트·교체 가능.
"""

from __future__ import annotations

import re

from bidmate_rag.retrieval.filters import is_comparison_query, should_boost_tables

# 부스팅 기본 상수 (boost_config 미지정 시 폴백용)
SECTION_BOOST = 0.15
TABLE_BOOST = 0.08
METADATA_BOOST = 0.12
MAX_TOTAL_BOOST = 0.30
MIN_MATCH_LEN = 3  # 정규화된 텍스트 최소 매칭 길이


def _normalize_text(value: object) -> str:
    """텍스트를 소문자·특수문자 제거·확장자 제거하여 정규화한다."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\.(pdf|hwp|hwpx|docx|doc)$", "", text)
    return re.sub(r"[\W_]+", "", text)


def _contains_normalized(query_norm: str, value: object) -> bool:
    """정규화된 질의에 대상 텍스트가 포함되는지 확인한다."""
    candidate = _normalize_text(value)
    return len(candidate) >= MIN_MATCH_LEN and candidate in query_norm


def _metadata_matches_query(query_norm: str, result) -> bool:
    """청크의 메타데이터(기관명·사업명·파일명)가 질의와 매칭되는지 확인한다."""
    metadata = result.chunk.metadata
    # 기관명 후보: resolved_agency, original_agency
    # 일반 발주 기관명은 너무 넓게 매칭되어 점수 왜곡이 커서 제외한다.
    agency_candidates = (
        metadata.get("resolved_agency", ""),
        metadata.get("original_agency", ""),
    )
    # 사업명 후보: 사업명, 파일명, doc_id
    project_candidates = (
        metadata.get("사업명", ""),
        metadata.get("파일명", ""),
        result.chunk.doc_id,
    )
    return any(_contains_normalized(query_norm, value) for value in (*agency_candidates, *project_candidates))


def build_reranker_text(result) -> str:
    """Cross-Encoder 입력용 메타 포함 텍스트를 생성한다.

    Args:
        result: RetrievedChunk 객체.

    Returns:
        발주기관/사업명이 있으면 본문 앞에 붙인 문자열, 없으면 본문만 반환.
    """
    agency = str(result.chunk.metadata.get("발주 기관", "") or "").strip()
    project = str(result.chunk.metadata.get("사업명", "") or "").strip()

    prefix_parts: list[str] = []
    if agency:
        prefix_parts.append(f"발주기관: {agency}")
    if project:
        prefix_parts.append(f"사업명: {project}")

    prefix = " | ".join(prefix_parts)
    if prefix:
        return f"[{prefix}]\n{result.chunk.text}"
    return result.chunk.text


def cross_encoder_rerank(reranker, query: str, results: list, top_k: int) -> list:
    """Cross-Encoder 모델로 질문-청크 쌍의 관련성을 판단하여 재정렬한다.

    Args:
        reranker: Cross-Encoder 모델 인스턴스.
        query: 사용자 질의 문자열.
        results: 벡터 검색으로 가져온 후보 청크 리스트.
        top_k: 최종 반환할 청크 수.

    Returns:
        관련성 높은 순으로 정렬된 상위 top_k개 RetrievedChunk 리스트.
        reranker가 None이면 입력 그대로 반환.
    """
    if not reranker or not results:
        return results

    pairs = [[query, build_reranker_text(r)] for r in results]
    scores = reranker.predict(pairs)

    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    reranked = []
    for rank, (result, ce_score) in enumerate(scored[:top_k], start=1):
        result.rerank_score = float(ce_score)
        result.rank = rank
        reranked.append(result)

    return reranked


def _assign_ranks(results: list) -> list:
    """결과 리스트에 1부터 순서대로 rank를 부여한다."""
    for index, result in enumerate(results, start=1):
        result.rank = index
    return results


def rerank_with_boost(
    results: list,
    query: str,
    section_hint: str | None,
    boost_config: dict | None = None,
) -> list:
    """섹션/테이블 부스팅 기반 리랭킹.

    Args:
        results: 청크 리스트.
        query: 사용자 질의 문자열.
        section_hint: 질문에서 추출된 섹션 힌트.
        boost_config: 부스팅 가중치 설정. None이면 기본값 사용.

    Returns:
        부스팅 적용 후 재정렬된 리스트.
    """
    if not results:
        return results

    cfg = boost_config or {}
    section_weight = cfg.get("section", 0.12)
    table_weight = cfg.get("table", 0.08)
    max_total = cfg.get("max_total", 0.15)

    query_norm = _normalize_text(query)
    table_boost = should_boost_tables(query)
    metadata_boost_enabled = not is_comparison_query(query)
    if not section_hint and not table_boost and not any(
        metadata_boost_enabled and _metadata_matches_query(query_norm, result) for result in results
    ):
        return _assign_ranks(results)

    def boosted_score(result) -> float:
        bonus = 0.0
        if section_hint and section_hint in result.chunk.section:
            bonus += section_weight
        if table_boost and result.chunk.content_type == "table":
            bonus += table_weight
        if metadata_boost_enabled and _metadata_matches_query(query_norm, result):
            bonus += METADATA_BOOST
        bonus = min(bonus, max_total)
        return result.score + bonus

    ordered = sorted(
        enumerate(results),
        key=lambda item: boosted_score(item[1]),
        reverse=True,
    )
    reranked = []
    for index, (_, result) in enumerate(ordered, start=1):
        result.rank = index
        reranked.append(result)
    return reranked
