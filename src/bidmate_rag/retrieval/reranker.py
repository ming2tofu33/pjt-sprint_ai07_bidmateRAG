"""Reranking logic for RAG retrieval.

Cross-Encoder 리랭킹과 섹션/테이블 부스팅을 담당한다.
retriever.py의 오케스트레이션과 분리하여 독립적으로 테스트·교체 가능.
"""

from __future__ import annotations

import re

from bidmate_rag.retrieval.filters import is_comparison_query, should_boost_tables

# 부스팅 기본 상수 (boost_config 미지정 시 폴백용)
SECTION_BOOST = 0.20
TABLE_BOOST = 0.08
METADATA_BOOST = 0.12
MAX_TOTAL_BOOST = 0.25
MIN_MATCH_LEN = 3  # 정규화된 텍스트 최소 매칭 길이
SECTION_HINT_MIN_MATCH_LEN = 2


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


def _section_hint_matches_result(section_hint: str | None, result) -> bool:
    """section_hint가 섹션명뿐 아니라 본문/메타 텍스트에도 드러나는지 확인한다."""
    hint_norm = _normalize_text(section_hint)
    if len(hint_norm) < SECTION_HINT_MIN_MATCH_LEN:
        return False

    candidates = (
        result.chunk.section,
        result.chunk.text,
        getattr(result.chunk, "text_with_meta", ""),
    )
    return any(hint_norm in _normalize_text(value) for value in candidates)


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


def cross_encoder_rerank(
    reranker, query: str, results: list, top_k: int | None = None
) -> list:
    """Cross-Encoder 모델로 질문-청크 쌍의 관련성을 판단하여 재정렬한다.

    Args:
        reranker: Cross-Encoder 모델 인스턴스.
        query: 사용자 질의 문자열.
        results: 벡터 검색으로 가져온 후보 청크 리스트.
        top_k: 유지할 청크 수. None이면 풀 전체를 유지해 downstream boost가
            전체 후보에 대해 재정렬하도록 한다.

    Returns:
        관련성 높은 순으로 정렬된 RetrievedChunk 리스트.
        reranker가 None이면 입력 그대로 반환.

    Note:
        CE 점수를 ``result.score``와 ``result.rerank_score`` 양쪽에 기록한다.
        이후 ``rerank_with_boost``가 CE 점수에 섹션/테이블 가산점을 더해
        일관된 기준으로 재정렬하기 위함이다.
    """
    if not reranker or not results:
        return results

    pairs = [[query, build_reranker_text(r)] for r in results]
    scores = reranker.predict(pairs)

    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    limit = len(scored) if top_k is None else top_k
    reranked = []
    for rank, (result, ce_score) in enumerate(scored[:limit], start=1):
        ce_value = float(ce_score)
        result.rerank_score = ce_value
        result.score = ce_value
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
    section_weight = cfg.get("section", SECTION_BOOST)
    table_weight = cfg.get("table", TABLE_BOOST)
    metadata_weight = cfg.get("metadata", METADATA_BOOST)
    max_total = cfg.get("max_total", MAX_TOTAL_BOOST)

    query_norm = _normalize_text(query)
    table_boost = should_boost_tables(query)
    metadata_boost_enabled = not is_comparison_query(query)
    if not section_hint and not table_boost and not any(
        metadata_boost_enabled and _metadata_matches_query(query_norm, result) for result in results
    ):
        return _assign_ranks(results)

    def boosted_score(result) -> float:
        bonus = 0.0
        if _section_hint_matches_result(section_hint, result):
            bonus += section_weight
        if table_boost and result.chunk.content_type == "table":
            bonus += table_weight
        if metadata_boost_enabled and _metadata_matches_query(query_norm, result):
            bonus += metadata_weight
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
