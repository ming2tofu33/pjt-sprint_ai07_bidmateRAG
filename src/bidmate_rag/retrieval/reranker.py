"""Reranking logic for RAG retrieval.

Cross-Encoder 리랭킹과 섹션/테이블 부스팅을 담당한다.
retriever.py의 오케스트레이션과 분리하여 독립적으로 테스트·교체 가능.
"""

from __future__ import annotations

from bidmate_rag.retrieval.filters import should_boost_tables


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
    for rank, (result, score) in enumerate(scored[:top_k], start=1):
        result.score = float(score)
        result.rank = rank
        reranked.append(result)

    return reranked


def _assign_ranks(results: list) -> list:
    """결과 리스트에 1부터 순서대로 rank를 부여한다."""
    for index, result in enumerate(results, start=1):
        result.rank = index
    return results


def rerank_with_boost(results: list, query: str, section_hint: str | None) -> list:
    """섹션/테이블 부스팅 기반 리랭킹.

    Args:
        results: 청크 리스트.
        query: 사용자 질의 문자열.
        section_hint: 질문에서 추출된 섹션 힌트.

    Returns:
        부스팅 적용 후 재정렬된 리스트.
    """
    if not results:
        return results

    table_boost = should_boost_tables(query)
    if not section_hint and not table_boost:
        return _assign_ranks(results)

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
