"""Evaluation metrics."""

from __future__ import annotations

import math

from bidmate_rag.schema import EvalSample, GenerationResult, RetrievedChunk


def _rewrite_cost_usd(result: GenerationResult) -> float:
    """Return the rewrite cost recorded for a generation result."""
    debug = result.debug or {}
    if "rewrite_cost_usd" in debug:
        return float(debug.get("rewrite_cost_usd", 0.0) or 0.0)
    total_cost = debug.get("total_cost_usd")
    if total_cost is None:
        return 0.0
    return max(round(float(total_cost or 0.0) - float(result.cost_usd or 0.0), 6), 0.0)


def _match_expected(chunk: RetrievedChunk, expected_doc_ids: list[str]) -> bool:
    """검색된 청크가 정답 문서에 해당하는지 확인

    Args:
        chunk: 검색된 청크.
        expected_doc_ids: 정답 문서 식별자 리스트 (doc_id, 사업명, 파일명 중 하나).

    The eval CSVs (`data/eval/eval_v*/eval_batch_*.csv`) populate `ground_truth_docs`
    with `파일명` strings, so this function must compare against `파일명` to
    make Hit Rate / MRR / nDCG metrics meaningful.
    부분 매칭 추가: 사업명이 잘려있어도 파일명에 포함되면 매칭 성공
    """
    사업명 = chunk.chunk.metadata.get("사업명") or ""
    파일명 = chunk.chunk.metadata.get("파일명") or ""
    doc_id = chunk.chunk.doc_id or ""

    for exp in expected_doc_ids:
        if not exp:
            continue
        if (
            doc_id == exp
            or 사업명 == exp
            or 파일명 == exp
            or (사업명 and 사업명 in exp)  # 사업명이 기대값에 포함
            or (exp in 파일명)             # 기대값이 파일명에 포함
        ):
            return True
    return False

  


def calc_hit_rate(
    chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5
) -> float | None:
    """상위 k개 결과 중 정답 문서가 하나라도 포함되었는지 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 확인할지.

    Returns:
        정답 포함 시 1.0, 미포함 시 0.0, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 상위 k개 중 정답이 하나라도 있으면 1.0
    return 1.0 if any(_match_expected(chunk, expected_doc_ids) for chunk in chunks[:k]) else 0.0


def calc_mrr(chunks: list[RetrievedChunk], expected_doc_ids: list[str]) -> float | None:
    """정답 문서가 처음 등장하는 순위의 역수(Mean Reciprocal Rank)를 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.

    Returns:
        1/순위 값 (1위=1.0, 2위=0.5, ...), 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 첫 번째 정답이 나타나는 순위의 역수를 반환
    for index, chunk in enumerate(chunks, start=1):
        if _match_expected(chunk, expected_doc_ids):
            return 1.0 / index
    return 0.0


def calc_ndcg(
    chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5
) -> float | None:
    """상위 k개 결과의 순위 품질을 nDCG(normalized Discounted Cumulative Gain)로 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 평가할지.

    Returns:
        0~1 사이 nDCG 점수, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 각 청크의 관련도: 정답이면 2, 아니면 0
    relevances = [2 if _match_expected(chunk, expected_doc_ids) else 0 for chunk in chunks[:k]]
    # DCG: 순위가 낮을수록 log로 할인
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))
    # iDCG: 이상적인 순서(정답이 모두 상위에 위치)로 계산
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def calc_map(chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5) -> float | None:
    """상위 k개 결과의 Mean Average Precision을 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 평가할지.

    Returns:
        0~1 사이 MAP 점수, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 정답 수 (중복 문서 제거를 위해 set 사용하지 않음 — 평가셋 기준 그대로)
    num_relevant = len(expected_doc_ids)
    hits = 0
    precision_sum = 0.0
    # 이미 매칭된 doc을 추적하여 같은 문서의 청크 중복 카운트 방지
    seen_docs: set[str] = set()
    for index, chunk in enumerate(chunks[:k], start=1):
        if _match_expected(chunk, expected_doc_ids):
            # 같은 문서의 다른 청크가 이미 매칭되었으면 스킵
            doc_key = chunk.chunk.doc_id
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
            hits += 1
            # 해당 순위에서의 precision을 누적
            precision_sum += hits / index
    return precision_sum / num_relevant if num_relevant else 0.0


def summarize_generation_results(results: list[GenerationResult]) -> dict[str, float]:
    """생성 결과들의 평균 지연시간과 총 비용을 요약

    Args:
        results: GenerationResult 리스트.

    Returns:
        avg_latency_ms와 total_cost_usd를 포함하는 딕셔너리.
    """
    if not results:
        return {"avg_latency_ms": 0.0, "total_cost_usd": 0.0}
    return {
        # 전체 결과의 평균 응답 시간 (밀리초)
        "avg_latency_ms": round(sum(result.latency_ms for result in results) / len(results), 3),
        # 전체 결과의 누적 API 비용 (USD)
        "total_cost_usd": round(sum(result.cost_usd for result in results), 6),
    }


def summarize_run_operations(
    results: list[GenerationResult],
    *,
    judge_total_cost_usd: float = 0.0,
) -> dict[str, float]:
    """평가 실행의 비용/토큰/지연 운영 지표를 요약한다.

    Args:
        results: GenerationResult 리스트.
        judge_total_cost_usd: Judge가 사용한 누적 비용.

    Returns:
        생성 비용, judge 비용, 총 비용, 토큰, 평균 지연을 담은 딕셔너리.
        재작성 토큰(`rewrite_*`)이 없으면 0으로 채운다.
    """
    if not results:
        return {
            "generation_cost_usd": 0.0,
            "rewrite_cost_usd": 0.0,
            "judge_cost_usd": round(float(judge_total_cost_usd or 0.0), 6),
            "total_cost_usd": round(float(judge_total_cost_usd or 0.0), 6),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "rewrite_prompt_tokens": 0,
            "rewrite_completion_tokens": 0,
            "rewrite_total_tokens": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
        }

    generation_cost = round(sum(float(result.cost_usd or 0.0) for result in results), 6)
    rewrite_cost = round(sum(_rewrite_cost_usd(result) for result in results), 6)
    prompt_tokens = sum(int((result.token_usage or {}).get("prompt", 0) or 0) for result in results)
    completion_tokens = sum(
        int((result.token_usage or {}).get("completion", 0) or 0) for result in results
    )
    rewrite_prompt_tokens = sum(
        int((result.token_usage or {}).get("rewrite_prompt", 0) or 0) for result in results
    )
    rewrite_completion_tokens = sum(
        int((result.token_usage or {}).get("rewrite_completion", 0) or 0) for result in results
    )
    rewrite_total_tokens = sum(
        int((result.token_usage or {}).get("rewrite_total", 0) or 0) for result in results
    )
    generation_total_tokens = sum(
        int((result.token_usage or {}).get("total", 0) or 0) for result in results
    )
    avg_latency_ms = round(sum(float(result.latency_ms or 0.0) for result in results) / len(results), 3)
    judge_cost = round(float(judge_total_cost_usd or 0.0), 6)

    return {
        "generation_cost_usd": generation_cost,
        "rewrite_cost_usd": rewrite_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": round(generation_cost + rewrite_cost + judge_cost, 6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "rewrite_prompt_tokens": rewrite_prompt_tokens,
        "rewrite_completion_tokens": rewrite_completion_tokens,
        "rewrite_total_tokens": rewrite_total_tokens,
        "total_tokens": generation_total_tokens + rewrite_total_tokens,
        "avg_latency_ms": avg_latency_ms,
    }


def aggregate_retrieval_metrics_by_type(
    samples: list[EvalSample],
    results: list[GenerationResult],
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Type별(A/B/C/D/E)로 Hit Rate / MRR / nDCG / MAP을 분리 집계한다.

    멀티턴(Type C)이 단일턴(A/B) 대비 얼마나 빠지는지 보려는 용도.
    expected_doc_titles/expected_doc_ids가 비어 있으면 해당 샘플은 집계에서 제외
    (기존 `_aggregate_retrieval_metrics`와 동일 정책). type 메타데이터가 없으면
    '(unknown)' 버킷으로 모은다.

    Returns:
        {"A": {"n": 16, "hit_rate@5": 0.81, "mrr": 0.72, "ndcg@5": 0.85, "map@5": 0.78},
         "B": {...}, "C": {...}, ...}
        — 키 순서는 타입 알파벳 순 정렬.
    """
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for sample, result in zip(samples, results, strict=False):
        expected = sample.expected_doc_ids or sample.expected_doc_titles
        if not expected:
            continue
        type_key = str(sample.metadata.get("type") or "").strip() or "(unknown)"

        hit = calc_hit_rate(result.retrieved_chunks, expected, k=k)
        if hit is None:
            continue
        mrr = calc_mrr(result.retrieved_chunks, expected) or 0.0
        ndcg = calc_ndcg(result.retrieved_chunks, expected, k=k) or 0.0
        map_score = calc_map(result.retrieved_chunks, expected, k=k) or 0.0

        bucket = totals.setdefault(
            type_key,
            {f"hit_rate@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0, f"map@{k}": 0.0},
        )
        bucket[f"hit_rate@{k}"] += hit
        bucket["mrr"] += mrr
        bucket[f"ndcg@{k}"] += ndcg
        bucket[f"map@{k}"] += map_score
        counts[type_key] = counts.get(type_key, 0) + 1

    out: dict[str, dict[str, float]] = {}
    for type_key in sorted(totals.keys()):
        n = counts[type_key]
        bucket = totals[type_key]
        out[type_key] = {
            "n": n,
            f"hit_rate@{k}": round(bucket[f"hit_rate@{k}"] / n, 4),
            "mrr": round(bucket["mrr"] / n, 4),
            f"ndcg@{k}": round(bucket[f"ndcg@{k}"] / n, 4),
            f"map@{k}": round(bucket[f"map@{k}"] / n, 4),
        }
    return out
