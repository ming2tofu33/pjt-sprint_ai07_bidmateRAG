"""Tests for evaluation/metrics.py."""

from __future__ import annotations

from bidmate_rag.evaluation.metrics import (
    aggregate_retrieval_metrics_by_type,
    calc_hit_rate,
    calc_map,
    calc_mrr,
    calc_ndcg,
    summarize_run_operations,
)
from bidmate_rag.schema import Chunk, EvalSample, GenerationResult, RetrievedChunk


def _make_chunk(
    *,
    doc_id: str = "DOC-001",
    사업명: str = "샘플 사업",
    파일명: str = "기관명_샘플 사업.hwp",
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=0.9,
        chunk=Chunk(
            chunk_id=f"chunk-{rank}",
            doc_id=doc_id,
            text="본문",
            text_with_meta="본문",
            char_count=2,
            section="",
            content_type="text",
            chunk_index=0,
            metadata={"사업명": 사업명, "파일명": 파일명},
        ),
    )


def test_match_by_doc_id():
    chunks = [_make_chunk(doc_id="DOC-001")]
    assert calc_hit_rate(chunks, ["DOC-001"], k=5) == 1.0


def test_match_by_사업명():
    chunks = [_make_chunk(사업명="공공포털 구축")]
    assert calc_hit_rate(chunks, ["공공포털 구축"], k=5) == 1.0


def test_match_by_파일명():
    """Eval CSV ground_truth_docs는 파일명 형식이라 이 매칭이 가장 중요."""
    chunks = [_make_chunk(파일명="한국가스공사_차세대 ERP.hwp")]
    assert calc_hit_rate(chunks, ["한국가스공사_차세대 ERP.hwp"], k=5) == 1.0


def test_no_match_returns_zero():
    chunks = [_make_chunk(doc_id="DOC-001", 사업명="A", 파일명="A.hwp")]
    assert calc_hit_rate(chunks, ["완전히 다른 파일.hwp"], k=5) == 0.0


def test_empty_expected_returns_none():
    chunks = [_make_chunk()]
    assert calc_hit_rate(chunks, [], k=5) is None
    assert calc_mrr(chunks, []) is None
    assert calc_ndcg(chunks, [], k=5) is None
    assert calc_map(chunks, [], k=5) is None


def test_mrr_uses_파일명_at_rank_3():
    chunks = [
        _make_chunk(doc_id="A", 사업명="A", 파일명="a.hwp", rank=1),
        _make_chunk(doc_id="B", 사업명="B", 파일명="b.hwp", rank=2),
        _make_chunk(doc_id="C", 사업명="C", 파일명="target.hwp", rank=3),
    ]
    # rank 3에서 매칭 → MRR = 1/3
    assert calc_mrr(chunks, ["target.hwp"]) == 1 / 3


def test_ndcg_파일명_top_position():
    chunks = [
        _make_chunk(파일명="hit.hwp", rank=1),
        _make_chunk(파일명="miss1.hwp", rank=2),
        _make_chunk(파일명="miss2.hwp", rank=3),
    ]
    # 1위에서 hit → DCG = 2/log2(2) = 2.0, iDCG = 2.0 → nDCG = 1.0
    assert calc_ndcg(chunks, ["hit.hwp"], k=5) == 1.0


def test_map_single_doc():
    """정답 문서 1개 → MAP은 MRR과 동일한 값."""
    chunks = [
        _make_chunk(파일명="miss.hwp", rank=1),
        _make_chunk(파일명="hit.hwp", rank=2),
    ]
    # 2위에서 매칭 → precision=1/2 → AP=0.5/1=0.5
    assert calc_map(chunks, ["hit.hwp"], k=5) == 0.5


def test_map_multi_doc():
    """정답 문서 2개 → 두 문서를 모두 상위에서 찾았는지 평가."""
    chunks = [
        _make_chunk(doc_id="A", 파일명="hit1.hwp", rank=1),
        _make_chunk(doc_id="B", 파일명="miss.hwp", rank=2),
        _make_chunk(doc_id="C", 파일명="hit2.hwp", rank=3),
    ]
    # 1위: precision=1/1, 3위: precision=2/3 → AP=(1+2/3)/2=0.8333...
    result = calc_map(chunks, ["hit1.hwp", "hit2.hwp"], k=5)
    assert round(result, 4) == 0.8333


def test_map_duplicate_chunk_same_doc():
    """같은 문서의 청크가 여러 개 검색되어도 중복 카운트하지 않음."""
    chunks = [
        _make_chunk(doc_id="A", 파일명="hit.hwp", rank=1),
        _make_chunk(doc_id="A", 파일명="hit.hwp", rank=2),
    ]
    # 같은 doc_id → 1위에서만 카운트 → AP=1/1=1.0
    assert calc_map(chunks, ["hit.hwp"], k=5) == 1.0


def test_hit_rate_outside_topk():
    chunks = [
        _make_chunk(파일명="miss1.hwp", rank=1),
        _make_chunk(파일명="miss2.hwp", rank=2),
        _make_chunk(파일명="miss3.hwp", rank=3),
        _make_chunk(파일명="miss4.hwp", rank=4),
        _make_chunk(파일명="miss5.hwp", rank=5),
        _make_chunk(파일명="hit.hwp", rank=6),  # k=5 밖
    ]
    assert calc_hit_rate(chunks, ["hit.hwp"], k=5) == 0.0


def test_summarize_run_operations_aggregates_cost_tokens_and_latency() -> None:
    results = [
        GenerationResult(
            question_id="q1",
            question="질문1",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="openai",
            llm_model="gpt-5-mini",
            answer="답변1",
            latency_ms=1000.0,
            token_usage={"prompt": 100, "completion": 20, "total": 120},
            cost_usd=0.001,
        ),
        GenerationResult(
            question_id="q2",
            question="질문2",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="openai",
            llm_model="gpt-5-mini",
            answer="답변2",
            latency_ms=3000.0,
            token_usage={
                "prompt": 200,
                "completion": 50,
                "total": 250,
                "rewrite_prompt": 40,
                "rewrite_completion": 10,
                "rewrite_total": 50,
            },
            cost_usd=0.002,
            debug={"rewrite_cost_usd": 0.0003},
        ),
    ]

    summary = summarize_run_operations(results, judge_total_cost_usd=0.0008)

    assert summary == {
        "generation_cost_usd": 0.003,
        "rewrite_cost_usd": 0.0003,
        "judge_cost_usd": 0.0008,
        "total_cost_usd": 0.0041,
        "prompt_tokens": 300,
        "completion_tokens": 70,
        "rewrite_prompt_tokens": 40,
        "rewrite_completion_tokens": 10,
        "rewrite_total_tokens": 50,
        "total_tokens": 420,
        "avg_latency_ms": 2000.0,
    }


def test_summarize_run_operations_returns_zeroed_defaults_for_empty_results() -> None:
    summary = summarize_run_operations([], judge_total_cost_usd=0.0008)

    assert summary["generation_cost_usd"] == 0.0
    assert summary["rewrite_cost_usd"] == 0.0
    assert summary["judge_cost_usd"] == 0.0008
    assert summary["total_cost_usd"] == 0.0008
    assert summary["total_tokens"] == 0
    assert summary["avg_latency_ms"] == 0.0



def _make_sample(
    *, question_id: str, type_value: str, expected: list[str]
) -> EvalSample:
    return EvalSample(
        question_id=question_id,
        question="q",
        expected_doc_titles=expected,
        metadata={"type": type_value},
    )


def _make_chunk_with_title(title: str, *, chunk_id: str = "c-1") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}-doc",
        text="t",
        text_with_meta="t",
        char_count=1,
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={"파일명": title},
    )
    return RetrievedChunk(rank=1, score=0.9, chunk=chunk)


def _make_result(
    *, question_id: str, retrieved: list[RetrievedChunk]
) -> GenerationResult:
    return GenerationResult(
        question_id=question_id,
        question="q",
        scenario="scenario_b",
        run_id="run-1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        llm_provider="fake",
        llm_model="fake-model",
        answer="a",
        retrieved_chunk_ids=[c.chunk.chunk_id for c in retrieved],
        retrieved_doc_ids=[c.chunk.doc_id for c in retrieved],
        retrieved_chunks=retrieved,
        latency_ms=1.0,
        token_usage={},
    )


def test_aggregate_retrieval_metrics_by_type_groups_by_metadata_type() -> None:
    """Type A(2 hits), Type C(1 miss) 샘플을 넣고 그룹별 집계 검증."""
    samples = [
        _make_sample(question_id="q1", type_value="A", expected=["doc1.hwp"]),
        _make_sample(question_id="q2", type_value="A", expected=["doc2.hwp"]),
        _make_sample(question_id="q3", type_value="C", expected=["doc3.hwp"]),
    ]
    results = [
        _make_result(question_id="q1", retrieved=[_make_chunk_with_title("doc1.hwp")]),
        _make_result(question_id="q2", retrieved=[_make_chunk_with_title("doc2.hwp")]),
        _make_result(question_id="q3", retrieved=[_make_chunk_with_title("other.hwp")]),
    ]

    breakdown = aggregate_retrieval_metrics_by_type(samples, results, k=5)

    assert breakdown["A"]["n"] == 2
    assert breakdown["A"]["hit_rate@5"] == 1.0
    assert breakdown["A"]["mrr"] == 1.0
    assert breakdown["C"]["n"] == 1
    assert breakdown["C"]["hit_rate@5"] == 0.0
    assert breakdown["C"]["mrr"] == 0.0


def test_aggregate_retrieval_metrics_by_type_handles_missing_type_and_expected() -> None:
    """type 누락 / expected 누락 샘플 처리 — 전자는 '(unknown)'에 들어감, 후자는 집계 제외."""
    samples = [
        _make_sample(question_id="q1", type_value="A", expected=["doc1.hwp"]),
        EvalSample(
            question_id="q2",
            question="q",
            expected_doc_titles=["doc2.hwp"],
            metadata={},  # type 누락
        ),
        EvalSample(
            question_id="q3",
            question="q",
            expected_doc_titles=[],  # expected 없음
            metadata={"type": "C"},
        ),
    ]
    results = [
        _make_result(question_id="q1", retrieved=[_make_chunk_with_title("doc1.hwp")]),
        _make_result(question_id="q2", retrieved=[_make_chunk_with_title("doc2.hwp")]),
        _make_result(question_id="q3", retrieved=[_make_chunk_with_title("anything.hwp")]),
    ]

    breakdown = aggregate_retrieval_metrics_by_type(samples, results, k=5)

    assert breakdown["A"]["n"] == 1
    assert breakdown["A"]["hit_rate@5"] == 1.0
    assert breakdown["(unknown)"]["n"] == 1
    assert breakdown["(unknown)"]["hit_rate@5"] == 1.0
    # Type C는 expected 없어 집계 제외 → 키 자체가 없음
    assert "C" not in breakdown
