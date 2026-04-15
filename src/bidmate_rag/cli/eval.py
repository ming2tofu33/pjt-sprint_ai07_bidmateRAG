"""RAG 평가 벤치마크 CLI 진입점.

CLI 인자를 파싱하고 런타임 파이프라인을 조립한 뒤,
evaluation.pipeline.execute_evaluation에 위임한다.

사용 예시::

    uv run bidmate-eval \\
        --evaluation-path data/eval/eval_v1/eval_batch_01.csv \\
        --provider-config configs/providers/openai_gpt5mini.yaml \\
        --limit 5 --filter-type A,B
"""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from bidmate_rag.evaluation.dataset import load_eval_samples
from bidmate_rag.evaluation.pipeline import EvaluationArtifacts, execute_evaluation
from bidmate_rag.evaluation.schema_validator import (
    render_validation_report,
    validate_eval_samples,
)
from bidmate_rag.pipelines.runtime import _resolve_metadata_path, build_runtime_pipeline
from bidmate_rag.schema import EvalSample, GenerationResult


def _split_csv(value: str | None) -> list[str] | None:
    """쉼표로 구분된 CLI 인자를 리스트로 파싱한다.

    Args:
        value: 쉼표 구분 문자열 (예: "A,B") 또는 None.

    Returns:
        파싱된 문자열 리스트 또는 None.
    """
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_filters(
    samples: list[EvalSample],
    *,
    types: list[str] | None,
    difficulties: list[str] | None,
    limit: int | None,
) -> list[EvalSample]:
    """평가 샘플을 타입/난이도로 필터링하고 개수를 제한한다.

    Args:
        samples: 전체 평가 샘플 리스트.
        types: 유지할 질문 타입 리스트 (예: ["A", "B"]).
        difficulties: 유지할 난이도 리스트 (예: ["하", "중"]).
        limit: 최대 샘플 수 제한.

    Returns:
        필터링된 EvalSample 리스트.
    """
    filtered = samples
    # 질문 타입 필터 (A, B 등)
    if types:
        type_set = {t for t in types}
        filtered = [s for s in filtered if str(s.metadata.get("type", "")) in type_set]
    # 난이도 필터 (하, 중, 상 등)
    if difficulties:
        diff_set = {d for d in difficulties}
        filtered = [s for s in filtered if str(s.metadata.get("difficulty", "")) in diff_set]
    # 개수 제한
    if limit is not None and limit >= 0:
        filtered = filtered[:limit]
    return filtered


def _print_summary(
    samples: list[EvalSample],
    results: list[GenerationResult],
    overall_metrics: dict[str, float],
) -> None:
    """벤치마크 실행 결과를 콘솔에 출력한다.

    Args:
        samples: 평가에 사용된 샘플 리스트.
        results: 각 샘플에 대한 생성 결과 리스트.
        overall_metrics: 전체 검색 지표 (Hit Rate, MRR 등).
    """
    if not results:
        print("(no results)")
        return

    # 각 결과를 요약 행으로 변환
    rows = []
    for sample, result in zip(samples, results, strict=False):
        rows.append(
            {
                "id": result.question_id,
                "type": sample.metadata.get("type", ""),
                "difficulty": sample.metadata.get("difficulty", ""),
                "tokens": int(result.token_usage.get("total", 0) or 0),
                "latency_ms": round(result.latency_ms),
                "error": bool(result.error),
            }
        )
    df = pd.DataFrame(rows)

    # 전체 요약 출력
    print()
    print(f"=== Run summary ({len(df)} questions) ===")
    print(
        f"errors={int(df['error'].sum())}  "
        f"avg_tokens={df['tokens'].mean():.0f}  "
        f"avg_latency_ms={df['latency_ms'].mean():.0f}"
    )
    # 검색 지표 출력 (Hit Rate, MRR 등)
    if overall_metrics:
        metric_str = "  ".join(f"{k}={v}" for k, v in overall_metrics.items())
        print(f"retrieval: {metric_str}")

    # 질문 타입별 통계
    if df["type"].astype(bool).any():
        print("\n-- by type --")
        by_type = (
            df.groupby("type")
            .agg(
                n=("id", "count"),
                avg_tokens=("tokens", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                errors=("error", "sum"),
            )
            .round(0)
        )
        print(by_type.to_string())

    # 난이도별 통계
    if df["difficulty"].astype(bool).any():
        print("\n-- by difficulty --")
        by_diff = (
            df.groupby("difficulty")
            .agg(
                n=("id", "count"),
                avg_tokens=("tokens", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                errors=("error", "sum"),
            )
            .round(0)
        )
        print(by_diff.to_string())


def _print_artifacts(artifacts: EvaluationArtifacts) -> None:
    """평가 산출물 경로와 LLM 판정 비용을 출력한다.

    Args:
        artifacts: 평가 실행 산출물 (경로, 비용, 토큰 수 등).
    """
    print()
    print(f"run_id:    {artifacts.run_id}")
    print(f"run jsonl: {artifacts.run_path}")
    print(f"summary:   {artifacts.summary_path}")
    print(f"meta:      {artifacts.meta_path}")
    # LLM 판정을 실행했으면 비용/토큰 출력
    if not artifacts.judge_skipped:
        print(
            f"judge:     ${artifacts.judge_total_cost_usd:.4f} "
            f"({artifacts.judge_total_tokens} tokens)"
        )


def _build_progress_callback(enabled: bool):
    """Return a CLI-only progress callback when requested.

    The evaluation pipeline already supports progress callbacks for UIs. This
    helper keeps the CLI behavior opt-in so Streamlit and API paths remain
    unchanged.
    """
    if not enabled:
        return None

    def _progress(done: int, total: int, sample: EvalSample) -> None:
        preview = " ".join(str(sample.question).split())
        if len(preview) > 60:
            preview = f"{preview[:57]}..."
        print(f"[{done}/{total}] {sample.question_id} {preview}", flush=True)

    return _progress


def main() -> None:
    """CLI 인자를 파싱하고 평가 벤치마크를 실행한다."""
    load_dotenv()

    # CLI 인자 정의
    parser = argparse.ArgumentParser(
        prog="bidmate-eval",
        description="Run the BidMate RAG benchmark from the command line.",
    )
    parser.add_argument("--evaluation-path", required=True)  # 평가셋 파일 경로
    parser.add_argument("--provider-config", required=True)  # 프로바이더 설정 YAML
    parser.add_argument("--base-config", default="configs/base.yaml")  # 기본 설정 YAML
    parser.add_argument("--experiment-config", default=None)  # 실험 설정 YAML (선택)
    parser.add_argument("--runs-dir", default="artifacts/logs/runs")  # 실행 로그 저장 경로
    parser.add_argument(
        "--benchmarks-dir", default="artifacts/logs/benchmarks"
    )  # 벤치마크 저장 경로
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions after other filters are applied.",
    )
    parser.add_argument(
        "--filter-type",
        default=None,
        help="Comma-separated question types to keep, e.g. 'A,B'.",
    )
    parser.add_argument(
        "--filter-difficulty",
        default=None,
        help="Comma-separated difficulties to keep, e.g. '하,중'.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override run_id (otherwise auto-generated as bench-XXXXXXXX).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-judge evaluation (faithfulness etc.).",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM model used by the judge (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--judge-v2",
        action="store_true",
        help="Use evidence-first judge v2 (keeps v1 as default).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="평가셋 스키마 검증에서 경고가 1건이라도 발견되면 평가 중단.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="평가셋 스키마 검증 자체를 건너뜀 (legacy 호환).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="평가 진행 상황을 질문 단위로 콘솔에 출력.",
    )
    args = parser.parse_args()

    # 1. 런타임 파이프라인 조립 (설정 → 프로바이더 → 리트리버 → LLM)
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )

    # 2. 평가셋 로딩 ("다중" 필터 → $in 변환을 위해 기관 목록 전달)
    agency_list = getattr(pipeline.retriever.metadata_store, "agency_list", [])
    all_samples = load_eval_samples(args.evaluation_path, agency_list=agency_list)

    # 3. 평가셋 스키마 검증 (metadata_filter 키/값, ground_truth_docs 존재 여부 등)
    if not args.no_validate:
        metadata_path = _resolve_metadata_path(runtime, None)
        report = validate_eval_samples(all_samples, cleaned_documents_path=metadata_path)
        print(render_validation_report(report))
        if not report.is_valid(strict=args.strict):
            raise SystemExit("❌ 평가셋 검증 실패. 문제를 수정하거나 --no-validate로 우회하세요.")

    # 4. 타입/난이도/개수 필터 적용
    samples = _apply_filters(
        all_samples,
        types=_split_csv(args.filter_type),
        difficulties=_split_csv(args.filter_difficulty),
        limit=args.limit,
    )
    print(
        f"Loaded {len(all_samples)} samples from {args.evaluation_path}; "
        f"running {len(samples)} after filters."
    )
    if not samples:
        raise SystemExit("No samples remain after filtering.")

    # 5. 평가 실행 (검색 → LLM 생성 → 지표 계산 → 산출물 저장)
    artifacts = execute_evaluation(
        samples,
        pipeline=pipeline,
        runtime=runtime,
        embedder=embedder,
        eval_path=args.evaluation_path,
        config_paths={
            "base": args.base_config,
            "provider": args.provider_config,
            "experiment": args.experiment_config,
        },
        runs_dir=args.runs_dir,
        benchmarks_dir=args.benchmarks_dir,
        run_id=args.run_id,
        skip_judge=args.skip_judge,
        judge_model=args.judge_model,
        progress_callback=_build_progress_callback(args.progress),
        judge_v2=args.judge_v2,
    )

    # 6. 결과 요약 및 산출물 경로 출력
    _print_summary(samples, artifacts.benchmark.results, artifacts.metrics)
    _print_artifacts(artifacts)


if __name__ == "__main__":
    main()
