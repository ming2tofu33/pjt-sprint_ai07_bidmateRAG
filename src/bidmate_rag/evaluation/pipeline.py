"""Single source of truth for evaluation execution.

This module owns the full evaluation pipeline (benchmark execution + retrieval
metrics + LLM judge + persistence + run metadata) so that both the CLI
(``cli/eval.py``) and the Streamlit UI (``app/api/routes.py``) call exactly
the same code path. Any new metric, cost, or report field should be added
here — not in CLI or UI layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from bidmate_rag.config.settings import RuntimeConfig
from bidmate_rag.evaluation.judge import LLMJudge
from bidmate_rag.evaluation.judge_v2 import LLMJudgeV2
from bidmate_rag.evaluation.metrics import calc_hit_rate, calc_map, calc_mrr, calc_ndcg
from bidmate_rag.evaluation.runner import (
    BenchmarkRunner,
    ProgressCallback,
    persist_benchmark_summary,
    persist_run_results,
)
from bidmate_rag.pipelines.runtime import collection_name_for_config
from bidmate_rag.schema import BenchmarkRunResult, EvalSample, GenerationResult
from bidmate_rag.tracking.git_info import capture_git_info


@dataclass
class EvaluationArtifacts:
    """All outputs produced by ``execute_evaluation`` for one run."""

    run_id: str
    benchmark: BenchmarkRunResult
    run_path: Path
    summary_path: Path
    meta_path: Path
    judge_total_cost_usd: float = 0.0
    judge_total_tokens: int = 0
    judge_skipped: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


def execute_evaluation(
    samples: list[EvalSample],
    *,
    pipeline,
    runtime: RuntimeConfig,
    embedder,
    eval_path: str,
    config_paths: dict[str, str | None] | None = None,
    runs_dir: str | Path = "artifacts/logs/runs",
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
    run_id: str | None = None,
    skip_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    judge_v2: bool = False,
    progress_callback: ProgressCallback | None = None,
    top_k: int | None = None,
) -> EvaluationArtifacts:
    """Run an evaluation end-to-end and write all artifacts to disk.

    Args:
        samples: Already-loaded eval samples (caller handles filtering/limit).
        pipeline: A built ``RAGChatPipeline`` instance.
        runtime: The composed ``RuntimeConfig`` (used for metadata snapshot).
        embedder: The embedding provider used by the pipeline (for metadata).
        eval_path: Path to the eval CSV (recorded in meta.json).
        config_paths: Optional dict with keys ``base``/``provider``/``experiment``
            mapping to the config file paths used to build the pipeline. Stored
            verbatim in meta.json so reports can link back.
        runs_dir: Directory where ``{run_id}.jsonl`` and ``{run_id}.meta.json``
            are written.
        benchmarks_dir: Directory where ``{experiment_name}.parquet`` is written.
        run_id: Optional pre-generated run id. Defaults to ``bench-XXXXXXXX``.
        skip_judge: If True, skip the LLM judge entirely (faster, cheaper).
        judge_model: LLM used by the judge when ``skip_judge`` is False.
        progress_callback: ``(done, total, sample) -> None`` invoked after
            each sample is answered (before judge runs).

    Returns:
        ``EvaluationArtifacts`` with run id, benchmark result, file paths, and
        aggregated judge cost/tokens.
    """
    runs_path = Path(runs_dir)
    benchmarks_path = Path(benchmarks_dir)
    resolved_run_id = run_id or f"bench-{uuid4().hex[:8]}"

    meta_path = _write_run_meta(
        runs_dir=runs_path,
        run_id=resolved_run_id,
        experiment_name=runtime.experiment.name,
        runtime=runtime,
        collection_name=collection_name_for_config(runtime),
        eval_path=eval_path,
        config_paths=config_paths or {},
        judge_skipped=skip_judge,
    )

    # ExperimentConfig.retrieval_top_k가 비어있으면 ProjectConfig 기본값 5.
    top_k = top_k or runtime.experiment.retrieval_top_k or runtime.project.default_retrieval_top_k or 5

    def answer_fn(sample: EvalSample) -> GenerationResult:
        # 평가셋의 metadata_filter / history를 retrieval에 실제로 적용
        # (이전엔 dataset.py가 sample.metadata에 저장만 하고 무시되던 상태)
        sample_meta = sample.metadata or {}
        return pipeline.answer(
            sample.question,
            top_k=top_k,
            chat_history=sample_meta.get("history") or None,
            metadata_filter=sample_meta.get("metadata_filter") or None,
            question_id=sample.question_id,
            scenario=runtime.provider.scenario or runtime.provider.provider,
            run_id=resolved_run_id,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
        )

    benchmark = BenchmarkRunner(answer_fn).run(
        experiment_name=runtime.experiment.name,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        provider_label=f"{runtime.provider.provider}:{runtime.provider.model}",
        samples=samples,
        progress_callback=progress_callback,
    )

    _aggregate_retrieval_metrics(samples, benchmark)

    judge_cost = 0.0
    judge_tokens = 0
    if not skip_judge:
        judge_cost, judge_tokens = _run_judge(samples, benchmark, judge_model, judge_v2=judge_v2)
        _update_run_meta(
            meta_path,
            judge_total_cost_usd=judge_cost,
            judge_total_tokens=judge_tokens,
            judge_mode="v2" if judge_v2 else "v1",
        )

    run_path = persist_run_results(benchmark.results, runs_dir=runs_path, run_id=resolved_run_id)
    summary_path = persist_benchmark_summary(
        [benchmark.to_summary_record()],
        benchmarks_dir=benchmarks_path,
        experiment_name=runtime.experiment.name,
    )

    return EvaluationArtifacts(
        run_id=resolved_run_id,
        benchmark=benchmark,
        run_path=run_path,
        summary_path=summary_path,
        meta_path=meta_path,
        judge_total_cost_usd=judge_cost,
        judge_total_tokens=judge_tokens,
        judge_skipped=skip_judge,
        metrics=dict(benchmark.metrics),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _aggregate_retrieval_metrics(samples: list[EvalSample], benchmark: BenchmarkRunResult) -> None:
    """Compute Hit Rate@5 / MRR / nDCG@5 / MAP@5 and merge into ``benchmark.metrics``."""
    totals = {"hit_rate@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "map@5": 0.0}
    scored = 0
    for sample, result in zip(samples, benchmark.results, strict=False):
        # Eval CSVs put 파일명 in `ground_truth_docs`, which dataset.py maps to
        # `expected_doc_titles`. Fall back so retrieval metrics actually run.
        expected = sample.expected_doc_ids or sample.expected_doc_titles
        if not expected:
            continue
        hit = calc_hit_rate(result.retrieved_chunks, expected, k=5)
        mrr = calc_mrr(result.retrieved_chunks, expected)
        ndcg = calc_ndcg(result.retrieved_chunks, expected, k=5)
        map_score = calc_map(result.retrieved_chunks, expected, k=5)
        if hit is not None:
            totals["hit_rate@5"] += hit
            totals["mrr"] += mrr or 0.0
            totals["ndcg@5"] += ndcg or 0.0
            totals["map@5"] += map_score or 0.0
            scored += 1
    if scored:
        benchmark.metrics.update({key: round(value / scored, 4) for key, value in totals.items()})


def _run_judge(
    samples: list[EvalSample],
    benchmark: BenchmarkRunResult,
    judge_model: str,
    *,
    judge_v2: bool = False,
) -> tuple[float, int]:
    """Run LLM judge on each sample, mutate result.judge_scores, return cost/tokens."""
    judge = LLMJudgeV2(model=judge_model) if judge_v2 else LLMJudge(model=judge_model)
    totals = {key: 0.0 for key in judge.METRIC_KEYS}
    judged = 0
    for sample, result in zip(samples, benchmark.results, strict=False):
        contexts = [chunk.chunk.text for chunk in result.retrieved_chunks]
        scores = judge.evaluate(
            question=sample.question,
            answer=result.answer,
            contexts=contexts,
            expected_answer=sample.metadata.get("ground_truth_answer"),
        )
        result.judge_scores = scores.to_dict()
        if scores.error:
            continue
        for key in judge.METRIC_KEYS:
            totals[key] += getattr(scores, key)
        judged += 1
    if judged:
        benchmark.metrics.update({key: round(value / judged, 4) for key, value in totals.items()})
    return round(judge.cumulative_cost_usd, 6), judge.cumulative_tokens


def _write_run_meta(
    runs_dir: Path,
    run_id: str,
    experiment_name: str,
    runtime: RuntimeConfig,
    collection_name: str,
    eval_path: str,
    config_paths: dict[str, str | None],
    judge_skipped: bool = False,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(UTC)
    meta = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_kst": now_utc.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "git": capture_git_info(),
        "configs": {k: v for k, v in config_paths.items() if v},
        "notes_path": runtime.experiment.notes_path,
        "config_snapshot": runtime.model_dump(),
        "eval_path": eval_path,
        "collection_name": collection_name,
        "judge_skipped": judge_skipped,
        "judge_total_cost_usd": 0.0,
        "judge_total_tokens": 0,
    }
    out_path = runs_dir / f"{run_id}.meta.json"
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _update_run_meta(meta_path: Path, **fields: Any) -> None:
    if not meta_path.exists():
        return
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data.update(fields)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
