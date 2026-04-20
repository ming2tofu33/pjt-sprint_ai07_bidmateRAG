"""Benchmark runner and result persistence helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from bidmate_rag.schema import BenchmarkRunResult, EvalSample, GenerationResult


def persist_run_results(
    results: list[GenerationResult], runs_dir: str | Path, run_id: str | None = None
) -> Path:
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or (
        results[0].run_id if results else datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    )
    output_path = runs_path / f"{resolved_run_id}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(result.to_record(), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return output_path


def persist_benchmark_summary(
    records: list[dict], benchmarks_dir: str | Path, experiment_name: str
) -> Path:
    """Append-or-replace 방식으로 benchmark summary parquet에 행 추가.

    같은 ``experiment_name``으로 여러 provider 평가를 순차로 돌리거나,
    같은 run을 다시 실행해도 이전 row들이 보존되도록 합니다.

    동작:
      - 파일이 없으면 새로 생성
      - 있으면 기존 parquet 읽고, 새 records의 ``run_id``와 같은 기존 row는
        제거(replace) 후 새 행 append → 같은 run을 다시 평가하면 마지막 결과
        우선, 다른 run은 그대로 보존
    """
    benchmark_path = Path(benchmarks_dir)
    benchmark_path.mkdir(parents=True, exist_ok=True)
    output_path = benchmark_path / f"{experiment_name}.parquet"
    new_df = pd.DataFrame(records)
    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
        except Exception:
            existing = pd.DataFrame()
        if "run_id" in new_df.columns and "run_id" in existing.columns:
            new_run_ids = set(new_df["run_id"].astype(str))
            existing = existing[~existing["run_id"].astype(str).isin(new_run_ids)]
        # 컬럼 union (새 컬럼이 추가되어도 안전하게)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    combined.to_parquet(output_path, index=False)
    return output_path


ProgressCallback = Callable[[int, int, "EvalSample"], None]


class BenchmarkRunner:
    """Run an evaluation dataset through an answer function and persist results."""

    def __init__(self, answer_fn: Callable[[EvalSample], GenerationResult]):
        self.answer_fn = answer_fn

    def run(
        self,
        experiment_name: str,
        scenario: str,
        provider_label: str,
        samples: list[EvalSample],
        progress_callback: ProgressCallback | None = None,
    ) -> BenchmarkRunResult:
        """Iterate samples and collect results.

        ``progress_callback(done, total, sample)`` is invoked after each sample
        so that UIs (Streamlit progress bar etc.) can render progress without
        coupling the runner to any specific frontend.
        """
        results: list[GenerationResult] = []
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            results.append(self.answer_fn(sample))
            if progress_callback is not None:
                progress_callback(index, total, sample)
        return BenchmarkRunResult(
            experiment_name=experiment_name,
            run_id=results[0].run_id if results else datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
            scenario=scenario,
            provider_label=provider_label,
            samples=samples,
            results=results,
            metrics={},
        )
