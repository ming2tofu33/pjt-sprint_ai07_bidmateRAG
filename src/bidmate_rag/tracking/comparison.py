"""Multi-run benchmark 비교 모듈.

여러 평가 run의 메트릭을 한 표로 합치고, 메트릭별 best/worst를 분석해
마크다운으로 렌더링합니다. ``bidmate-compare`` CLI 본체에서 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 비교 표에 보일 메트릭 컬럼들 (정해진 순서로 노출)
_METRIC_COLUMNS = [
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "avg_latency_ms",
    "total_cost_usd",
]

# 각 메트릭의 방향성 — best는 max인지 min인지
_HIGHER_IS_BETTER = {
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
}


@dataclass
class ComparisonData:
    """비교 결과 데이터. 메트릭 DataFrame과 표시할 컬럼 목록을 담는다."""

    rows: pd.DataFrame        # run별 메트릭이 담긴 DataFrame
    metric_columns: list[str]  # 비교 표에 표시할 메트릭 컬럼 목록


def load_runs_for_comparison(
    run_ids: list[str] | None = None,
    experiment_name: str | None = None,
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
) -> ComparisonData:
    """run_id 리스트 또는 experiment_name으로 비교 대상을 로드한다.

    Args:
        run_ids: 비교할 run_id 리스트.
        experiment_name: 실험 이름 (해당 실험의 모든 run 비교).
        benchmarks_dir: 벤치마크 parquet 디렉터리 경로.

    Returns:
        ComparisonData 인스턴스.

    Raises:
        ValueError: run_ids와 experiment_name 모두 None인 경우.
        FileNotFoundError: 벤치마크 디렉터리나 파일이 없는 경우.
    """
    if not experiment_name and not run_ids:
        raise ValueError("Either experiment_name or run_ids is required")

    benchmark_path = Path(benchmarks_dir)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmarks dir not found: {benchmark_path}")

    frames: list[pd.DataFrame] = []
    if experiment_name:
        # 특정 실험의 parquet 파일에서 모든 run 로드
        target = benchmark_path / f"{experiment_name}.parquet"
        if not target.exists():
            raise FileNotFoundError(f"benchmark file not found: {target}")
        frames.append(pd.read_parquet(target))
    else:
        # 전체 parquet 스캔하여 해당 run_id만 수집
        run_id_set = set(run_ids or [])
        for parquet_file in sorted(benchmark_path.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
            except Exception:
                continue
            if "run_id" not in df.columns:
                continue
            matched = df[df["run_id"].astype(str).isin(run_id_set)]
            if not matched.empty:
                frames.append(matched)

    if not frames:
        raise ValueError("No matching runs found")

    # 여러 DataFrame을 하나로 합치고 비교할 메트릭 컬럼 추출
    combined = pd.concat(frames, ignore_index=True, sort=False)
    metric_cols = [c for c in _METRIC_COLUMNS if c in combined.columns]
    return ComparisonData(rows=combined, metric_columns=metric_cols)


def render_comparison_markdown(data: ComparisonData) -> str:
    """비교 결과를 마크다운으로 렌더링한다.

    Args:
        data: ComparisonData 인스턴스.

    Returns:
        마크다운 형식의 비교 리포트 문자열.
    """
    df = data.rows.copy()
    if df.empty:
        return "(no runs to compare)"

    # 헤더 및 요약 정보
    lines = ["# Run Comparison", ""]
    lines.append(f"**Total runs**: {len(df)}")
    if "experiment_name" in df.columns:
        exps = sorted(df["experiment_name"].dropna().astype(str).unique().tolist())
        if exps:
            lines.append(f"**Experiments**: {', '.join(exps)}")
    lines.append("")

    # 메트릭 표 (run × metric)
    lines.append("## 메트릭 비교")
    lines.append("")
    label_cols = [c for c in ("run_id", "experiment_name", "provider_label") if c in df.columns]
    show = df[label_cols + data.metric_columns]
    lines.append(_df_to_markdown(show))
    lines.append("")

    # 메트릭별 best/worst — 각 메트릭에서 가장 좋은/나쁜 run 표시
    if data.metric_columns:
        lines.append("## 메트릭별 최우/최저")
        lines.append("")
        lines.append("| 메트릭 | 최우 run | 값 | 최저 run | 값 |")
        lines.append("| --- | --- | --- | --- | --- |")
        run_id_col = df["run_id"] if "run_id" in df.columns else pd.Series(["?"] * len(df))
        for col in data.metric_columns:
            s = df[col].dropna()
            if s.empty:
                continue
            if col in _HIGHER_IS_BETTER:
                best_idx, worst_idx = s.idxmax(), s.idxmin()
            else:
                best_idx, worst_idx = s.idxmin(), s.idxmax()
            best_run = run_id_col.iloc[best_idx] if best_idx in run_id_col.index else run_id_col[best_idx]
            worst_run = run_id_col.iloc[worst_idx] if worst_idx in run_id_col.index else run_id_col[worst_idx]
            lines.append(
                f"| {col} | `{best_run}` | {s[best_idx]:.4f} | "
                f"`{worst_run}` | {s[worst_idx]:.4f} |"
            )

    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """DataFrame을 마크다운 표로 변환한다 (외부 의존성 없음).

    Args:
        df: 변환할 DataFrame.

    Returns:
        마크다운 표 문자열.
    """
    if df.empty:
        return "(empty)"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if pd.isna(v):
                cells.append("N/A")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
