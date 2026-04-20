"""Tests for tracking/comparison.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bidmate_rag.tracking.comparison import (
    ComparisonData,
    load_runs_for_comparison,
    render_comparison_markdown,
)


def _make_parquet(
    path: Path,
    rows: list[dict],
) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _summary_row(run_id: str, exp: str, **metrics) -> dict:
    base = {
        "experiment_name": exp,
        "run_id": run_id,
        "scenario": "openai",
        "provider_label": "openai:gpt-5-mini",
        "num_samples": 2,
        "avg_latency_ms": 1000.0,
        "total_cost_usd": 0.001,
        "hit_rate@5": 0.5,
        "mrr": 0.5,
        "ndcg@5": 0.5,
        "faithfulness": 0.5,
        "answer_relevance": 0.5,
        "context_precision": 0.5,
        "context_recall": 0.5,
    }
    base.update(metrics)
    return base


# ---------------------------------------------------------------------------
# load_runs_for_comparison
# ---------------------------------------------------------------------------


def test_requires_either_experiment_or_run_ids():
    with pytest.raises(ValueError, match="Either experiment_name or run_ids"):
        load_runs_for_comparison()


def test_load_by_experiment_name(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [
            _summary_row("run-A", "exp1", **{"hit_rate@5": 0.6}),
            _summary_row("run-B", "exp1", **{"hit_rate@5": 0.8}),
        ],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    assert len(data.rows) == 2
    assert set(data.rows["run_id"]) == {"run-A", "run-B"}


def test_load_by_experiment_name_missing_file(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="benchmark file not found"):
        load_runs_for_comparison(experiment_name="missing", benchmarks_dir=tmp_path)


def test_load_by_run_ids_across_experiments(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [_summary_row("run-A", "exp1"), _summary_row("run-B", "exp1")],
    )
    _make_parquet(
        tmp_path / "exp2.parquet",
        [_summary_row("run-C", "exp2"), _summary_row("run-D", "exp2")],
    )
    data = load_runs_for_comparison(
        run_ids=["run-A", "run-C"], benchmarks_dir=tmp_path
    )
    assert len(data.rows) == 2
    assert set(data.rows["run_id"]) == {"run-A", "run-C"}
    assert set(data.rows["experiment_name"]) == {"exp1", "exp2"}


def test_load_by_run_ids_no_match_raises(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [_summary_row("run-A", "exp1")],
    )
    with pytest.raises(ValueError, match="No matching runs"):
        load_runs_for_comparison(run_ids=["nonexistent"], benchmarks_dir=tmp_path)


def test_metric_columns_only_includes_present(tmp_path):
    """parquet에 일부 metric 컬럼만 있어도 OK."""
    _make_parquet(
        tmp_path / "exp1.parquet",
        [{"experiment_name": "exp1", "run_id": "run-A", "hit_rate@5": 0.7}],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    assert "hit_rate@5" in data.metric_columns
    assert "faithfulness" not in data.metric_columns


# ---------------------------------------------------------------------------
# render_comparison_markdown
# ---------------------------------------------------------------------------


def test_render_includes_key_sections(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [
            _summary_row("run-A", "exp1", **{"hit_rate@5": 0.6}),
            _summary_row("run-B", "exp1", **{"hit_rate@5": 0.9}),
        ],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    md = render_comparison_markdown(data)
    assert "# Run Comparison" in md
    assert "Total runs**: 2" in md
    assert "exp1" in md
    assert "## 메트릭 비교" in md
    assert "## 메트릭별 최우/최저" in md
    assert "run-A" in md
    assert "run-B" in md


def test_higher_is_better_for_hit_rate(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [
            _summary_row("low", "exp1", **{"hit_rate@5": 0.3}),
            _summary_row("high", "exp1", **{"hit_rate@5": 0.9}),
        ],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    md = render_comparison_markdown(data)
    # hit_rate@5 행에서 best=high, worst=low
    hit_lines = [line for line in md.split("\n") if line.startswith("| hit_rate@5 |")]
    assert hit_lines, "hit_rate@5 row missing"
    line = hit_lines[0]
    assert "`high`" in line
    assert "`low`" in line
    # high가 best 위치(앞), low가 worst(뒤)
    assert line.index("`high`") < line.index("`low`")


def test_lower_is_better_for_cost(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [
            _summary_row("cheap", "exp1", total_cost_usd=0.001),
            _summary_row("expensive", "exp1", total_cost_usd=0.5),
        ],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    md = render_comparison_markdown(data)
    cost_lines = [line for line in md.split("\n") if line.startswith("| total_cost_usd |")]
    assert cost_lines
    line = cost_lines[0]
    # cost는 낮을수록 best → cheap이 best, expensive가 worst
    assert line.index("`cheap`") < line.index("`expensive`")


def test_render_handles_n_a_values(tmp_path):
    _make_parquet(
        tmp_path / "exp1.parquet",
        [
            _summary_row("run-A", "exp1"),
            {"experiment_name": "exp1", "run_id": "partial"},
        ],
    )
    data = load_runs_for_comparison(experiment_name="exp1", benchmarks_dir=tmp_path)
    md = render_comparison_markdown(data)
    assert "N/A" in md  # partial run의 missing 값들


def test_render_empty_dataframe():
    data = ComparisonData(rows=pd.DataFrame(), metric_columns=[])
    assert render_comparison_markdown(data) == "(no runs to compare)"
