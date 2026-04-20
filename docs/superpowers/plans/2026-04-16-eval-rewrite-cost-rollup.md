# Evaluation Rewrite Cost Rollup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make evaluation outputs report rewrite cost separately and include it in aggregated total cost.

**Architecture:** Update the shared evaluation cost aggregator first, then propagate the new metric through CLI output, benchmark summaries, and markdown reports. Keep existing per-sample behavior unchanged and use the aggregated run metrics as the source of truth wherever possible.

**Tech Stack:** Python, Pydantic, pytest

---

## File Structure

- Modify: `src/bidmate_rag/evaluation/metrics.py`
- Modify: `src/bidmate_rag/cli/eval.py`
- Modify: `src/bidmate_rag/schema.py`
- Modify: `src/bidmate_rag/tracking/markdown_report.py`
- Modify: `src/bidmate_rag/tracking/templates.py`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_cli_eval.py`
- Test: `tests/unit/test_schema.py`
- Test: `tests/unit/test_markdown_report.py`

### Task 1: Aggregate Rewrite Cost

**Files:**
- Modify: `src/bidmate_rag/evaluation/metrics.py`
- Test: `tests/unit/test_metrics.py`

- [ ] Add `rewrite_cost_usd` aggregation to `summarize_run_operations()`.
- [ ] Make aggregated `total_cost_usd` equal generation + rewrite + judge.
- [ ] Extend the metric unit test with per-result rewrite cost and updated totals.

### Task 2: Surface The Metric In CLI Output

**Files:**
- Modify: `src/bidmate_rag/cli/eval.py`
- Test: `tests/unit/test_cli_eval.py`

- [ ] Add `rewrite_cost_usd` to the `ops:` summary line.
- [ ] Add `rewrite_cost_usd` to the artifact cost line.
- [ ] Update the CLI unit test to assert the new field and total.

### Task 3: Persist Consistent Summary Rows

**Files:**
- Modify: `src/bidmate_rag/schema.py`
- Test: `tests/unit/test_schema.py`

- [ ] Update `BenchmarkRunResult.to_summary_record()` to carry generation,
  rewrite, judge, and total cost fields.
- [ ] Prefer aggregated `metrics` values when available.
- [ ] Extend schema tests to verify the rolled-up total.

### Task 4: Align Markdown Reports

**Files:**
- Modify: `src/bidmate_rag/tracking/markdown_report.py`
- Modify: `src/bidmate_rag/tracking/templates.py`
- Test: `tests/unit/test_markdown_report.py`

- [ ] Add a dedicated rewrite-cost row to the report template.
- [ ] Use run-meta aggregated costs when present and fall back safely for old
  runs.
- [ ] Verify the report shows rewrite cost and that the grand total includes it.

### Task 5: Verify End To End

**Files:**
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_cli_eval.py`
- Test: `tests/unit/test_schema.py`
- Test: `tests/unit/test_markdown_report.py`

- [ ] Run the targeted unit tests for the touched cost-reporting paths.
- [ ] Review the diff to confirm no unrelated behavior changed.

## Self-Review

- Spec coverage: aggregation, CLI, parquet summary, and report output are all
  covered.
- Placeholder scan: no TODO or TBD markers remain.
- Type consistency: the same four cost names are used throughout the plan:
  `generation_cost_usd`, `rewrite_cost_usd`, `judge_cost_usd`,
  `total_cost_usd`.
