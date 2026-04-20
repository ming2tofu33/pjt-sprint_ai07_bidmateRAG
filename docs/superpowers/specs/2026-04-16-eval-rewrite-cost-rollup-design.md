# Evaluation Rewrite Cost Rollup Design

**Date**: 2026-04-16
**Status**: Approved

## Overview

The evaluation pipeline already records rewrite token usage and per-sample
`rewrite_cost_usd`, but aggregated run outputs still treat `total_cost_usd` as
`generation + judge`. This causes the same run to show different totals between
single-question debug output, batch summaries, parquet summaries, and markdown
reports.

This design makes evaluation cost reporting consistent across the full reporting
path.

## Goal

- Expose `rewrite_cost_usd` as an explicit aggregated metric.
- Make evaluation `total_cost_usd` include rewrite cost.
- Keep CLI summary, run `meta.json`, benchmark parquet, and markdown report on
  the same cost definition.

## Non-Goals

- Changing per-sample generation behavior.
- Changing embedding cost handling.
- Reworking the judge pipeline or pricing rules.

## Cost Definition

- `generation_cost_usd`: sum of `GenerationResult.cost_usd`
- `rewrite_cost_usd`: sum of `GenerationResult.debug["rewrite_cost_usd"]`
- `judge_cost_usd`: aggregated judge cost
- `total_cost_usd`: `generation_cost_usd + rewrite_cost_usd + judge_cost_usd`

For markdown reports, the displayed grand total remains:

- `grand_total_cost = total_cost_usd + embedding_cost_usd`

## Change Areas

### 1. Evaluation Aggregation

`src/bidmate_rag/evaluation/metrics.py`

- Aggregate `rewrite_cost_usd` from each `GenerationResult`.
- Return `rewrite_cost_usd` in `summarize_run_operations()`.
- Update aggregated `total_cost_usd` to include rewrite cost.

### 2. CLI Output

`src/bidmate_rag/cli/eval.py`

- Print `rewrite_cost_usd` in the run summary.
- Print `rewrite_cost_usd` in the artifact cost recap.

### 3. Benchmark Summary

`src/bidmate_rag/schema.py`

- Include `generation_cost_usd`, `rewrite_cost_usd`, `judge_cost_usd`, and the
  updated `total_cost_usd` in `BenchmarkRunResult.to_summary_record()`.
- Prefer already-aggregated benchmark metrics when present so parquet rows use
  the same numbers as run metadata.

### 4. Markdown Report

`src/bidmate_rag/tracking/markdown_report.py`
`src/bidmate_rag/tracking/templates.py`

- Show rewrite cost as its own row.
- Build report costs from run metadata when available so the report matches the
  evaluation summary.
- Keep embedding cost separate and add it on top of the LLM total.

## Error Handling

- Missing `rewrite_cost_usd` falls back to `0.0`.
- Existing runs without the new aggregated fields still render correctly by
  falling back to per-result sums.

## Test Plan

- Update unit tests for evaluation metrics aggregation.
- Update CLI summary tests.
- Update schema summary tests.
- Update markdown report tests for rewrite cost and total rollup.
