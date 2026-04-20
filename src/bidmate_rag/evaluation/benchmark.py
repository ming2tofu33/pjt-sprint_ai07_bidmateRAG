"""Compatibility wrapper for benchmark helpers."""

from bidmate_rag.evaluation.runner import (
    BenchmarkRunner,
    persist_benchmark_summary,
    persist_run_results,
)

__all__ = ["BenchmarkRunner", "persist_benchmark_summary", "persist_run_results"]
