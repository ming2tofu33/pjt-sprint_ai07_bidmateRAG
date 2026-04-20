from pathlib import Path

import pandas as pd

from bidmate_rag.evaluation.runner import persist_benchmark_summary, persist_run_results
from bidmate_rag.schema import GenerationResult


def test_persist_run_results_and_summary(tmp_path: Path) -> None:
    result = GenerationResult(
        question_id="q-1",
        question="질문",
        scenario="scenario_b",
        run_id="run-1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        answer="답변",
        retrieved_chunk_ids=["chunk-1"],
        retrieved_doc_ids=["doc-1"],
        latency_ms=120,
        token_usage={"total": 30},
    )

    run_path = persist_run_results([result], tmp_path / "runs")
    summary_path = persist_benchmark_summary(
        [
            {
                "experiment_name": "generation-compare",
                "provider_label": "openai-gpt5-mini",
                "num_samples": 1,
                "avg_latency_ms": 120,
            }
        ],
        tmp_path / "benchmarks",
        "generation-compare",
    )

    assert run_path.exists()
    assert summary_path.exists()
    assert '"question_id":"q-1"' in run_path.read_text(encoding="utf-8")

    df = pd.read_parquet(summary_path)
    assert df.iloc[0]["provider_label"] == "openai-gpt5-mini"
