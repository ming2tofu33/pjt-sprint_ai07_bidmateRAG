from __future__ import annotations

from types import SimpleNamespace

from bidmate_rag.cli.eval import _print_summary
from bidmate_rag.config.settings import (
    ExperimentConfig,
    ProjectConfig,
    ProviderConfig,
    RuntimeConfig,
)
from bidmate_rag.schema import EvalSample, GenerationResult


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(provider="openai", model="gpt-5-mini", scenario="scenario_b"),
        experiment=ExperimentConfig(name="ad-hoc"),
    )


def test_eval_cli_prints_progress_when_enabled(monkeypatch, capsys) -> None:
    from bidmate_rag.cli import eval as eval_cli

    samples = [
        EvalSample(question_id="Q001", question="첫 번째 질문"),
        EvalSample(question_id="Q002", question="두 번째 질문"),
    ]
    pipeline = SimpleNamespace(
        retriever=SimpleNamespace(metadata_store=SimpleNamespace(agency_list=[]))
    )
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")

    monkeypatch.setattr(
        eval_cli,
        "build_runtime_pipeline",
        lambda **kwargs: (pipeline, _runtime(), embedder, None),
    )
    monkeypatch.setattr(eval_cli, "load_eval_samples", lambda *args, **kwargs: samples)

    class _Report:
        def is_valid(self, strict: bool = False) -> bool:
            return True

    monkeypatch.setattr(eval_cli, "validate_eval_samples", lambda *args, **kwargs: _Report())
    monkeypatch.setattr(eval_cli, "render_validation_report", lambda report: "validation ok")
    monkeypatch.setattr(eval_cli, "_resolve_metadata_path", lambda runtime, path: SimpleNamespace())
    monkeypatch.setattr(eval_cli, "_print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_cli, "_print_artifacts", lambda *args, **kwargs: None)

    def fake_execute_evaluation(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None
        for index, sample in enumerate(samples, start=1):
            progress_callback(index, len(samples), sample)
        return SimpleNamespace(benchmark=SimpleNamespace(results=[]), metrics={})

    monkeypatch.setattr(eval_cli, "execute_evaluation", fake_execute_evaluation)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bidmate-eval",
            "--evaluation-path",
            "data/eval/eval_v1/eval_batch_01.csv",
            "--provider-config",
            "configs/providers/openai_gpt5mini.yaml",
            "--limit",
            "2",
            "--progress",
        ],
    )

    eval_cli.main()
    out = capsys.readouterr().out

    assert "[1/2]" in out
    assert "Q001" in out
    assert "첫 번째 질문" in out
    assert "[2/2]" in out
    assert "Q002" in out


def test_eval_cli_does_not_print_progress_by_default(monkeypatch, capsys) -> None:
    from bidmate_rag.cli import eval as eval_cli

    samples = [EvalSample(question_id="Q001", question="첫 번째 질문")]
    pipeline = SimpleNamespace(
        retriever=SimpleNamespace(metadata_store=SimpleNamespace(agency_list=[]))
    )
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")

    monkeypatch.setattr(
        eval_cli,
        "build_runtime_pipeline",
        lambda **kwargs: (pipeline, _runtime(), embedder, None),
    )
    monkeypatch.setattr(eval_cli, "load_eval_samples", lambda *args, **kwargs: samples)

    class _Report:
        def is_valid(self, strict: bool = False) -> bool:
            return True

    monkeypatch.setattr(eval_cli, "validate_eval_samples", lambda *args, **kwargs: _Report())
    monkeypatch.setattr(eval_cli, "render_validation_report", lambda report: "validation ok")
    monkeypatch.setattr(eval_cli, "_resolve_metadata_path", lambda runtime, path: SimpleNamespace())
    monkeypatch.setattr(eval_cli, "_print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_cli, "_print_artifacts", lambda *args, **kwargs: None)

    def fake_execute_evaluation(*args, progress_callback=None, **kwargs):
        assert progress_callback is None
        return SimpleNamespace(benchmark=SimpleNamespace(results=[]), metrics={})

    monkeypatch.setattr(eval_cli, "execute_evaluation", fake_execute_evaluation)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bidmate-eval",
            "--evaluation-path",
            "data/eval/eval_v1/eval_batch_01.csv",
            "--provider-config",
            "configs/providers/openai_gpt5mini.yaml",
            "--limit",
            "1",
        ],
    )

    eval_cli.main()
    out = capsys.readouterr().out

    assert "[1/1]" not in out
    assert "Q001" not in out


def test_print_summary_includes_cost_sections(capsys) -> None:
    from bidmate_rag.cli import eval as eval_cli

    samples = [
        EvalSample(question_id="Q001", question="질문", metadata={"type": "C", "difficulty": "중"})
    ]
    results = [
        GenerationResult(
            question_id="Q001",
            question="질문",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="openai",
            llm_model="gpt-5-mini",
            answer="답변",
            latency_ms=1500.0,
            token_usage={"prompt": 100, "completion": 20, "total": 120},
            cost_usd=0.0012,
        )
    ]
    metrics = {
        "hit_rate@5": 1.0,
        "mrr": 1.0,
        "faithfulness": 0.9,
    }
    ops_metrics = {
        "generation_cost_usd": 0.0012,
        "rewrite_cost_usd": 0.0002,
        "judge_cost_usd": 0.0004,
        "total_cost_usd": 0.0018,
        "total_tokens": 120,
        "avg_latency_ms": 1500.0,
        "rewrite_total_tokens": 0,
    }

    eval_cli._print_summary(samples, results, metrics, ops_metrics)
    out = capsys.readouterr().out

    assert "retrieval: hit_rate@5=1.0  mrr=1.0" in out
    assert "judge:     faithfulness=0.9" in out
    assert "ops:" in out
    assert "generation_cost_usd=$0.0012" in out
    assert "rewrite_cost_usd=$0.0002" in out
    assert "judge_cost_usd=$0.0004" in out
    assert "total_cost_usd=$0.0018" in out


def test_print_summary_hides_rewrite_sections_when_unused(capsys) -> None:
    from bidmate_rag.cli import eval as eval_cli

    samples = [
        EvalSample(question_id="Q001", question="질문", metadata={"type": "C", "difficulty": "중"})
    ]
    results = [
        GenerationResult(
            question_id="Q001",
            question="질문",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="openai",
            llm_model="gpt-5-mini",
            answer="응답",
            latency_ms=1500.0,
            token_usage={"prompt": 100, "completion": 20, "total": 120},
            cost_usd=0.0012,
        )
    ]
    metrics = {"hit_rate@5": 1.0}
    ops_metrics = {
        "generation_cost_usd": 0.0012,
        "rewrite_cost_usd": 0.0,
        "judge_cost_usd": 0.0004,
        "total_cost_usd": 0.0016,
        "total_tokens": 120,
        "avg_latency_ms": 1500.0,
        "rewrite_total_tokens": 0,
    }

    eval_cli._print_summary(samples, results, metrics, ops_metrics)
    eval_cli._print_artifacts(
        SimpleNamespace(
            run_id="run-1",
            run_path="runs/run-1.jsonl",
            summary_path="benchmarks/test.parquet",
            meta_path="runs/run-1.meta.json",
            ops_metrics=ops_metrics,
            judge_skipped=True,
            judge_total_cost_usd=0.0,
            judge_total_tokens=0,
        )
    )
    out = capsys.readouterr().out

    assert "rewrite_cost_usd=" not in out
    assert "rewrite_total_tokens=" not in out
    assert " rewrite=" not in out


def test_print_summary_renders_per_type_retrieval_section(capsys) -> None:
    """overall_metrics에 retrieval_by_type가 있으면 CLI가 별도 섹션으로 출력한다."""
    samples = [
        EvalSample(
            question_id="q1",
            question="q",
            expected_doc_titles=["q1.hwp"],
            metadata={"type": "A", "difficulty": "중"},
        ),
        EvalSample(
            question_id="q2",
            question="q",
            expected_doc_titles=["q2.hwp"],
            metadata={"type": "C", "difficulty": "중"},
        ),
    ]
    results = [
        GenerationResult(
            question_id="q1",
            question="q",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="fake",
            llm_model="fake-model",
            answer="ans",
            retrieved_chunk_ids=[],
            retrieved_doc_ids=[],
            retrieved_chunks=[],
            latency_ms=1.0,
            token_usage={"total": 100},
            cost_usd=0.001,
        ),
        GenerationResult(
            question_id="q2",
            question="q",
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="fake",
            llm_model="fake-model",
            answer="ans",
            retrieved_chunk_ids=[],
            retrieved_doc_ids=[],
            retrieved_chunks=[],
            latency_ms=1.0,
            token_usage={"total": 100},
            cost_usd=0.001,
        ),
    ]
    overall_metrics = {
        "hit_rate@5": 0.5,
        "mrr": 0.5,
        "ndcg@5": 0.5,
        "map@5": 0.5,
        "retrieval_by_type": {
            "A": {"n": 1, "hit_rate@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "map@5": 1.0},
            "C": {"n": 1, "hit_rate@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "map@5": 0.0},
        },
    }

    _print_summary(samples, results, overall_metrics)
    output = capsys.readouterr().out
    assert "by type (retrieval)" in output
    # Type A row must appear before Type C row (alphabetical order).
    a_idx = output.find("A  ")
    c_idx = output.find("C  ")
    assert 0 <= a_idx < c_idx, f"Expected A before C in output:\n{output}"
    # hit_rate@5 column should show 1.0 for Type A and 0.0 for Type C.
    assert "1.0" in output
    assert "0.0" in output
