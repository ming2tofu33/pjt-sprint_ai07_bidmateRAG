from __future__ import annotations

from types import SimpleNamespace

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
