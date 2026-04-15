from __future__ import annotations

from types import SimpleNamespace

from bidmate_rag.config.settings import (
    ExperimentConfig,
    ProjectConfig,
    ProviderConfig,
    RuntimeConfig,
)
from bidmate_rag.schema import EvalSample


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
