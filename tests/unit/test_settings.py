import json
from pathlib import Path

from bidmate_rag.config.settings import load_runtime_config
from bidmate_rag.config.settings import ExperimentConfig, ProjectConfig, ProviderConfig, RuntimeConfig
from bidmate_rag.evaluation.pipeline import execute_evaluation
from bidmate_rag.schema import EvalSample, GenerationResult


def test_load_runtime_config_merges_base_provider_and_experiment(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"

    base.write_text("default_retrieval_top_k: 5\ndefault_chunk_size: 1000\n")
    provider.write_text("provider: openai\nmodel: gpt-5-mini\nscenario: scenario_b\n")
    experiment.write_text("name: generation-compare\nmode: generation_only\nretrieval_top_k: 8\n")

    config = load_runtime_config(base, provider, experiment)

    assert config.project.default_retrieval_top_k == 5
    assert config.provider.model == "gpt-5-mini"
    assert config.experiment.retrieval_top_k == 8
    assert config.experiment.mode == "generation_only"


def test_load_runtime_config_reads_experiment_notes_path(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"

    base.write_text("project_name: bidmate-rag\n")
    provider.write_text("provider: openai\nmodel: gpt-5-mini\n")
    experiment.write_text(
        "name: budget-quality\nmode: full_rag\nnotes_path: configs/experiments/notes/example-budget-quality.yaml\n"
    )

    config = load_runtime_config(base, provider, experiment)

    assert config.experiment.notes_path == "configs/experiments/notes/example-budget-quality.yaml"


def test_execute_evaluation_persists_notes_path_in_meta(tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(provider="openai", model="gpt-5-mini", scenario="scenario_b"),
        experiment=ExperimentConfig(
            name="budget-quality",
            mode="full_rag",
            notes_path="configs/experiments/notes/example-budget-quality.yaml",
        ),
    )
    sample = EvalSample(question_id="q-1", question="질문")

    class _Embedder:
        provider_name = "openai"
        model_name = "text-embedding-3-small"

    class _Pipeline:
        def answer(self, question: str, **kwargs) -> GenerationResult:
            return GenerationResult(
                question_id="q-1",
                question=question,
                scenario=kwargs["scenario"],
                run_id=kwargs["run_id"],
                embedding_provider=kwargs["embedding_provider"],
                embedding_model=kwargs["embedding_model"],
                llm_provider="openai",
                llm_model="gpt-5-mini",
                answer="답변",
            )

    artifacts = execute_evaluation(
        [sample],
        pipeline=_Pipeline(),
        runtime=runtime,
        embedder=_Embedder(),
        eval_path="configs/evals/example.csv",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        run_id="bench-test1234",
        skip_judge=True,
    )

    meta = json.loads(artifacts.meta_path.read_text(encoding="utf-8"))
    assert meta["notes_path"] == "configs/experiments/notes/example-budget-quality.yaml"
    assert meta["config_snapshot"]["experiment"]["notes_path"] == (
        "configs/experiments/notes/example-budget-quality.yaml"
    )
