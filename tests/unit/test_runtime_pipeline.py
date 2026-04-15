from pathlib import Path

from bidmate_rag.pipelines import runtime as runtime_module


def test_build_runtime_pipeline_passes_multiturn_flag_to_retriever(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    retrieval = tmp_path / "retrieval.yaml"
    base.write_text("project_name: bidmate-rag\n", encoding="utf-8")
    provider.write_text("provider: openai\nmodel: gpt-5-mini\n", encoding="utf-8")
    experiment.write_text("name: ad-hoc\n", encoding="utf-8")
    retrieval.write_text("enable_multiturn: false\n", encoding="utf-8")

    captured: dict = {}

    class FakeVectorStore:
        def count(self) -> int:
            return 1

    class FakeRetriever:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakePipeline:
        def __init__(self, retriever, llm) -> None:
            self.retriever = retriever
            self.llm = llm

    monkeypatch.setattr(runtime_module, "build_embedding_provider", lambda _: object())
    monkeypatch.setattr(runtime_module, "build_llm_provider", lambda _: object())
    monkeypatch.setattr(runtime_module, "ChromaVectorStore", lambda **kwargs: FakeVectorStore())
    monkeypatch.setattr(runtime_module, "_load_reranker", lambda _: None)
    monkeypatch.setattr(runtime_module, "RAGRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "RAGChatPipeline", FakePipeline)

    pipeline, *_ = runtime_module.build_runtime_pipeline(
        base_config_path=base,
        provider_config_path=provider,
        experiment_config_path=experiment,
        retrieval_config_path=retrieval,
        metadata_path=tmp_path / "missing.parquet",
    )

    assert captured["enable_multiturn"] is False
    assert isinstance(pipeline, FakePipeline)
