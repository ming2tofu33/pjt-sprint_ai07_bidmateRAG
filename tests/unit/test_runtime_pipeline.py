import shutil
from pathlib import Path

from bidmate_rag.pipelines import runtime as runtime_module


def test_build_runtime_pipeline_passes_multiturn_flag_to_retriever(
    monkeypatch,
) -> None:
    tmp_path = Path(".pytest_tmp_runtime_pipeline")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base = tmp_path / "base.yaml"
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"
    retrieval = tmp_path / "retrieval.yaml"
    base.write_text("project_name: bidmate-rag\n", encoding="utf-8")
    provider.write_text("provider: openai\nmodel: gpt-5-mini\n", encoding="utf-8")
    experiment.write_text("name: ad-hoc\n", encoding="utf-8")
    retrieval.write_text(
        "enable_multiturn: false\n"
        "hybrid:\n"
        "  enabled: true\n"
        "  dense_pool_multiplier: 3\n"
        "  sparse_pool_multiplier: 3\n"
        "  rrf_k: 60\n",
        encoding="utf-8",
    )

    captured: dict = {}
    sparse_sentinel = object()

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
    monkeypatch.setattr(
        runtime_module.BM25SparseStore,
        "from_parquet",
        lambda *_args, **_kwargs: sparse_sentinel,
    )
    monkeypatch.setattr(runtime_module, "_load_reranker", lambda _: None)
    monkeypatch.setattr(runtime_module, "RAGRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "RAGChatPipeline", FakePipeline)

    chunks = tmp_path / "chunks.parquet"
    chunks.write_text("placeholder", encoding="utf-8")

    pipeline, *_ = runtime_module.build_runtime_pipeline(
        base_config_path=base,
        provider_config_path=provider,
        experiment_config_path=experiment,
        retrieval_config_path=retrieval,
        metadata_path=tmp_path / "missing.parquet",
        chunks_path=chunks,
    )

    assert captured["enable_multiturn"] is False
    assert captured["boost_config"] == {"section": 0.12, "table": 0.08, "max_total": 0.15}
    assert captured["hybrid_config"] == {
        "enabled": True,
        "dense_pool_multiplier": 3,
        "sparse_pool_multiplier": 3,
        "rrf_k": 60,
    }
    assert captured["sparse_store"] is sparse_sentinel
    assert isinstance(pipeline, FakePipeline)

    shutil.rmtree(tmp_path, ignore_errors=True)
