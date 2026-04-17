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
        "  rrf_k: 60\n"
        "rewrite:\n"
        "  mode: llm_with_rule_fallback\n"
        "memory:\n"
        "  enabled: true\n"
        "  summary_buffer:\n"
        "    max_recent_turns: 4\n"
        "    max_summary_chars: 120\n"
        "  slot_memory:\n"
        "    enabled: true\n"
        "debug_trace:\n"
        "  enabled: true\n",
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
        def __init__(self, retriever, llm, memory=None, debug_trace_enabled=True) -> None:
            self.retriever = retriever
            self.llm = llm
            self.memory = memory
            self.debug_trace_enabled = debug_trace_enabled

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
    assert captured["rewrite_llm"] is None
    assert captured["rewrite_mode"] == "llm_with_rule_fallback"
    assert captured["memory"] is not None
    assert captured["debug_trace_enabled"] is True
    assert captured["sparse_store"] is sparse_sentinel
    assert isinstance(pipeline, FakePipeline)
    assert pipeline.memory is not None

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_runtime_pipeline_passes_llm_to_retriever_when_multiturn_enabled(
    monkeypatch,
) -> None:
    tmp_path = Path(".pytest_tmp_runtime_pipeline_llm")
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
        "enable_multiturn: true\n"
        "hybrid:\n"
        "  enabled: false\n"
        "  dense_pool_multiplier: 3\n"
        "  sparse_pool_multiplier: 3\n"
        "  rrf_k: 60\n"
        "rewrite:\n"
        "  mode: llm_with_rule_fallback\n"
        "memory:\n"
        "  enabled: true\n"
        "  summary_buffer:\n"
        "    max_recent_turns: 4\n"
        "    max_summary_chars: 120\n"
        "  slot_memory:\n"
        "    enabled: true\n"
        "debug_trace:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )

    captured: dict = {}
    llm_sentinel = object()

    class FakeVectorStore:
        def count(self) -> int:
            return 1

    class FakeRetriever:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakePipeline:
        def __init__(self, retriever, llm, memory=None, debug_trace_enabled=True) -> None:
            self.retriever = retriever
            self.llm = llm
            self.memory = memory
            self.debug_trace_enabled = debug_trace_enabled

    monkeypatch.setattr(runtime_module, "build_embedding_provider", lambda _: object())
    monkeypatch.setattr(runtime_module, "build_llm_provider", lambda _: llm_sentinel)
    monkeypatch.setattr(runtime_module, "ChromaVectorStore", lambda **kwargs: FakeVectorStore())
    monkeypatch.setattr(runtime_module, "_load_reranker", lambda _: None)
    monkeypatch.setattr(runtime_module, "RAGRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "RAGChatPipeline", FakePipeline)

    chunks = tmp_path / "chunks.parquet"
    chunks.write_text("placeholder", encoding="utf-8")

    runtime_module.build_runtime_pipeline(
        base_config_path=base,
        provider_config_path=provider,
        experiment_config_path=experiment,
        retrieval_config_path=retrieval,
        metadata_path=tmp_path / "missing.parquet",
        chunks_path=chunks,
    )

    assert captured["enable_multiturn"] is True
    assert captured["rewrite_llm"] is llm_sentinel
    assert captured["rewrite_mode"] == "llm_with_rule_fallback"
    assert captured["memory"] is not None
    assert captured["debug_trace_enabled"] is True

    shutil.rmtree(tmp_path, ignore_errors=True)
