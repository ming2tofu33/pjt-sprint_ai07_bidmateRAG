from bidmate_rag.schema import (
    BenchmarkRunResult,
    Chunk,
    Document,
    EvalSample,
    GenerationResult,
    RetrievedChunk,
)


def test_document_roundtrip_with_metadata() -> None:
    document = Document(
        doc_id="doc-001",
        source_path="data/raw/rfp/sample.hwp",
        file_type="hwp",
        title="샘플 사업",
        organization="샘플 기관",
        raw_text="본문",
        metadata={"공고 번호": "20240001"},
        parser_name="kordoc",
    )

    record = document.model_dump(mode="json")
    restored = Document.model_validate(record)

    assert restored.doc_id == "doc-001"
    assert restored.metadata["공고 번호"] == "20240001"
    assert restored.parser_name == "kordoc"


def test_generation_result_exposes_storage_record() -> None:
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="요구사항",
        text_with_meta="[발주기관: 기관]\n요구사항",
        char_count=4,
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={"파일명": "sample.hwp", "사업명": "샘플 사업", "발주 기관": "기관"},
    )
    result = GenerationResult(
        question_id="q-1",
        question="요구사항 알려줘",
        scenario="scenario_b",
        run_id="run-1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        answer="답변",
        retrieved_chunk_ids=["chunk-1"],
        retrieved_doc_ids=["doc-1"],
        retrieved_chunks=[RetrievedChunk(rank=1, score=0.9, chunk=chunk)],
        latency_ms=120.0,
        token_usage={"prompt": 10, "completion": 20, "total": 30},
        cost_usd=0.001,
        judge_scores={"faithfulness": 2},
        human_scores={"clarity": 1},
    )

    record = result.to_record()

    assert record["question_id"] == "q-1"
    assert record["retrieved_chunk_ids"] == ["chunk-1"]
    assert record["judge_scores"]["faithfulness"] == 2


def test_benchmark_run_result_summarizes_metrics() -> None:
    sample = EvalSample(question_id="q-1", question="질문", expected_doc_ids=["doc-1"])
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
        latency_ms=100,
        token_usage={"total": 30},
        cost_usd=0.001,
        judge_scores={"faithfulness": 2, "relevance": 2},
        debug={"rewrite_cost_usd": 0.0003},
    )

    benchmark = BenchmarkRunResult(
        experiment_name="generation-compare",
        run_id="run-1",
        scenario="scenario_b",
        provider_label="openai-gpt5-mini",
        samples=[sample],
        results=[result],
        metrics={"hit_rate@5": 1.0, "judge_cost_usd": 0.0008},
    )

    summary = benchmark.to_summary_record()

    assert summary["experiment_name"] == "generation-compare"
    assert summary["provider_label"] == "openai-gpt5-mini"
    assert summary["num_samples"] == 1
    assert summary["avg_latency_ms"] == 100
    assert summary["generation_cost_usd"] == 0.001
    assert summary["rewrite_cost_usd"] == 0.0003
    assert summary["judge_cost_usd"] == 0.0008
    assert summary["total_cost_usd"] == 0.0021
