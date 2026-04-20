"""Unit tests for web_api retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace

from bidmate_rag.providers.llm.base import StreamDelta
from bidmate_rag.schema import Chunk, GenerationResult, RetrievedChunk
from bidmate_rag.web_api import retrieval_helpers as helpers
from bidmate_rag.web_api.retrieval_helpers import split_and_merge_chunks


def _make_chunk(doc_id: str, rank: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=score,
        chunk=Chunk(
            chunk_id=f"{doc_id}_{rank}",
            doc_id=doc_id,
            text=f"chunk {rank} of {doc_id}",
            text_with_meta=f"[doc={doc_id}] chunk {rank}",
            char_count=20,
            section="",
            content_type="text",
            chunk_index=rank - 1,
            metadata={"파일명": doc_id},
        ),
    )


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query, chat_history=None, top_k=5, metadata_filter=None):
        self.calls.append(
            {
                "query": query,
                "chat_history": list(chat_history or []),
                "top_k": top_k,
                "metadata_filter": dict(metadata_filter or {}),
            }
        )
        doc_id = metadata_filter["$or"][0]["doc_id"]
        return [_make_chunk(doc_id, i + 1, 0.9 - 0.1 * i) for i in range(top_k)]


class _ScoreMapRetriever:
    def __init__(self, score_map: dict[str, list[float]]) -> None:
        self.score_map = score_map
        self.calls: list[dict] = []

    def retrieve(self, query, chat_history=None, top_k=5, metadata_filter=None):
        self.calls.append(
            {
                "query": query,
                "chat_history": list(chat_history or []),
                "top_k": top_k,
                "metadata_filter": dict(metadata_filter or {}),
            }
        )
        doc_id = metadata_filter["$or"][0]["doc_id"]
        scores = self.score_map[doc_id][:top_k]
        return [_make_chunk(doc_id, i + 1, score) for i, score in enumerate(scores)]


class _FakePipelineRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    def retrieve(self, query, chat_history=None, top_k=5, metadata_filter=None):
        self.calls.append(
            {
                "query": query,
                "chat_history": list(chat_history or []),
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            }
        )
        return self.chunks[:top_k]


class _FakeLLM:
    provider_name = "openai"
    model_name = "gpt-5-mini"

    def __init__(self) -> None:
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return GenerationResult(
            question_id="q-web",
            question=kwargs["question"],
            scenario=kwargs["generation_config"]["scenario"],
            run_id=kwargs["generation_config"]["run_id"],
            embedding_provider=kwargs["generation_config"]["embedding_provider"],
            embedding_model=kwargs["generation_config"]["embedding_model"],
            llm_provider="openai",
            llm_model="gpt-5-mini",
            answer="테스트 응답",
            retrieved_chunk_ids=[c.chunk.chunk_id for c in kwargs["context_chunks"]],
            retrieved_doc_ids=[c.chunk.doc_id for c in kwargs["context_chunks"]],
            retrieved_chunks=kwargs["context_chunks"],
            latency_ms=12.3,
            token_usage={"prompt": 10, "completion": 5, "total": 15},
            cost_usd=0.0,
            context="ctx",
        )

    def generate_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield StreamDelta(text="테")
        yield self.generate(**kwargs)


class _FailLLM:
    provider_name = "openai"
    model_name = "gpt-5-mini"

    def generate(self, **kwargs):
        raise AssertionError("계산형 질문은 LLM generate를 타면 안 됩니다.")


def _make_calc_answer():
    return SimpleNamespace(
        mode="budget_difference",
        answer="핵심 답변:\n- 차액은 700,000,000원입니다.\n\n계산 근거:\n- 테스트",
    )


def test_per_doc_split_retrieves_each_doc_separately() -> None:
    retriever = _FakeRetriever()
    merged = split_and_merge_chunks(
        retriever,
        query="요구사항",
        mentioned_doc_ids=["A", "B", "C"],
        top_k=9,
    )

    assert len(retriever.calls) == 3
    per_doc_ks = {
        call["metadata_filter"]["$or"][0]["doc_id"]: call["top_k"] for call in retriever.calls
    }
    assert per_doc_ks == {"A": 5, "B": 5, "C": 5}
    assert len(merged) == 9
    assert {c.chunk.doc_id for c in merged} == {"A", "B", "C"}


def test_per_doc_split_resorts_by_score() -> None:
    retriever = _FakeRetriever()
    merged = split_and_merge_chunks(
        retriever,
        query="비교",
        mentioned_doc_ids=["A", "B"],
        top_k=4,
    )
    scores = [c.score for c in merged]
    assert scores == sorted(scores, reverse=True)


def test_per_doc_split_uses_larger_pool_for_comparison_queries() -> None:
    retriever = _FakeRetriever()
    split_and_merge_chunks(
        retriever,
        query="A 사업과 B 사업을 비교해줘",
        mentioned_doc_ids=["A", "B"],
        top_k=4,
    )

    per_doc_ks = {call["top_k"] for call in retriever.calls}
    assert per_doc_ks == {6}


def test_per_doc_split_minimum_k_is_three() -> None:
    retriever = _FakeRetriever()
    split_and_merge_chunks(
        retriever,
        query="q",
        mentioned_doc_ids=["A", "B", "C", "D", "E", "F"],
        top_k=5,
    )
    per_doc_ks = {call["top_k"] for call in retriever.calls}
    assert per_doc_ks == {5}


def test_comparison_merge_keeps_at_least_one_chunk_per_doc_when_top_k_allows() -> None:
    retriever = _ScoreMapRetriever(
        {
            "A": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
            "B": [0.70, 0.69, 0.68, 0.67, 0.66, 0.65],
            "C": [0.20, 0.19, 0.18, 0.17, 0.16, 0.15],
        }
    )
    merged = split_and_merge_chunks(
        retriever,
        query="A, B, C 사업을 비교해줘",
        mentioned_doc_ids=["A", "B", "C"],
        top_k=5,
    )

    assert len(merged) == 5
    assert {chunk.chunk.doc_id for chunk in merged} == {"A", "B", "C"}


def test_per_doc_split_forwards_chat_history() -> None:
    retriever = _FakeRetriever()
    history = [{"role": "user", "content": "이전 질문"}]

    split_and_merge_chunks(
        retriever,
        query="후속 질문",
        mentioned_doc_ids=["A", "B"],
        top_k=4,
        chat_history=history,
    )

    assert [call["chat_history"] for call in retriever.calls] == [history, history]


def test_web_query_forwards_chat_history_to_retriever_and_llm(monkeypatch) -> None:
    history = [
        {"role": "user", "content": "국민연금공단 ERP 사업 알려줘"},
        {"role": "assistant", "content": "차세대 ERP 사업입니다."},
    ]
    chunks = [_make_chunk("chunk-1", 1, 0.9)]
    retriever = _FakePipelineRetriever(chunks)
    llm = _FakeLLM()
    pipeline = SimpleNamespace(retriever=retriever, calculation_engine=None)
    runtime = SimpleNamespace(provider=SimpleNamespace(scenario="scenario_b", provider="openai"))
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")

    monkeypatch.setattr(
        helpers,
        "get_pipeline",
        lambda provider_config, chunking_config: (pipeline, runtime, embedder, llm),
    )

    helpers.web_query(
        question="그 사업 일정은?",
        augmented_query="그 사업 일정은?",
        mentioned_doc_ids=[],
        provider_config="openai_gpt5mini",
        chunking_config=None,
        system_prompt=None,
        top_k=3,
        max_context_chars=8000,
        chat_history=history,
    )

    assert retriever.calls[0]["chat_history"] == history
    assert llm.generate_calls[0]["history"] == history


def test_web_query_stream_forwards_chat_history_to_retriever_and_llm(monkeypatch) -> None:
    history = [{"role": "user", "content": "첫 질문"}]
    chunks = [_make_chunk("chunk-1", 1, 0.9)]
    retriever = _FakePipelineRetriever(chunks)
    llm = _FakeLLM()
    pipeline = SimpleNamespace(retriever=retriever, calculation_engine=None)
    runtime = SimpleNamespace(provider=SimpleNamespace(scenario="scenario_b", provider="openai"))
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")

    monkeypatch.setattr(
        helpers,
        "get_pipeline",
        lambda provider_config, chunking_config: (pipeline, runtime, embedder, llm),
    )

    events = list(
        helpers.web_query_stream(
            question="후속 질문",
            augmented_query="후속 질문",
            mentioned_doc_ids=[],
            provider_config="openai_gpt5mini",
            chunking_config=None,
            system_prompt=None,
            top_k=3,
            max_context_chars=8000,
            chat_history=history,
        )
    )

    assert [event[0] for event in events] == ["retrieval", "token", "done"]
    assert retriever.calls[0]["chat_history"] == history
    assert llm.stream_calls[0]["history"] == history


def test_web_query_uses_calculation_engine_before_llm(monkeypatch) -> None:
    chunks = [_make_chunk("doc-a.hwp", 1, 0.9), _make_chunk("doc-b.hwp", 2, 0.8)]
    retriever = _FakePipelineRetriever(chunks)
    pipeline = SimpleNamespace(
        retriever=retriever,
        calculation_engine=SimpleNamespace(try_answer=lambda **_kwargs: _make_calc_answer()),
    )
    runtime = SimpleNamespace(provider=SimpleNamespace(scenario="scenario_b", provider="openai"))
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")
    llm = _FailLLM()

    monkeypatch.setattr(
        helpers,
        "get_pipeline",
        lambda provider_config, chunking_config: (pipeline, runtime, embedder, llm),
    )

    result = helpers.web_query(
        question="두 사업 예산 차이가 얼마야?",
        augmented_query="두 사업 예산 차이가 얼마야?",
        mentioned_doc_ids=[],
        provider_config="openai_gpt5mini",
        chunking_config=None,
        system_prompt=None,
        top_k=3,
        max_context_chars=8000,
        chat_history=[],
    )

    assert result.answer.startswith("핵심 답변:")
    assert result.retrieved_doc_ids == ["doc-a.hwp", "doc-b.hwp"]


def test_web_query_stream_uses_calculation_engine_before_llm(monkeypatch) -> None:
    chunks = [_make_chunk("doc-a.hwp", 1, 0.9), _make_chunk("doc-b.hwp", 2, 0.8)]
    retriever = _FakePipelineRetriever(chunks)
    pipeline = SimpleNamespace(
        retriever=retriever,
        calculation_engine=SimpleNamespace(try_answer=lambda **_kwargs: _make_calc_answer()),
    )
    runtime = SimpleNamespace(provider=SimpleNamespace(scenario="scenario_b", provider="openai"))
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")
    llm = _FailLLM()

    monkeypatch.setattr(
        helpers,
        "get_pipeline",
        lambda provider_config, chunking_config: (pipeline, runtime, embedder, llm),
    )

    events = list(
        helpers.web_query_stream(
            question="두 사업 예산 차이가 얼마야?",
            augmented_query="두 사업 예산 차이가 얼마야?",
            mentioned_doc_ids=[],
            provider_config="openai_gpt5mini",
            chunking_config=None,
            system_prompt=None,
            top_k=3,
            max_context_chars=8000,
            chat_history=[],
        )
    )

    assert [event[0] for event in events] == ["retrieval", "token", "done"]
    assert events[1][1].startswith("핵심 답변:")
    assert events[2][1].retrieved_doc_ids == ["doc-a.hwp", "doc-b.hwp"]
