"""Unit tests for web_api retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

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
            metadata={},
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
    # 9 // 3 + 2 = 5
    assert per_doc_ks == {"A": 5, "B": 5, "C": 5}

    # merged는 top_k=9로 절단된 상위 9개 — 각 문서에서 최소 1개씩 포함
    assert len(merged) == 9
    doc_ids = {c.chunk.doc_id for c in merged}
    assert doc_ids == {"A", "B", "C"}


def test_per_doc_split_resorts_by_score() -> None:
    retriever = _FakeRetriever()
    merged = split_and_merge_chunks(
        retriever,
        query="비교",
        mentioned_doc_ids=["A", "B"],
        top_k=4,
    )
    # 상위부터 내림차순으로 정렬됐는지 확인
    scores = [c.score for c in merged]
    assert scores == sorted(scores, reverse=True)


def test_per_doc_split_minimum_k_is_three() -> None:
    retriever = _FakeRetriever()
    split_and_merge_chunks(
        retriever,
        query="q",
        mentioned_doc_ids=["A", "B", "C", "D", "E", "F"],
        top_k=5,
    )
    # 5 // 6 = 0, max(0, 3) + 2 = 5
    per_doc_ks = {call["top_k"] for call in retriever.calls}
    assert per_doc_ks == {5}


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
    pipeline = SimpleNamespace(retriever=retriever)
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
    pipeline = SimpleNamespace(retriever=retriever)
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
