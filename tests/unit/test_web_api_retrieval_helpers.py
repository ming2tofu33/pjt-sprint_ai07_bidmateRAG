"""Unit tests for per_doc_split_query."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from bidmate_rag.schema import Chunk, GenerationResult, RetrievedChunk
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
            {"query": query, "top_k": top_k, "metadata_filter": dict(metadata_filter or {})}
        )
        doc_id = metadata_filter["doc_id"]
        return [_make_chunk(doc_id, i + 1, 0.9 - 0.1 * i) for i in range(top_k)]


def test_per_doc_split_retrieves_each_doc_separately() -> None:
    retriever = _FakeRetriever()
    merged = split_and_merge_chunks(
        retriever,
        query="요구사항",
        mentioned_doc_ids=["A", "B", "C"],
        top_k=9,
    )

    assert len(retriever.calls) == 3
    per_doc_ks = {call["metadata_filter"]["doc_id"]: call["top_k"] for call in retriever.calls}
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
