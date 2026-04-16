"""Shared data schemas for documents, chunks, and responses."""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """인제스트 과정에서 생성되는 정규화된 문서 레코드."""

    doc_id: str
    source_path: str
    file_type: str
    title: str
    organization: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str | None = None
    parser_version: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """파싱·정제된 문서에서 생성된 청크."""

    chunk_id: str
    doc_id: str
    text: str
    text_with_meta: str
    char_count: int
    section: str = ""
    content_type: str = "text"
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "text_with_meta": self.text_with_meta,
            "char_count": self.char_count,
            "section": self.section,
            "content_type": self.content_type,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


class RetrievedChunk(BaseModel):
    """리트리버가 반환하는 청크와 검색 점수."""

    rank: int
    score: float
    rerank_score: float | None = None
    chunk: Chunk

    def to_record(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "content_type": self.chunk.content_type,
            "metadata": self.chunk.metadata,
            "text": self.chunk.text,
        }


class GenerationResult(BaseModel):
    """평가 또는 라이브 질의에 대해 생성된 하나의 답변."""

    question_id: str
    question: str
    scenario: str
    run_id: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
    judge_scores: dict[str, Any] = Field(default_factory=dict)
    human_scores: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    context: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "answer": self.answer,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_chunks": [chunk.to_record() for chunk in self.retrieved_chunks],
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "judge_scores": self.judge_scores,
            "human_scores": self.human_scores,
            "error": self.error,
            "context": self.context,
        }


class EvalSample(BaseModel):
    """평가셋의 질문 하나."""

    question_id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_doc_titles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunResult(BaseModel):
    """벤치마크 실행 요약 및 질문별 원시 결과."""

    experiment_name: str
    run_id: str
    scenario: str
    provider_label: str
    samples: list[EvalSample] = Field(default_factory=list)
    results: list[GenerationResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def to_summary_record(self) -> dict[str, Any]:
        latencies = [result.latency_ms for result in self.results if result.latency_ms is not None]
        costs = [result.cost_usd for result in self.results if result.cost_usd is not None]
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "scenario": self.scenario,
            "provider_label": self.provider_label,
            "num_samples": len(self.samples) or len(self.results),
            "avg_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
            "total_cost_usd": round(sum(costs), 6),
            **self.metrics,
        }
