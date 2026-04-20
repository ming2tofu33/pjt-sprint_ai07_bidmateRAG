"""Pydantic models for the BidMate web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """카탈로그 리스트용 문서 요약."""

    id: str
    title: str
    agency: str
    agency_type: str
    domain: str
    budget: float
    budget_label: str
    deadline: str | None = None
    char_count: int


class DocumentDetail(DocumentSummary):
    """단일 문서 상세 (Quick Facts 카드용)."""

    summary_oneline: str | None = None
    quick_facts: list[dict[str, str]] = Field(default_factory=list)


class DocumentContent(BaseModel):
    """문서 미리보기용 전체 마크다운 본문."""

    doc_id: str
    title: str
    markdown: str
    char_count: int


class SlashCommandMeta(BaseModel):
    """프론트 드롭다운용 슬래시 커맨드 메타."""

    id: str
    label: str
    description: str
    icon: str
    requires_doc: bool = False
    requires_multi_doc: bool = False


class QueryRequest(BaseModel):
    """POST /api/query 요청 바디.

    `provider_config`/`chunking_config`/`top_k`/`max_context_chars`를
    생략하면 서버의 `configs/web.yaml` 기본값으로 폴백된다. 웹 UI 배포 조합
    변경 시 그 파일만 수정하면 전체 클라이언트에 반영된다.
    """

    question: str
    provider_config: str | None = None
    chunking_config: str | None = None
    mentioned_doc_ids: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    command: str | None = None
    top_k: int | None = None
    max_context_chars: int | None = None


class Citation(BaseModel):
    """답변 본문의 [n]과 매칭되는 근거 카드."""

    id: int
    doc_id: str
    doc_title: str
    section: str
    content_type: str
    text: str
    score: float


class QueryMetadata(BaseModel):
    """쿼리 응답의 실행 메타데이터."""

    model: str
    token_usage: dict[str, Any]
    latency_ms: float
    cost_usd: float
    command_applied: str | None
    filter_applied: dict[str, Any] | None
    retrieval_strategy: Literal["single", "per_doc_split", "static"]
    per_doc_k: int | None = None
    answer_source: Literal["llm", "calculation", "static"] = "llm"
    answer_source_label: str | None = None
    calculation_mode: str | None = None
    calculation_label: str | None = None


class QueryResponse(BaseModel):
    """POST /api/query 응답."""

    answer: str
    citations: list[Citation]
    metadata: QueryMetadata
