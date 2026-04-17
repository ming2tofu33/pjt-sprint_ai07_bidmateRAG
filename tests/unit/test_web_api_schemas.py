"""Unit tests for web_api Pydantic schemas."""

from __future__ import annotations

from bidmate_rag.web_api.schemas import (
    Citation,
    DocumentDetail,
    DocumentSummary,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    SlashCommandMeta,
)


def test_document_summary_roundtrip() -> None:
    summary = DocumentSummary(
        id="doc-1.hwp",
        title="한영대학교 학사정보시스템 고도화",
        agency="한영대학",
        agency_type="대학교",
        domain="교육/학습",
        budget=130000000.0,
        budget_label="1.3억",
        deadline="2024-12-15",
        char_count=45230,
    )
    data = summary.model_dump()
    assert data["budget_label"] == "1.3억"
    revived = DocumentSummary.model_validate(data)
    assert revived.id == "doc-1.hwp"


def test_query_request_defaults() -> None:
    """Optional 필드는 None으로 폴백 — 실제 값은 configs/web.yaml에서 채워진다."""
    req = QueryRequest(question="요구사항 알려줘")
    assert req.mentioned_doc_ids == []
    assert req.history == []
    assert req.command is None
    assert req.provider_config is None
    assert req.chunking_config is None
    assert req.top_k is None
    assert req.max_context_chars is None


def test_query_response_contains_citations() -> None:
    citation = Citation(
        id=1,
        doc_id="doc-1.hwp",
        doc_title="학사 시스템",
        section="Ⅳ 제안요청 내용",
        content_type="text",
        text="사용자 인증 ...",
        score=0.87,
    )
    metadata = QueryMetadata(
        model="gpt-5-mini",
        token_usage={"prompt": 100, "completion": 50, "total": 150, "cached": 0},
        latency_ms=1234.5,
        cost_usd=0.001,
        command_applied=None,
        filter_applied={},
        retrieval_strategy="single",
        per_doc_k=None,
    )
    response = QueryResponse(
        answer="답변",
        citations=[citation],
        metadata=metadata,
    )
    data = response.model_dump()
    assert len(data["citations"]) == 1
    assert data["metadata"]["retrieval_strategy"] == "single"


def test_document_detail_quick_facts_list() -> None:
    detail = DocumentDetail(
        id="doc-1.hwp",
        title="학사 시스템",
        agency="한영대학",
        agency_type="대학교",
        domain="교육/학습",
        budget=130000000.0,
        budget_label="1.3억",
        deadline="2024-12-15",
        char_count=45230,
        summary_oneline=None,
        quick_facts=[
            {"label": "발주기관", "value": "한영대학"},
            {"label": "사업금액", "value": "1.3억"},
        ],
    )
    assert detail.quick_facts[0]["label"] == "발주기관"


def test_slash_command_meta_serialization() -> None:
    cmd = SlashCommandMeta(
        id="요약",
        label="/요약",
        description="사업 개요를 bullet로 요약",
        icon="📋",
        requires_doc=False,
        requires_multi_doc=False,
    )
    assert cmd.model_dump()["id"] == "요약"
