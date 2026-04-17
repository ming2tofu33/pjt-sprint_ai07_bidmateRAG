"""Integration tests for web_api routes (TestClient based)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from bidmate_rag.storage.metadata_store import MetadataStore
from bidmate_rag.web_api.routes import router


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "공고 번호": "20240000001",
                "파일명": "doc-1.hwp",
                "사업명": "학사 시스템 고도화",
                "발주 기관": "한영대학",
                "기관유형": "대학교",
                "사업도메인": "교육/학습",
                "사업 금액": 130000000.0,
                "입찰 참여 마감일": "2024-12-15",
                "정제_글자수": 45230,
                "사업 요약": "학사 시스템 고도화 사업",
            },
            {
                "공고 번호": "20240000002",
                "파일명": "doc-2.hwp",
                "사업명": "이러닝 시스템",
                "발주 기관": "국민연금공단",
                "기관유형": "공기업/준정부기관",
                "사업도메인": "교육/학습",
                "사업 금액": 1230000000.0,
                "입찰 참여 마감일": "2024-11-30",
                "정제_글자수": 60000,
                "사업 요약": "",
            },
        ]
    )


@pytest.fixture
def client(sample_frame):
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.metadata_store = MetadataStore(sample_frame)
        app.state.web_config = {
            "provider_config": "openai_gpt5mini",
            "chunking_config": None,
            "top_k": 5,
            "max_context_chars": 8000,
        }
        yield

    test_app = FastAPI(title="BidMate Web API (Test)", version="0.1.0", lifespan=test_lifespan)
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.include_router(router, prefix="/api")

    with TestClient(test_app) as test_client:
        yield test_client


def test_get_documents_returns_list(client) -> None:
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["documents"]) == 2
    first = data["documents"][0]
    assert first["id"] == "20240000001"
    assert first["agency"] == "한영대학"
    assert first["budget_label"].endswith("억")


def test_get_document_detail(client) -> None:
    response = client.get("/api/documents/20240000001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "20240000001"
    assert len(data["quick_facts"]) == 5
    labels = [f["label"] for f in data["quick_facts"]]
    assert "발주기관" in labels
    assert "사업금액" in labels


def test_get_document_detail_not_found(client) -> None:
    response = client.get("/api/documents/99999999999")
    assert response.status_code == 404


def test_get_commands_returns_twelve(client) -> None:
    response = client.get("/api/commands")
    assert response.status_code == 200
    data = response.json()
    assert len(data["commands"]) == 12
    ids = {c["id"] for c in data["commands"]}
    assert "비교" in ids
    compare = next(c for c in data["commands"] if c["id"] == "비교")
    assert compare["requires_multi_doc"] is True


from unittest.mock import patch

from bidmate_rag.schema import GenerationResult, Chunk, RetrievedChunk


def _make_generation_result(answer: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    return GenerationResult(
        question_id="q-test",
        question="요구사항 알려줘",
        scenario="scenario_b",
        run_id="run-test",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        answer=answer,
        retrieved_chunk_ids=[c.chunk.chunk_id for c in chunks],
        retrieved_doc_ids=list({c.chunk.doc_id for c in chunks}),
        retrieved_chunks=chunks,
        latency_ms=1234.5,
        token_usage={"prompt": 100, "completion": 50, "total": 150, "cached": 0},
        cost_usd=0.001,
        context="...",
    )


def _make_chunk(chunk_id: str, doc_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=0.9 - 0.1 * rank,
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=f"chunk text {rank}",
            text_with_meta=f"[doc={doc_id}] chunk {rank}",
            char_count=50,
            section="Ⅳ 제안요청",
            content_type="text",
            chunk_index=rank,
            metadata={"사업명": doc_id},
        ),
    )


def test_post_query_no_mentions_no_command(client) -> None:
    chunks = [_make_chunk("c1", "doc-1.hwp", 1), _make_chunk("c2", "doc-1.hwp", 2)]
    fake_result = _make_generation_result("## 답변\n본문 [1][2]", chunks)

    with patch("bidmate_rag.web_api.routes.web_query", return_value=fake_result):
        response = client.post(
            "/api/query",
            json={
                "question": "요구사항 알려줘",
                "provider_config": "openai_gpt5mini",
                "chunking_config": "chunking_1000_150",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("## 답변")
    assert len(data["citations"]) == 2
    assert data["citations"][0]["id"] == 1
    assert data["metadata"]["retrieval_strategy"] == "single"
    assert data["metadata"]["command_applied"] is None


def test_post_query_with_single_mention(client) -> None:
    chunks = [_make_chunk("c1", "20240000001", 1)]
    fake_result = _make_generation_result("답변", chunks)

    captured = {}
    def _fake_web_query(**kwargs):
        captured.update(kwargs)
        return fake_result

    with patch("bidmate_rag.web_api.routes.web_query", side_effect=_fake_web_query):
        response = client.post(
            "/api/query",
            json={
                "question": "사업 개요",
                "provider_config": "openai_gpt5mini",
                "chunking_config": "chunking_1000_150",
                "mentioned_doc_ids": ["20240000001"],
            },
        )
    assert response.status_code == 200
    # 멘션된 doc_id가 web_query에 전달됐는지 확인
    assert captured["mentioned_doc_ids"] == ["20240000001"]
    data = response.json()
    assert data["metadata"]["filter_applied"] == {"doc_id": "20240000001"}


def test_post_query_forwards_history_to_web_query(client) -> None:
    chunks = [_make_chunk("c1", "20240000001", 1)]
    fake_result = _make_generation_result("답변", chunks)

    captured = {}

    def _fake_web_query(**kwargs):
        captured.update(kwargs)
        return fake_result

    history = [
        {"role": "user", "content": "국민연금공단 사업 알려줘"},
        {"role": "assistant", "content": "차세대 ERP 사업입니다."},
    ]
    with patch("bidmate_rag.web_api.routes.web_query", side_effect=_fake_web_query):
        response = client.post(
            "/api/query",
            json={
                "question": "그 사업 일정은?",
                "provider_config": "openai_gpt5mini",
                "history": history,
            },
        )

    assert response.status_code == 200
    assert captured["chat_history"] == history


def test_post_query_with_command_augments_query(client) -> None:
    chunks = [_make_chunk("c1", "doc-1.hwp", 1)]
    fake_result = _make_generation_result("표 형식 답변", chunks)

    captured = {}
    def _fake_web_query(**kwargs):
        captured.update(kwargs)
        return fake_result

    with patch("bidmate_rag.web_api.routes.web_query", side_effect=_fake_web_query):
        response = client.post(
            "/api/query",
            json={
                "question": "알려줘",
                "provider_config": "openai_gpt5mini",
                "chunking_config": "chunking_1000_150",
                "mentioned_doc_ids": ["doc-1.hwp"],
                "command": "요구사항",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["command_applied"] == "요구사항"
    # 시스템 프롬프트가 커맨드별로 바뀌었는지
    assert captured["system_prompt"] is not None
    assert "기능 요구사항" in captured["system_prompt"]
    # top_k가 커맨드 기본값(12)로 바뀌었는지
    assert captured["top_k"] == 12


def test_post_query_validates_requires_multi_doc(client) -> None:
    response = client.post(
        "/api/query",
        json={
            "question": "비교해줘",
            "provider_config": "openai_gpt5mini",
            "chunking_config": "chunking_1000_150",
            "mentioned_doc_ids": ["doc-1.hwp"],
            "command": "비교",
        },
    )
    assert response.status_code == 400
    assert "2개 이상" in response.json()["detail"]


def test_post_query_static_help_command(client) -> None:
    response = client.post(
        "/api/query",
        json={
            "question": "",
            "provider_config": "openai_gpt5mini",
            "chunking_config": "chunking_1000_150",
            "command": "도움말",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["retrieval_strategy"] == "static"
    assert "/요약" in data["answer"]


def test_post_query_unknown_command(client) -> None:
    response = client.post(
        "/api/query",
        json={
            "question": "알려줘",
            "provider_config": "openai_gpt5mini",
            "chunking_config": "chunking_1000_150",
            "command": "없는커맨드",
        },
    )
    assert response.status_code == 400
    assert "unknown command" in response.json()["detail"]
