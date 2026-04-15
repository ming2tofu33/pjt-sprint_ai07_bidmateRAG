"""FastAPI routes for BidMate web_api."""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.web_api.commands import COMMAND_REGISTRY, SlashCommand
from bidmate_rag.web_api.retrieval_helpers import web_query
from bidmate_rag.web_api.schemas import (
    Citation,
    DocumentContent,
    DocumentDetail,
    DocumentSummary,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    SlashCommandMeta,
)

router = APIRouter()

PDF_ROOT = Path("data/raw/PDF")


def _normalize_filename_stem(name: str) -> str:
    """파일명을 매칭 키로 정규화.

    메타데이터의 `파일명` 컬럼은 원본 HWP/PDF 파일명을 담고 있는데,
    `data/raw/PDF/` 디렉토리의 실제 파일명과 미세하게 달라지는 케이스가
    있다 (공백 ↔ `+`, NBSP, 뒷공백/마침표). NFC 정규화 + 구분자 통합으로
    98/100 전건 매칭이 된다.
    """
    stem = Path(name).stem
    stem = unicodedata.normalize("NFC", stem)
    stem = stem.replace("+", " ").replace("\u00a0", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.rstrip(" .")


def _build_pdf_index() -> dict[str, Path]:
    """PDF 디렉토리의 파일들을 정규화 키로 색인."""
    if not PDF_ROOT.exists():
        return {}
    return {_normalize_filename_stem(p.name): p for p in PDF_ROOT.glob("*.pdf")}


def _format_budget_label(amount: float) -> str:
    """숫자 금액을 한국어 라벨로."""
    if not amount or (isinstance(amount, float) and math.isnan(amount)):
        return "-"
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.1f}억"
    if amount >= 10_000:
        return f"{amount / 10_000:.0f}만"
    return f"{int(amount)}"


def _resolve_doc_id(row: pd.Series) -> str:
    """Chunker의 `_chunk_doc_id`와 일치: 공고 번호 → 파일명 폴백.

    18/100 문서가 공고 번호가 없어서 파일명으로 저장됨. 빈 id는 frontend에서
    React key 충돌을 유발하므로 반드시 non-empty를 반환해야 한다.
    """
    notice_id = str(row.get("공고 번호") or "").strip()
    if notice_id and notice_id.lower() not in ("nan", "none"):
        return notice_id
    return str(row.get("파일명") or "").strip()


def _row_to_summary(row: pd.Series) -> DocumentSummary:
    budget = float(row.get("사업 금액") or 0)
    return DocumentSummary(
        id=_resolve_doc_id(row),
        title=str(row.get("사업명", "")),
        agency=str(row.get("발주 기관", "")),
        agency_type=str(row.get("기관유형", "")),
        domain=str(row.get("사업도메인", "")),
        budget=budget,
        budget_label=_format_budget_label(budget),
        deadline=str(row.get("입찰 참여 마감일", "")) or None,
        char_count=int(row.get("정제_글자수") or 0),
    )


@router.get("/documents")
def list_documents(request: Request) -> dict[str, Any]:
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    if frame.empty:
        return {"documents": [], "total": 0}
    documents = [_row_to_summary(row).model_dump() for _, row in frame.iterrows()]
    return {"documents": documents, "total": len(documents)}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, request: Request) -> dict[str, Any]:
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    # 공고 번호 또는 파일명으로 매칭 (일부 문서는 공고 번호 없음)
    notice_match = frame["공고 번호"].astype(str) == doc_id
    filename_match = (
        frame["파일명"].astype(str) == doc_id if "파일명" in frame.columns else False
    )
    match = frame[notice_match | filename_match]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    row = match.iloc[0]
    summary = _row_to_summary(row)
    summary_oneline = str(row.get("사업 요약") or "") or None
    quick_facts = [
        {"label": "발주기관", "value": summary.agency or "-"},
        {"label": "사업금액", "value": summary.budget_label},
        {"label": "마감일", "value": summary.deadline or "-"},
        {"label": "도메인", "value": summary.domain or "-"},
        {"label": "문서크기", "value": f"{summary.char_count:,}자"},
    ]
    detail = DocumentDetail(
        **summary.model_dump(),
        summary_oneline=summary_oneline,
        quick_facts=quick_facts,
    )
    return detail.model_dump()


@router.get("/documents/{doc_id}/pdf")
def get_document_pdf(doc_id: str, request: Request) -> FileResponse:
    """문서 원본 PDF 스트리밍 (iframe 미리보기용).

    metadata_store에서 `파일명`을 찾아 `data/raw/PDF/`의 해당 PDF 파일을
    `FileResponse`로 응답한다. 파일명 정규화(공백/+/NBSP/NFC)로 매칭.
    """
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    notice_match = frame["공고 번호"].astype(str) == doc_id
    filename_match = (
        frame["파일명"].astype(str) == doc_id if "파일명" in frame.columns else False
    )
    match = frame[notice_match | filename_match]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    row = match.iloc[0]
    meta_filename = str(row.get("파일명") or "")
    if not meta_filename:
        raise HTTPException(status_code=404, detail="no 파일명 in metadata")

    normalized = _normalize_filename_stem(meta_filename)
    pdf_index = _build_pdf_index()
    pdf_path = pdf_index.get(normalized)
    if pdf_path is None or not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"pdf not found for '{meta_filename}' (normalized: '{normalized}')",
        )
    # HTTP 헤더는 latin-1만 허용하므로 한글 파일명은 RFC 5987 형식으로 인코딩.
    # ASCII fallback + filename*= UTF-8 쌍을 함께 보내면 모든 브라우저가 해석 가능.
    disposition = (
        f"inline; filename=\"document.pdf\"; "
        f"filename*=UTF-8''{quote(pdf_path.name)}"
    )
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )


@router.get("/documents/{doc_id}/content")
def get_document_content(doc_id: str, request: Request) -> dict[str, Any]:
    """문서 미리보기용: 정제된 마크다운 본문 반환.

    `본문_정제` 컬럼(6종 노이즈 정제 완료)을 그대로 내려준다.
    원본 kordoc 출력(`본문_마크다운`)이 아니라 cleaner를 거친 버전을 쓴다.
    """
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    notice_match = frame["공고 번호"].astype(str) == doc_id
    filename_match = (
        frame["파일명"].astype(str) == doc_id if "파일명" in frame.columns else False
    )
    match = frame[notice_match | filename_match]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    row = match.iloc[0]
    markdown = str(row.get("본문_정제") or row.get("본문_마크다운") or "")
    content = DocumentContent(
        doc_id=_resolve_doc_id(row),
        title=str(row.get("사업명", "")),
        markdown=markdown,
        char_count=len(markdown),
    )
    return content.model_dump()


@router.get("/commands")
def list_commands() -> dict[str, Any]:
    commands = [
        SlashCommandMeta(
            id=cmd.id,
            label=cmd.label,
            description=cmd.description,
            icon=cmd.icon,
            requires_doc=cmd.requires_doc,
            requires_multi_doc=cmd.requires_multi_doc,
        ).model_dump()
        for cmd in COMMAND_REGISTRY.values()
    ]
    return {"commands": commands}


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        citations.append(
            Citation(
                id=idx,
                doc_id=chunk.doc_id,
                doc_title=str(chunk.metadata.get("사업명", "")),
                section=chunk.section or "",
                content_type=chunk.content_type or "text",
                text=chunk.text[:500],
                score=float(rc.score),
            )
        )
    return citations


def _validate_command(cmd: SlashCommand, mentioned_doc_ids: list[str]) -> None:
    if cmd.requires_doc and not mentioned_doc_ids:
        raise HTTPException(status_code=400, detail=f"{cmd.label}는 문서 멘션이 필요합니다")
    if cmd.requires_multi_doc and len(mentioned_doc_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"{cmd.label}는 2개 이상 문서 멘션이 필요합니다",
        )


def _static_response(cmd: SlashCommand) -> QueryResponse:
    payload = cmd.static_payload or {}
    return QueryResponse(
        answer=payload.get("answer", ""),
        citations=[],
        metadata=QueryMetadata(
            model="-",
            token_usage={},
            latency_ms=0.0,
            cost_usd=0.0,
            command_applied=cmd.id,
            filter_applied=None,
            retrieval_strategy="static",
            per_doc_k=None,
        ),
    )


def _result_to_response(
    result: GenerationResult,
    cmd: SlashCommand | None,
    filter_applied: dict | None,
    strategy: str,
    per_doc_k: int | None,
) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        citations=_build_citations(result.retrieved_chunks),
        metadata=QueryMetadata(
            model=result.llm_model,
            token_usage=result.token_usage,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            command_applied=cmd.id if cmd else None,
            filter_applied=filter_applied,
            retrieval_strategy=strategy,
            per_doc_k=per_doc_k,
        ),
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    cmd = COMMAND_REGISTRY.get(req.command) if req.command else None

    # 1. 알 수 없는 커맨드
    if req.command and cmd is None:
        raise HTTPException(status_code=400, detail=f"unknown command: {req.command}")

    # 2. validation
    if cmd:
        _validate_command(cmd, req.mentioned_doc_ids)

    # 3. static response (/도움말, /초기화)
    if cmd and cmd.static_response:
        return _static_response(cmd)

    # 4. 쿼리 증강
    augmented_query = req.question
    if cmd and cmd.query_augmentation:
        augmented_query = f"{req.question} {cmd.query_augmentation}".strip()

    # 5. 시스템 프롬프트
    system_prompt = cmd.system_prompt if cmd and cmd.system_prompt else None

    # 6. top_k
    top_k = cmd.top_k if cmd else req.top_k

    # 7. 통합 RAG 경로 — `web_query`가 멘션 0/1/N+ 모두 처리
    result = web_query(
        question=req.question,
        augmented_query=augmented_query,
        mentioned_doc_ids=req.mentioned_doc_ids,
        provider_config=req.provider_config,
        chunking_config=req.chunking_config,
        system_prompt=system_prompt,
        top_k=top_k,
        max_context_chars=req.max_context_chars,
    )

    # 8. metadata 라벨링 (멘션 개수에 따라)
    mention_count = len(req.mentioned_doc_ids)
    if mention_count >= 2:
        strategy = "per_doc_split"
        filter_applied = {"doc_id": {"$in": req.mentioned_doc_ids}}
        per_doc_k_val = max(top_k // mention_count, 3) + 2
    elif mention_count == 1:
        strategy = "single"
        filter_applied = {"doc_id": req.mentioned_doc_ids[0]}
        per_doc_k_val = None
    else:
        strategy = "single"
        filter_applied = None
        per_doc_k_val = None

    return _result_to_response(
        result,
        cmd,
        filter_applied=filter_applied,
        strategy=strategy,
        per_doc_k=per_doc_k_val,
    )
