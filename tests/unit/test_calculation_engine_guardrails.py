import pandas as pd

from bidmate_rag.generation.calculation_engine import CalculationEngine
from bidmate_rag.schema import Chunk, RetrievedChunk
from bidmate_rag.storage.calculation_store import (
    COL_AGENCY,
    COL_BID_END,
    COL_BID_START,
    COL_BODY_MARKDOWN,
    COL_BUDGET,
    COL_CANONICAL_FILE,
    COL_CLEANED_CHARS,
    COL_CLEANED_TEXT,
    COL_DOMAIN,
    COL_FILE_NAME,
    COL_FILE_TYPE,
    COL_INGEST_ENABLED,
    COL_INGEST_FILE,
    COL_IS_DUPLICATE,
    COL_NOTICE_ID,
    COL_ORIGINAL_AGENCY,
    COL_PUBLIC_DATE,
    COL_PUBLIC_YEAR,
    COL_RESOLVED_AGENCY,
    COL_TITLE,
    CalculationStore,
)


def _build_guardrail_store() -> CalculationStore:
    frame = pd.DataFrame(
        [
            {
                COL_NOTICE_ID: "2024-001",
                COL_TITLE: "차세대 포털 구축",
                COL_BUDGET: 1_000_000_000,
                COL_AGENCY: "고려대학교",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-08-12 11:00:00",
                COL_FILE_TYPE: "pdf",
                COL_FILE_NAME: "korea.pdf",
                COL_CANONICAL_FILE: "korea.pdf",
                COL_INGEST_FILE: "korea.pdf",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "고려대학교",
                COL_ORIGINAL_AGENCY: "고려대학교",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: (
                    "배정예산 850,000,000원\n"
                    "추정가격 920,000,000원\n"
                    "계약금액 830,000,000원\n"
                    "사업예산 1,000,000,000원"
                ),
            },
            {
                COL_NOTICE_ID: "2024-002",
                COL_TITLE: "RCMS 연계 모듈 변경 사업",
                COL_BUDGET: 100_000_000,
                COL_AGENCY: "광주과학기술원",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "rcms.hwp",
                COL_CANONICAL_FILE: "rcms.hwp",
                COL_INGEST_FILE: "rcms.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "광주과학기술원",
                COL_ORIGINAL_AGENCY: "광주과학기술원",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "사업예산 100,000,000원",
            },
            {
                COL_NOTICE_ID: "2024-003",
                COL_TITLE: "학사시스템 기능개선 사업",
                COL_BUDGET: 200_000_000,
                COL_AGENCY: "광주과학기술원",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "academic.hwp",
                COL_CANONICAL_FILE: "academic.hwp",
                COL_INGEST_FILE: "academic.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "광주과학기술원",
                COL_ORIGINAL_AGENCY: "광주과학기술원",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "사업예산 200,000,000원",
            },
            {
                COL_NOTICE_ID: "2024-004",
                COL_TITLE: "의료기기산업 기능개선 사업",
                COL_BUDGET: 50_000_000,
                COL_AGENCY: "한국보건산업진흥원",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "medical.hwp",
                COL_CANONICAL_FILE: "medical.hwp",
                COL_INGEST_FILE: "medical.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "한국보건산업진흥원",
                COL_ORIGINAL_AGENCY: "한국보건산업진흥원",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "보건",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "사업예산 50,000,000원",
            },
        ]
    )
    store = CalculationStore.create()
    store.rebuild_from_frame(frame)
    return store


def _chunk(doc_id: str, title: str, rank: int = 1) -> RetrievedChunk:
    text = f"{title} 본문"
    return RetrievedChunk(
        rank=rank,
        score=1.0 / rank,
        chunk=Chunk(
            chunk_id=f"{doc_id}-chunk-{rank}",
            doc_id=doc_id,
            text=text,
            text_with_meta=text,
            char_count=len(text),
            chunk_index=rank - 1,
            metadata={"파일명": doc_id, "사업명": title},
        ),
    )


def test_calculation_engine_does_not_route_bid_bond_page_question_to_window_days() -> None:
    store = _build_guardrail_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="입찰보증금 납부 기한과 구체적인 납부계좌 번호는 제안요청서의 몇 페이지에 나와 있습니까?",
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1)],
        )
        assert result is None
    finally:
        store.close()


def test_calculation_engine_does_not_route_security_question_to_window_days() -> None:
    store = _build_guardrail_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question=(
                "통합정보시스템 고도화 사업 입찰과 관련하여 참여인력 개별 보안서약서 대신 "
                "대표이사 명의의 통합 보안확약서 1건으로 갈음할 수 있습니까?"
            ),
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1)],
        )
        assert result is None
    finally:
        store.close()


def test_calculation_engine_does_not_route_ratio_question_to_budget_single() -> None:
    store = _build_guardrail_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="차세대 포털 구축 사업에서 지급되는 선금은 계약금액의 몇 프로입니까?",
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1)],
        )
        assert result is None
    finally:
        store.close()


def test_calculation_engine_filters_budget_candidates_by_quoted_targets() -> None:
    store = _build_guardrail_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question=(
                '"RCMS 연계 모듈 변경 사업", "학사시스템 기능개선 사업" '
                "두 사업 중 예산 규모가 가장 작은 사업은 무엇입니까?"
            ),
            retrieved_chunks=[
                _chunk("rcms.hwp", "RCMS 연계 모듈 변경 사업", 1),
                _chunk("academic.hwp", "학사시스템 기능개선 사업", 2),
                _chunk("medical.hwp", "의료기기산업 기능개선 사업", 3),
            ],
        )

        assert result is not None
        assert result.mode == "budget_min"
        assert "RCMS 연계 모듈 변경 사업" in result.answer
        assert "의료기기산업 기능개선 사업" not in result.answer
    finally:
        store.close()
