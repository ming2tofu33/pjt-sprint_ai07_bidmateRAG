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


def _build_store() -> CalculationStore:
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
                COL_CLEANED_CHARS: 5000,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: (
                    "배정예산: 850,000,000원\n"
                    "추정가격: 920,000,000원\n"
                    "예정가격: 910,000,000원\n"
                    "기초금액: 900,000,000원\n"
                    "계약금액: 830,000,000원\n"
                    "사업예산: 1,000,000,000원"
                ),
            },
            {
                COL_NOTICE_ID: "2024-002",
                COL_TITLE: "학사 시스템 기능개선",
                COL_BUDGET: 300_000_000,
                COL_AGENCY: "광주과학기술원",
                COL_PUBLIC_DATE: "2024-06-01 00:00:00",
                COL_BID_START: "2024-06-10 09:00:00",
                COL_BID_END: "2024-06-20 18:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "gist.hwp",
                COL_CANONICAL_FILE: "gist.hwp",
                COL_INGEST_FILE: "gist.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "광주과학기술원",
                COL_ORIGINAL_AGENCY: "광주과학기술원",
                COL_CLEANED_CHARS: 4200,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: (
                    "배정 예산 250,000,000원\n"
                    "추정 가격 280,000,000원\n"
                    "예정 가격 275,000,000원\n"
                    "기초 금액 270,000,000원\n"
                    "계약 금액 260,000,000원\n"
                    "사업예산: 300,000,000원"
                ),
            },
        ]
    )
    store = CalculationStore.create()
    store.rebuild_from_frame(frame)
    return store


def _chunk(doc_id: str, title: str, rank: int = 1, body: str | None = None) -> RetrievedChunk:
    text = body or f"{title} 관련 본문"
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
            metadata={"사업명": title, "파일명": doc_id},
        ),
    )


def test_calculation_engine_answers_budget_difference() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업 예산 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2),
            ],
        )

        assert result is not None
        assert result.mode == "budget_difference"
        assert "700,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_percentage_application() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="전체 예산의 30%를 사용한다고 가정하면 얼마야?",
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1)],
        )

        assert result is not None
        assert result.mode == "budget_percentage"
        assert "300,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_bid_window_days() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="입찰 기간은 며칠이야?",
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1)],
        )

        assert result is not None
        assert result.mode == "bid_window_days"
        assert "38일" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_budget_max() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="예산 규모가 가장 큰 사업은 뭐야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2),
            ],
        )

        assert result is not None
        assert result.mode == "budget_max"
        assert "차세대 포털 구축" in result.answer
        assert "1,000,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_budget_min() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="예산 규모가 가장 작은 사업은 뭐야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2),
            ],
        )

        assert result is not None
        assert result.mode == "budget_min"
        assert "학사 시스템 기능개선" in result.answer
        assert "300,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_budget_order_desc() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="예산 큰 순서대로 나열해줘",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2),
            ],
        )

        assert result is not None
        assert result.mode == "budget_order_desc"
        assert "1. 차세대 포털 구축" in result.answer
        assert "2. 학사 시스템 기능개선" in result.answer
    finally:
        store.close()


def test_calculation_engine_uses_explicit_budget_kind_when_requested() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업의 배정예산 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1, "배정예산: 850,000,000원"),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2, "배정 예산 250,000,000원"),
            ],
        )

        assert result is not None
        assert result.mode == "budget_difference"
        assert "배정예산" in result.answer
        assert "600,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_uses_store_budget_kind_before_chunk_fallback() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업의 배정예산 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1, "본문에는 일반 설명만 있음"),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2, "본문에는 일반 설명만 있음"),
            ],
        )

        assert result is not None
        assert result.mode == "budget_difference"
        assert "배정예산" in result.answer
        assert "600,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_returns_none_when_explicit_budget_kind_is_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                COL_NOTICE_ID: "2024-101",
                COL_TITLE: "테스트 사업 A",
                COL_BUDGET: 100_000_000,
                COL_AGENCY: "기관A",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "pdf",
                COL_FILE_NAME: "a.pdf",
                COL_CANONICAL_FILE: "a.pdf",
                COL_INGEST_FILE: "a.pdf",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "기관A",
                COL_ORIGINAL_AGENCY: "기관A",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "테스트",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "일반 설명만 있음",
            },
            {
                COL_NOTICE_ID: "2024-102",
                COL_TITLE: "테스트 사업 B",
                COL_BUDGET: 80_000_000,
                COL_AGENCY: "기관B",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "b.hwp",
                COL_CANONICAL_FILE: "b.hwp",
                COL_INGEST_FILE: "b.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "기관B",
                COL_ORIGINAL_AGENCY: "기관B",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "테스트",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "일반 설명만 있음",
            },
        ]
    )
    store = CalculationStore.create()
    store.rebuild_from_frame(frame)
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업의 추정가격 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("a.pdf", "테스트 사업 A", 1, "일반 설명만 있음"),
                _chunk("b.hwp", "테스트 사업 B", 2, "일반 설명만 있음"),
            ],
        )

        assert result is None
    finally:
        store.close()


def test_calculation_engine_answers_planned_price_difference() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업의 예정가격 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1, "본문에는 일반 설명만 있음"),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2, "본문에는 일반 설명만 있음"),
            ],
        )

        assert result is not None
        assert result.mode == "budget_difference"
        assert "예정가격" in result.answer
        assert "635,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_base_amount_difference() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="두 사업의 기초금액 차이는 얼마야?",
            retrieved_chunks=[
                _chunk("korea.pdf", "차세대 포털 구축", 1, "본문에는 일반 설명만 있음"),
                _chunk("gist.hwp", "학사 시스템 기능개선", 2, "본문에는 일반 설명만 있음"),
            ],
        )

        assert result is not None
        assert result.mode == "budget_difference"
        assert "기초금액" in result.answer
        assert "630,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_estimated_price_single_lookup() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="차세대 포털 구축 사업의 추정가격은 얼마야?",
            retrieved_chunks=[_chunk("korea.pdf", "차세대 포털 구축", 1, "본문에는 일반 설명만 있음")],
        )

        assert result is not None
        assert result.mode == "budget_single"
        assert "추정가격" in result.answer
        assert "920,000,000원" in result.answer
    finally:
        store.close()


def test_calculation_engine_answers_base_amount_single_lookup() -> None:
    store = _build_store()
    engine = CalculationEngine(store)
    try:
        result = engine.try_answer(
            question="학사 시스템 기능개선 사업의 기초금액은 얼마야?",
            retrieved_chunks=[_chunk("gist.hwp", "학사 시스템 기능개선", 1, "본문에는 일반 설명만 있음")],
        )

        assert result is not None
        assert result.mode == "budget_single"
        assert "기초금액" in result.answer
        assert "270,000,000원" in result.answer
    finally:
        store.close()
