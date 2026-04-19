from bidmate_rag.retrieval.reranker import (
    build_reranker_text,
    cross_encoder_rerank,
    rerank_with_boost,
)
from bidmate_rag.schema import Chunk, RetrievedChunk


def _make_chunk(
    chunk_id: str,
    score: float,
    *,
    agency: str = "",
    project: str = "",
    section: str = "",
    content_type: str = "text",
    file_name: str | None = None,
    doc_id: str | None = None,
    resolved_agency: str = "",
    original_agency: str = "",
    text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id or f"{chunk_id}-doc",
            text=text or f"{chunk_id} 본문",
            text_with_meta=text or f"{chunk_id} 본문",
            char_count=10,
            section=section,
            content_type=content_type,
            chunk_index=0,
            metadata={
                "발주 기관": agency,
                "사업명": project,
                "파일명": file_name or f"{chunk_id}.hwp",
                "resolved_agency": resolved_agency,
                "original_agency": original_agency,
            },
        ),
    )


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.last_pairs: list[list[str]] | None = None

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.last_pairs = pairs
        return self.scores


# ── build_reranker_text ──


def test_build_reranker_text_with_agency_and_project() -> None:
    rc = _make_chunk("c1", 0.9, agency="국민연금공단", project="차세대 포털")
    result = build_reranker_text(rc)
    assert result == "[발주기관: 국민연금공단 | 사업명: 차세대 포털]\nc1 본문"


def test_build_reranker_text_no_metadata() -> None:
    rc = _make_chunk("c1", 0.9)
    result = build_reranker_text(rc)
    assert result == "c1 본문"


# ── cross_encoder_rerank ──


def test_cross_encoder_rerank_sorts_by_score_and_trims() -> None:
    chunks = [
        _make_chunk("c1", 0.5, agency="A"),
        _make_chunk("c2", 0.4, agency="B"),
        _make_chunk("c3", 0.3, agency="C"),
    ]
    reranker = FakeReranker([0.1, 0.9, 0.5])

    results = cross_encoder_rerank(reranker, "질문", chunks, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["c2", "c3"]
    assert [r.score for r in results] == [0.4, 0.3]
    assert [r.rerank_score for r in results] == [0.9, 0.5]
    assert [r.rank for r in results] == [1, 2]


def test_cross_encoder_rerank_returns_as_is_when_no_reranker() -> None:
    chunks = [_make_chunk("c1", 0.9)]
    results = cross_encoder_rerank(None, "질문", chunks, top_k=1)
    assert results is chunks


# ── rerank_with_boost ──


def test_rerank_with_boost_section_match_promotes_lower_score() -> None:
    chunks = [
        _make_chunk("overview", 0.91, section="사업개요"),
        _make_chunk("budget", 0.80, section="예산", content_type="table"),
    ]

    results = rerank_with_boost(chunks, query="예산 표를 알려줘", section_hint="예산")

    assert results[0].chunk.chunk_id == "budget"
    assert results[0].rank == 1


def test_rerank_with_boost_default_section_weight_promotes_matching_section() -> None:
    chunks = [
        _make_chunk("overview", 0.85, section="사업개요"),
        _make_chunk("budget", 0.70, section="예산"),
    ]

    results = rerank_with_boost(chunks, query="예산 알려줘", section_hint="예산")

    assert results[0].chunk.chunk_id == "budget"
    assert results[0].rank == 1


def test_rerank_with_boost_uses_chunk_text_when_section_field_is_empty() -> None:
    chunks = [
        _make_chunk("overview", 0.88, section="사업개요", text="사업 개요와 일반 설명"),
        _make_chunk(
            "security",
            0.78,
            section="",
            text="SER-002 보안 요구사항 USB 반입 반출 통제 규정",
        ),
    ]

    results = rerank_with_boost(chunks, query="USB 반입 반출 규정 알려줘", section_hint="보안 요구사항")

    assert results[0].chunk.chunk_id == "security"
    assert results[0].rank == 1


def test_rerank_with_boost_cap_limits_total_bonus() -> None:
    """섹션+테이블 부스트 합산이 max_total을 넘지 않아야 한다."""
    chunks = [
        _make_chunk("top", 0.90, section="일반"),
        _make_chunk("boosted", 0.78, section="예산", content_type="table"),
    ]
    cfg = {"section": 0.12, "table": 0.08, "max_total": 0.10}

    results = rerank_with_boost(chunks, query="예산 표", section_hint="예산", boost_config=cfg)

    # 0.78 + min(0.12+0.08, 0.10) = 0.88 < 0.90 → top이 여전히 1위
    assert results[0].chunk.chunk_id == "top"


def test_rerank_with_boost_no_hint_preserves_order() -> None:
    chunks = [
        _make_chunk("c1", 0.9, section="사업개요"),
        _make_chunk("c2", 0.8, section="일반"),
    ]

    results = rerank_with_boost(chunks, query="사업 알려줘", section_hint=None)

    assert [r.chunk.chunk_id for r in results] == ["c1", "c2"]
    assert [r.rank for r in results] == [1, 2]


def test_rerank_with_boost_promotes_metadata_match() -> None:
    chunks = [
        _make_chunk("generic", 0.88, agency="조달청", project="일반 사업", section="일반"),
        _make_chunk(
            "target",
            0.80,
            agency="국민연금공단",
            project="이러닝시스템 운영 용역",
            section="일반",
            file_name="국민연금공단_이러닝시스템 운영 용역.hwp",
            doc_id="국민연금공단_이러닝시스템 운영 용역.hwp",
        ),
    ]

    results = rerank_with_boost(
        chunks,
        query="국민연금공단 이러닝시스템 운영 용역 요구사항 알려줘",
        section_hint=None,
    )

    assert results[0].chunk.chunk_id == "target"
    assert results[0].rank == 1


def test_rerank_with_boost_uses_agency_fallback_metadata_fields() -> None:
    chunks = [
        _make_chunk("generic", 0.88, agency="조달청", project="일반 사업", section="일반"),
        _make_chunk(
            "target",
            0.80,
            agency="",
            resolved_agency="국민연금공단",
            original_agency="국민연금공단",
            project="일반 사업",
            section="일반",
        ),
    ]

    results = rerank_with_boost(
        chunks,
        query="국민연금공단 사업 요구사항 알려줘",
        section_hint=None,
    )

    assert results[0].chunk.chunk_id == "target"
