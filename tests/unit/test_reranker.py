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
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"{chunk_id}-doc",
            text=f"{chunk_id} 본문",
            text_with_meta=f"{chunk_id} 본문",
            char_count=10,
            section=section,
            content_type=content_type,
            chunk_index=0,
            metadata={"발주 기관": agency, "사업명": project},
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
    assert [r.score for r in results] == [0.9, 0.5]
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


def test_rerank_with_boost_no_hint_preserves_order() -> None:
    chunks = [
        _make_chunk("c1", 0.9, section="사업개요"),
        _make_chunk("c2", 0.8, section="일반"),
    ]

    results = rerank_with_boost(chunks, query="사업 알려줘", section_hint=None)

    assert [r.chunk.chunk_id for r in results] == ["c1", "c2"]
    assert [r.rank for r in results] == [1, 2]
