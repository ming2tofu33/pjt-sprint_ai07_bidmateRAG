from bidmate_rag.generation.context_builder import build_context_block, build_numbered_context_block
from bidmate_rag.schema import Chunk, RetrievedChunk


def _make_retrieved_chunk(
    *,
    chunk_id: str,
    text: str,
    metadata: dict,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        text_with_meta=text,
        char_count=len(text),
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata=metadata,
    )
    return RetrievedChunk(rank=1, score=0.9, chunk=chunk)


def test_build_context_block_renders_source_and_metadata() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="첫 번째 청크",
            metadata={
                "사업명": "차세대 ERP 구축",
                "발주 기관": "한국가스공사",
                "파일명": "kgc_rfp.hwp",
                "사업 금액": 14107009000,
                "공개연도": 2024,
                "기관유형": "공기업/준정부기관",
                "사업도메인": "경영/행정",
            },
        )
    ]

    context = build_context_block(chunks)

    assert "[출처: 차세대 ERP 구축 | 한국가스공사 | kgc_rfp.hwp]" in context
    assert "사업 금액=14,107,009,000원" in context
    assert "공개연도=2024" in context
    assert "기관유형=공기업/준정부기관" in context
    assert "사업도메인=경영/행정" in context
    assert context.endswith("첫 번째 청크")


def test_build_context_block_preserves_human_readable_won_text() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="본문",
            metadata={"사업 금액": "약 3억원"},
        )
    ]

    context = build_context_block(chunks)

    assert "사업 금액=약 3억원" in context


def test_build_context_block_preserves_chunk_text_whitespace() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="  본문  ",
            metadata={"사업명": "사업", "발주 기관": "기관"},
        )
    ]

    context = build_context_block(chunks)

    assert context.endswith("  본문  ")
    assert "\n  본문  " in context


def test_build_context_block_omits_missing_and_nan_like_metadata() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="본문",
            metadata={
                "사업명": "  ",
                "발주 기관": "nan",
                "파일명": None,
                "사업 금액": float("nan"),
                "공개연도": "",
                "기관유형": "N/A",
                "사업도메인": "<NA>",
            },
        )
    ]

    context = build_context_block(chunks)

    assert "[출처:" not in context
    assert "사업명=" not in context
    assert "발주 기관=" not in context
    assert "파일명=" not in context
    assert "사업 금액=" not in context
    assert "공개연도=" not in context
    assert "기관유형=" not in context
    assert "사업도메인=" not in context
    assert context == "본문"


def test_build_context_block_respects_max_chars_budget() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="짧은 본문",
            metadata={"사업명": "사업 1", "발주 기관": "기관 1"},
        ),
        _make_retrieved_chunk(
            chunk_id="chunk-2",
            text="이 청크는 예산 때문에 포함되면 안 됩니다",
            metadata={"사업명": "사업 2", "발주 기관": "기관 2"},
        ),
    ]

    context = build_context_block(chunks, max_chars=40)

    assert "사업 1" in context
    assert "사업 2" not in context
    assert context == "[출처: 사업 1 | 기관 1]\n짧은 본문"


def test_build_numbered_context_block_groups_chunks_by_document() -> None:
    chunks = [
        _make_retrieved_chunk(
            chunk_id="chunk-1",
            text="첫 번째 문서의 첫 청크",
            metadata={
                "사업명": "사업 A",
                "발주 기관": "기관 A",
                "파일명": "a.hwp",
                "사업 금액": 100000000,
            },
        ),
        _make_retrieved_chunk(
            chunk_id="chunk-2",
            text="두 번째 문서의 첫 청크",
            metadata={
                "사업명": "사업 B",
                "발주 기관": "기관 B",
                "파일명": "b.hwp",
                "사업 금액": 200000000,
            },
        ),
        _make_retrieved_chunk(
            chunk_id="chunk-3",
            text="첫 번째 문서의 두 번째 청크",
            metadata={
                "사업명": "사업 A",
                "발주 기관": "기관 A",
                "파일명": "a.hwp",
                "사업 금액": 100000000,
            },
        ),
    ]

    context, used_indices = build_numbered_context_block(chunks, max_chars=2000)

    assert used_indices == [0, 1, 2]
    assert context.count("[문서: 사업 A | 기관 A | a.hwp]") == 1
    assert context.count("[문서: 사업 B | 기관 B | b.hwp]") == 1
    assert "[1] 섹션=요구사항" in context
    assert "[2] 섹션=요구사항" in context
    assert "[3] 섹션=요구사항" in context
    first_group_index = context.index("[문서: 사업 A | 기관 A | a.hwp]")
    second_group_index = context.index("[문서: 사업 B | 기관 B | b.hwp]")
    first_doc_second_chunk = context.index("첫 번째 문서의 두 번째 청크")
    assert first_group_index < first_doc_second_chunk < second_group_index
