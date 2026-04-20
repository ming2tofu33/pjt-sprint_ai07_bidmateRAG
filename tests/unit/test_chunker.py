from bidmate_rag.preprocessing.chunker import (
    chunk_document,
    split_by_headers,
    split_table_with_headers,
)


def test_split_by_headers_merges_small_sections() -> None:
    text = "# 개요\n짧음\n\n# 요구사항\n" + ("상세 요구사항 " * 60)

    sections = split_by_headers(text, min_size=50)

    assert len(sections) == 1
    assert "짧음" in sections[0]["text"]
    assert sections[0]["section"] == "요구사항"


def test_split_table_with_headers_repeats_header_when_table_is_large() -> None:
    text = "\n".join(
        [
            "표 설명",
            "| 항목 | 값 |",
            "| --- | --- |",
            "| A | 1 |",
            "| B | 2 |",
            "| C | 3 |",
            "| D | 4 |",
        ]
    )

    chunks = split_table_with_headers(text, max_size=40)

    assert len(chunks) >= 2
    assert all("| 항목 | 값 |" in chunk for chunk in chunks)


def test_chunk_document_adds_meta_prefix_and_preserves_table_type() -> None:
    text = (
        "# 요구사항\n"
        + ("문장 " * 300)
        + "\n\n# 예산\n| 항목 | 값 |\n| --- | --- |\n| 인건비 | 1억 |\n"
    )
    metadata = {"파일명": "sample.hwp", "사업명": "샘플 사업", "발주 기관": "샘플 기관"}

    chunks = chunk_document(text, metadata, chunk_size=300, chunk_overlap=50, max_table_size=80)

    assert len(chunks) >= 2
    assert chunks[0].text_with_meta.startswith("[발주기관: 샘플 기관 | 사업명: 샘플 사업]")
    assert any(chunk.content_type == "table" for chunk in chunks)
    assert all(chunk.metadata["파일명"] == "sample.hwp" for chunk in chunks)
