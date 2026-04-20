import pandas as pd

from bidmate_rag.storage.document_quality import build_document_quality_report


def test_build_document_quality_report_marks_empty_and_duplicate_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "파일명": "dup-source.hwp",
                "ingest_file": "canonical.hwp",
                "canonical_file": "canonical.hwp",
                "duplicate_group_id": "DUP-001",
                "is_duplicate": True,
                "ingest_enabled": False,
                "original_agency": "별칭기관",
                "resolved_agency": "대표기관",
                "파일형식": "hwp",
                "공개연도": 2024,
                "기관유형": "공공기관",
                "사업도메인": "행정",
                "본문_글자수": 0,
                "정제_글자수": 0,
            },
            {
                "파일명": "short.docx",
                "ingest_file": "short.docx",
                "canonical_file": "short.docx",
                "duplicate_group_id": "",
                "is_duplicate": False,
                "ingest_enabled": True,
                "original_agency": "테스트기관",
                "resolved_agency": "테스트기관",
                "파일형식": "docx",
                "공개연도": 2024,
                "기관유형": "공공기관",
                "사업도메인": "교육",
                "본문_글자수": 120,
                "정제_글자수": 120,
            },
            {
                "파일명": "good.pdf",
                "ingest_file": "good.pdf",
                "canonical_file": "good.pdf",
                "duplicate_group_id": "",
                "is_duplicate": False,
                "ingest_enabled": True,
                "original_agency": "테스트기관",
                "resolved_agency": "테스트기관",
                "파일형식": "pdf",
                "공개연도": 2024,
                "기관유형": "공공기관",
                "사업도메인": "교육",
                "본문_글자수": 2000,
                "정제_글자수": 1800,
            },
        ]
    )

    report = build_document_quality_report(frame, short_text_threshold=500)

    dup_row = report[report["파일명"] == "dup-source.hwp"].iloc[0]
    short_row = report[report["파일명"] == "short.docx"].iloc[0]
    good_row = report[report["파일명"] == "good.pdf"].iloc[0]

    assert dup_row["품질상태"] == "duplicate_skip"
    assert "duplicate_skip" in dup_row["품질플래그"]
    assert "empty_text" not in dup_row["품질플래그"]

    assert short_row["품질상태"] == "review_required"
    assert "short_text" in short_row["품질플래그"]
    assert "format_variant_docx" in short_row["품질플래그"]

    assert good_row["품질상태"] == "ok"
    assert good_row["품질플래그"] == ""
