from pathlib import Path

import pandas as pd

from bidmate_rag.loaders.metadata_loader import load_metadata_frame


def _write_metadata_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_duplicates_map(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_metadata_frame_without_duplicates_map_keeps_default_ingest_behavior(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata_csv(
        tmp_path / "data_list.csv",
        [
            {
                "공고 번호": "20240001",
                "공고 차수": 0,
                "사업명": "테스트 사업",
                "사업 금액": 100000000,
                "발주 기관": "테스트기관",
                "공개 일자": "2024-10-01",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "요약",
                "파일형식": "hwp",
                "파일명": "sample.hwp",
            }
        ],
    )

    frame = load_metadata_frame(metadata_path)

    assert frame.loc[0, "canonical_file"] == "sample.hwp"
    assert frame.loc[0, "ingest_file"] == "sample.hwp"
    assert bool(frame.loc[0, "ingest_enabled"]) is True
    assert bool(frame.loc[0, "is_duplicate"]) is False
    assert frame.loc[0, "resolved_agency"] == "테스트기관"
    assert frame.loc[0, "original_agency"] == "테스트기관"


def test_load_metadata_frame_promotes_duplicate_row_when_only_duplicate_exists_in_csv(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata_csv(
        tmp_path / "data_list.csv",
        [
            {
                "공고 번호": "20240002",
                "공고 차수": 0,
                "사업명": "의료기기산업 종합정보시스템 기능개선",
                "사업 금액": 50000000,
                "발주 기관": "BioIN",
                "공개 일자": "2024-09-05",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "요약",
                "파일형식": "hwp",
                "파일명": "BioIN_doc.hwp",
            }
        ],
    )
    duplicates_map_path = _write_duplicates_map(
        tmp_path / "duplicates_map.csv",
        [
            {
                "duplicate_group_id": "DUP-001",
                "source_file": "BioIN_doc.hwp",
                "canonical_file": "KHIDI_doc.hwp",
                "is_duplicate": True,
                "resolved_agency": "한국보건산업진흥원",
                "status": "confirmed",
                "reason": "same document",
                "metadata_merge_note": "alias only",
            }
        ],
    )

    frame = load_metadata_frame(metadata_path, duplicates_map_path=duplicates_map_path)

    assert frame.loc[0, "duplicate_group_id"] == "DUP-001"
    assert frame.loc[0, "canonical_file"] == "KHIDI_doc.hwp"
    assert frame.loc[0, "ingest_file"] == "KHIDI_doc.hwp"
    assert bool(frame.loc[0, "is_duplicate"]) is True
    assert bool(frame.loc[0, "ingest_enabled"]) is True
    assert frame.loc[0, "resolved_agency"] == "한국보건산업진흥원"
    assert frame.loc[0, "original_agency"] == "BioIN"


def test_load_metadata_frame_keeps_only_canonical_row_enabled_when_both_rows_exist(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata_csv(
        tmp_path / "data_list.csv",
        [
            {
                "공고 번호": "",
                "공고 차수": "",
                "사업명": "통합정보시스템 고도화 용역",
                "사업 금액": 140000000,
                "발주 기관": "국가과학기술지식정보서비스",
                "공개 일자": "2024-05-30 00:00:00",
                "입찰 참여 시작일": "2024-05-30 00:00:00",
                "입찰 참여 마감일": "2024-06-11 00:00:00",
                "사업 요약": "요약",
                "파일형식": "hwp",
                "파일명": "NTIS_doc.hwp",
            },
            {
                "공고 번호": "20240535775",
                "공고 차수": 0,
                "사업명": "통합정보시스템 고도화 용역",
                "사업 금액": 140000000,
                "발주 기관": "한국한의학연구원",
                "공개 일자": "2024-05-30 09:04:12",
                "입찰 참여 시작일": "2024-05-30 10:00:00",
                "입찰 참여 마감일": "2024-06-11 11:00:00",
                "사업 요약": "요약",
                "파일형식": "hwp",
                "파일명": "KIOM_doc.hwp",
            },
        ],
    )
    duplicates_map_path = _write_duplicates_map(
        tmp_path / "duplicates_map.csv",
        [
            {
                "duplicate_group_id": "DUP-002",
                "source_file": "NTIS_doc.hwp",
                "canonical_file": "KIOM_doc.hwp",
                "is_duplicate": True,
                "resolved_agency": "한국한의학연구원",
                "status": "confirmed",
                "reason": "same document",
                "metadata_merge_note": "metadata source only",
            },
            {
                "duplicate_group_id": "DUP-002",
                "source_file": "KIOM_doc.hwp",
                "canonical_file": "KIOM_doc.hwp",
                "is_duplicate": False,
                "resolved_agency": "한국한의학연구원",
                "status": "confirmed",
                "reason": "canonical",
                "metadata_merge_note": "canonical row",
            },
        ],
    )

    frame = load_metadata_frame(metadata_path, duplicates_map_path=duplicates_map_path)
    ntis_row = frame[frame["파일명"] == "NTIS_doc.hwp"].iloc[0]
    kiom_row = frame[frame["파일명"] == "KIOM_doc.hwp"].iloc[0]

    assert bool(ntis_row["is_duplicate"]) is True
    assert bool(ntis_row["ingest_enabled"]) is False
    assert ntis_row["ingest_file"] == "KIOM_doc.hwp"
    assert ntis_row["resolved_agency"] == "한국한의학연구원"
    assert ntis_row["original_agency"] == "국가과학기술지식정보서비스"

    assert bool(kiom_row["is_duplicate"]) is False
    assert bool(kiom_row["ingest_enabled"]) is True
    assert kiom_row["ingest_file"] == "KIOM_doc.hwp"
    assert kiom_row["resolved_agency"] == "한국한의학연구원"
    assert kiom_row["original_agency"] == "한국한의학연구원"


def test_load_metadata_frame_disables_docx_when_pdf_is_canonical_for_same_project(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata_csv(
        tmp_path / "data_list.csv",
        [
            {
                "공고 번호": "20240003",
                "공고 차수": 0,
                "사업명": "차세대 포털·학사 정보시스템 구축사업",
                "사업 금액": 11270000000,
                "발주 기관": "고려대학교",
                "공개 일자": "2024-04-01",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "요약",
                "파일형식": "docx",
                "파일명": "고려대학교_차세대 포털·학사 정보시스템 구축사업.docx",
            },
            {
                "공고 번호": "20240003",
                "공고 차수": 0,
                "사업명": "차세대 포털·학사 정보시스템 구축사업",
                "사업 금액": 11270000000,
                "발주 기관": "고려대학교",
                "공개 일자": "2024-04-01",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "요약",
                "파일형식": "pdf",
                "파일명": "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
            },
        ],
    )
    duplicates_map_path = _write_duplicates_map(
        tmp_path / "duplicates_map.csv",
        [
            {
                "duplicate_group_id": "DUP-003",
                "source_file": "고려대학교_차세대 포털·학사 정보시스템 구축사업.docx",
                "canonical_file": "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
                "is_duplicate": True,
                "resolved_agency": "고려대학교",
                "status": "confirmed",
                "reason": "same project different format",
                "metadata_merge_note": "docx duplicate",
            },
            {
                "duplicate_group_id": "DUP-003",
                "source_file": "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
                "canonical_file": "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf",
                "is_duplicate": False,
                "resolved_agency": "고려대학교",
                "status": "confirmed",
                "reason": "canonical pdf",
                "metadata_merge_note": "pdf canonical",
            },
        ],
    )

    frame = load_metadata_frame(metadata_path, duplicates_map_path=duplicates_map_path)
    docx_row = frame[frame["파일명"] == "고려대학교_차세대 포털·학사 정보시스템 구축사업.docx"].iloc[0]
    pdf_row = frame[frame["파일명"] == "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"].iloc[0]

    assert bool(docx_row["is_duplicate"]) is True
    assert bool(docx_row["ingest_enabled"]) is False
    assert docx_row["ingest_file"] == "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
    assert docx_row["resolved_agency"] == "고려대학교"

    assert bool(pdf_row["is_duplicate"]) is False
    assert bool(pdf_row["ingest_enabled"]) is True
    assert pdf_row["ingest_file"] == "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
    assert pdf_row["resolved_agency"] == "고려대학교"
