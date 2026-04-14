from pathlib import Path

import pandas as pd

from bidmate_rag.pipelines.ingest import run_ingest_pipeline


def _write_duplicates_map(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_ingest_pipeline_creates_parsed_cleaned_and_chunk_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    sample_file = raw_dir / "sample.hwp"
    sample_file.write_text("dummy", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "공고 번호": "20240001",
                "공고 차수": 0,
                "사업명": "교육 플랫폼 고도화",
                "사업 금액": 300000000,
                "발주 기관": "국민연금공단",
                "공개 일자": "2024-10-01",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "이러닝 시스템 고도화",
                "파일형식": "hwp",
                "파일명": "sample.hwp",
            }
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    def fake_parse(file_path: Path) -> dict:
        return {
            "파일명": file_path.name,
            "텍스트": (
                "Warning: test\n# 개요\n교육 플랫폼 구축<br>세부 내용\n"
                "# 예산\n| 항목 | 값 |\n| --- | --- |\n| 인건비 | 3억 |"
            ),
            "글자수": 120,
            "성공": True,
            "파서": "fake",
            "에러": None,
        }

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
    )

    assert outputs["parsed"].exists()
    assert outputs["cleaned"].exists()
    assert outputs["chunks"].exists()

    cleaned_df = pd.read_parquet(outputs["cleaned"])
    chunk_df = pd.read_parquet(outputs["chunks"])

    assert cleaned_df.loc[0, "본문_정제"].startswith("# 개요")
    assert "사업도메인" in chunk_df.columns
    assert chunk_df.loc[0, "발주 기관"] == "국민연금공단"


def test_run_ingest_pipeline_uses_canonical_file_for_duplicate_only_metadata_row(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    canonical_file = raw_dir / "KHIDI_doc.hwp"
    canonical_file.write_text("dummy", encoding="utf-8")

    pd.DataFrame(
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
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    duplicates_map_path = _write_duplicates_map(
        metadata_dir / "duplicates_map.csv",
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

    parsed_paths: list[str] = []

    def fake_parse(file_path: Path) -> dict:
        parsed_paths.append(file_path.name)
        return {
            "파일명": file_path.name,
            "텍스트": "# 개요\n기능개선",
            "글자수": 10,
            "성공": True,
            "파서": "fake",
            "에러": None,
        }

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
        duplicates_map_path=duplicates_map_path,
    )

    parsed_df = pd.read_parquet(outputs["parsed"])

    assert parsed_paths == ["KHIDI_doc.hwp"]
    assert parsed_df.loc[0, "파일명"] == "BioIN_doc.hwp"
    assert parsed_df.loc[0, "ingest_file"] == "KHIDI_doc.hwp"
    assert parsed_df.loc[0, "발주 기관"] == "한국보건산업진흥원"


def test_run_ingest_pipeline_skips_duplicate_row_when_canonical_row_exists(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    canonical_file = raw_dir / "KIOM_doc.hwp"
    canonical_file.write_text("dummy", encoding="utf-8")

    pd.DataFrame(
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
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    duplicates_map_path = _write_duplicates_map(
        metadata_dir / "duplicates_map.csv",
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

    parsed_paths: list[str] = []

    def fake_parse(file_path: Path) -> dict:
        parsed_paths.append(file_path.name)
        return {
            "파일명": file_path.name,
            "텍스트": "# 개요\n고도화",
            "글자수": 10,
            "성공": True,
            "파서": "fake",
            "에러": None,
        }

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
        duplicates_map_path=duplicates_map_path,
    )

    parsed_df = pd.read_parquet(outputs["parsed"])

    assert parsed_paths == ["KIOM_doc.hwp"]
    assert parsed_df["파일명"].to_list() == ["KIOM_doc.hwp"]


def test_run_ingest_pipeline_reuses_cached_parsed_output_with_dedup_targets(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    pd.DataFrame(
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
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    duplicates_map_path = _write_duplicates_map(
        metadata_dir / "duplicates_map.csv",
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

    cached_parsed_path = tmp_path / "cached.parquet"
    pd.DataFrame(
        [
            {
                "파일명": "BioIN_doc.hwp",
                "본문_마크다운": "# 개요\n캐시된 파싱 결과",
                "본문_글자수": 12,
            }
        ]
    ).to_parquet(cached_parsed_path, index=False)

    parse_called = False

    def fake_parse(_: Path) -> dict:
        nonlocal parse_called
        parse_called = True
        return {}

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
        duplicates_map_path=duplicates_map_path,
        parsed_path=cached_parsed_path,
    )

    parsed_df = pd.read_parquet(outputs["parsed"])

    assert parse_called is False
    assert parsed_df.loc[0, "파일명"] == "BioIN_doc.hwp"
    assert parsed_df.loc[0, "ingest_file"] == "KHIDI_doc.hwp"
    assert parsed_df.loc[0, "본문_마크다운"] == "# 개요\n캐시된 파싱 결과"


def test_run_ingest_pipeline_prefers_pdf_when_docx_is_marked_as_duplicate(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    canonical_file = raw_dir / "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
    canonical_file.write_text("dummy", encoding="utf-8")

    pd.DataFrame(
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
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    duplicates_map_path = _write_duplicates_map(
        metadata_dir / "duplicates_map.csv",
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

    parsed_paths: list[str] = []

    def fake_parse(file_path: Path) -> dict:
        parsed_paths.append(file_path.name)
        return {
            "파일명": file_path.name,
            "텍스트": "# 개요\n포털 구축",
            "글자수": 10,
            "성공": True,
            "파서": "fake",
            "에러": None,
        }

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
        duplicates_map_path=duplicates_map_path,
    )

    parsed_df = pd.read_parquet(outputs["parsed"])

    assert parsed_paths == ["고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"]
    assert parsed_df["파일명"].to_list() == ["고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"]
