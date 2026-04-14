"""Document quality report helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.loaders.metadata_loader import load_metadata_frame


def _to_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _build_quality_flags(row: pd.Series, short_text_threshold: int) -> list[str]:
    flags: list[str] = []
    cleaned_chars = int(row.get("정제_글자수", 0) or 0)
    source_file = str(row.get("파일명", "") or "")
    ingest_file = str(row.get("ingest_file", source_file) or source_file)
    file_type = str(row.get("파일형식", "") or "").lower()
    is_duplicate = _to_bool(row.get("is_duplicate", False))
    ingest_enabled = _to_bool(row.get("ingest_enabled", True))

    if is_duplicate and not ingest_enabled:
        flags.append("duplicate_skip")
    if ingest_enabled and source_file and ingest_file and source_file != ingest_file:
        flags.append("canonical_redirect")
    if ingest_enabled:
        if cleaned_chars == 0:
            flags.append("empty_text")
        elif cleaned_chars < short_text_threshold:
            flags.append("short_text")
    if file_type == "docx":
        flags.append("format_variant_docx")
    return flags


def _derive_quality_status(flags: list[str]) -> str:
    if "empty_text" in flags or "short_text" in flags:
        return "review_required"
    if "duplicate_skip" in flags:
        return "duplicate_skip"
    if "canonical_redirect" in flags:
        return "canonical_redirect"
    return "ok"


def _describe_flags(flags: list[str]) -> str:
    descriptions = {
        "duplicate_skip": "정본 행이 따로 있어 이 행은 ingest에서 제외됨",
        "canonical_redirect": "현재 메타데이터 행은 유지하되 실제 파싱은 정본 파일로 수행됨",
        "empty_text": "정제 본문 길이가 0자라 재확인 필요",
        "short_text": "정제 본문 길이가 짧아 파싱 품질 점검 필요",
        "format_variant_docx": "동일 문서의 포맷 변형본일 수 있어 정본 여부 점검 권장",
    }
    return " | ".join(descriptions[flag] for flag in flags)


def build_document_quality_report(
    frame: pd.DataFrame,
    short_text_threshold: int = 500,
) -> pd.DataFrame:
    """Build a human-reviewable quality checklist from cleaned documents."""

    working = frame.copy().fillna("")
    flags = [
        _build_quality_flags(row, short_text_threshold)
        for _, row in working.iterrows()
    ]
    statuses = [_derive_quality_status(items) for items in flags]
    reasons = [_describe_flags(items) for items in flags]

    report = pd.DataFrame(
        {
            "파일명": working.get("파일명", ""),
            "ingest_file": working.get("ingest_file", working.get("파일명", "")),
            "canonical_file": working.get("canonical_file", working.get("파일명", "")),
            "duplicate_group_id": working.get("duplicate_group_id", ""),
            "is_duplicate": working.get("is_duplicate", False).map(_to_bool),
            "ingest_enabled": working.get("ingest_enabled", True).map(_to_bool),
            "original_agency": working.get("original_agency", working.get("발주 기관", "")),
            "resolved_agency": working.get("resolved_agency", working.get("발주 기관", "")),
            "파일형식": working.get("파일형식", ""),
            "공개연도": working.get("공개연도", ""),
            "기관유형": working.get("기관유형", ""),
            "사업도메인": working.get("사업도메인", ""),
            "본문_글자수": working.get("본문_글자수", 0),
            "정제_글자수": working.get("정제_글자수", 0),
            "품질상태": statuses,
            "품질플래그": [", ".join(items) for items in flags],
            "점검사유": reasons,
            "검토상태": "todo",
            "검토메모": "",
        }
    )

    return report.sort_values(
        by=["품질상태", "정제_글자수", "파일명"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_document_quality_report_from_parquet(
    cleaned_documents_path: str | Path,
    short_text_threshold: int = 500,
) -> pd.DataFrame:
    """Load cleaned parquet and return a document quality checklist."""

    frame = pd.read_parquet(cleaned_documents_path)
    return build_document_quality_report(frame, short_text_threshold=short_text_threshold)


def build_document_quality_report_from_sources(
    metadata_path: str | Path,
    cleaned_documents_path: str | Path,
    duplicates_map_path: str | Path | None = None,
    short_text_threshold: int = 500,
) -> pd.DataFrame:
    """Build a quality report from the full metadata table plus cleaned parquet."""

    metadata_df = load_metadata_frame(
        metadata_path,
        duplicates_map_path=duplicates_map_path,
    )
    cleaned_df = pd.read_parquet(cleaned_documents_path)

    merge_columns = [
        column
        for column in [
            "파일명",
            "본문_글자수",
            "정제_글자수",
            "기관유형",
            "사업도메인",
            "기술스택",
            "공개연도",
        ]
        if column in cleaned_df.columns
    ]
    merged = metadata_df.merge(
        cleaned_df[merge_columns],
        on="파일명",
        how="left",
    )
    return build_document_quality_report(merged, short_text_threshold=short_text_threshold)
