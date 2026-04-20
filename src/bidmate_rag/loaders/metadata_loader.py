"""Metadata loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_DUPLICATES_MAP = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_quality"
    / "duplicates_map.csv"
)

_REQUIRED_DUPLICATE_COLUMNS = {
    "duplicate_group_id",
    "source_file",
    "canonical_file",
    "is_duplicate",
    "resolved_agency",
}


def _stringify_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _coerce_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_duplicates_map(duplicates_map_path: str | Path | None) -> pd.DataFrame:
    path = Path(duplicates_map_path) if duplicates_map_path else _DEFAULT_DUPLICATES_MAP
    if not path.exists():
        return pd.DataFrame(columns=sorted(_REQUIRED_DUPLICATE_COLUMNS))

    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = _REQUIRED_DUPLICATE_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"duplicates_map.csv is missing required columns: {missing}")

    frame = frame.copy()
    frame["source_file"] = frame["source_file"].map(_stringify_text)
    frame["canonical_file"] = frame["canonical_file"].map(_stringify_text)
    frame["duplicate_group_id"] = frame["duplicate_group_id"].map(_stringify_text)
    frame["resolved_agency"] = frame["resolved_agency"].map(_stringify_text)
    frame["is_duplicate"] = frame["is_duplicate"].map(_coerce_bool)
    return frame


def load_metadata_frame(
    metadata_path: str | Path,
    duplicates_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load metadata CSV and enrich it with duplicate-resolution columns."""

    frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
    frame = frame.copy()

    if "공고 번호" in frame.columns:
        frame["공고 번호"] = frame["공고 번호"].map(_stringify_text)
    if "파일명" in frame.columns:
        frame["파일명"] = frame["파일명"].map(_stringify_text)
    if "발주 기관" in frame.columns:
        frame["발주 기관"] = frame["발주 기관"].map(_stringify_text)

    file_names = frame["파일명"] if "파일명" in frame.columns else pd.Series("", index=frame.index)
    agencies = frame["발주 기관"] if "발주 기관" in frame.columns else pd.Series("", index=frame.index)

    frame["duplicate_group_id"] = pd.NA
    frame["canonical_file"] = file_names
    frame["is_duplicate"] = False
    frame["ingest_enabled"] = True
    frame["ingest_file"] = file_names
    frame["resolved_agency"] = agencies
    frame["original_agency"] = agencies

    duplicates_df = _load_duplicates_map(duplicates_map_path)
    if duplicates_df.empty:
        if "발주 기관" in frame.columns:
            frame["발주 기관"] = frame["resolved_agency"]
        return frame

    duplicate_lookup = duplicates_df.set_index("source_file").to_dict("index")
    present_files = set(file_names)

    for idx, row in frame.iterrows():
        source_file = row["파일명"]
        duplicate_info = duplicate_lookup.get(source_file)
        if duplicate_info is None:
            continue

        canonical_file = duplicate_info["canonical_file"] or source_file
        is_duplicate = bool(duplicate_info["is_duplicate"])
        resolved_agency = duplicate_info["resolved_agency"] or row["발주 기관"]

        frame.at[idx, "duplicate_group_id"] = duplicate_info["duplicate_group_id"] or pd.NA
        frame.at[idx, "canonical_file"] = canonical_file
        frame.at[idx, "is_duplicate"] = is_duplicate
        frame.at[idx, "ingest_file"] = canonical_file
        frame.at[idx, "resolved_agency"] = resolved_agency

        if is_duplicate:
            frame.at[idx, "ingest_enabled"] = canonical_file not in present_files

    if "발주 기관" in frame.columns:
        frame["발주 기관"] = frame["resolved_agency"]
    return frame
