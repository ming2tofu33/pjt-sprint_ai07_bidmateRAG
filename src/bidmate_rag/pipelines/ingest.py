"""Document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.loaders.hwp_loader import parse_hwp
from bidmate_rag.loaders.metadata_loader import load_metadata_frame
from bidmate_rag.loaders.pdf_loader import parse_pdf
from bidmate_rag.preprocessing.chunker import (
    chunk_document,
    classify_agency,
    classify_domain,
    extract_tech_stack,
)
from bidmate_rag.preprocessing.cleaner import clean_text


def _default_parse(file_path: Path) -> dict:
    """Select the proper parser based on the file suffix."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    return parse_hwp(file_path)


def _public_year(value) -> int:
    """Extract a four-digit year from the published date field."""
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 0


def _build_parsed_frame_from_cache(
    ingest_df: pd.DataFrame,
    parsed_source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project cached parsed output onto the current ingest target rows.

    The cached parquet may come from an older run that still contains duplicate
    rows. We therefore rebuild the final parsed frame using the *current*
    ``ingest_df`` as the source of truth and only reuse the parsed text payload.
    """
    required_cols = {"파일명", "본문_마크다운", "본문_글자수"}
    missing_cols = required_cols.difference(parsed_source_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"parsed_path is missing required columns: {missing}")

    records = parsed_source_df.to_dict(orient="records")
    parsed_by_file = {str(row["파일명"]): row for row in records}
    parsed_by_ingest = {}
    if "ingest_file" in parsed_source_df.columns:
        for row in records:
            ingest_file = str(row.get("ingest_file") or row["파일명"])
            parsed_by_ingest[ingest_file] = row

    rebuilt_rows: list[dict] = []
    for _, row in ingest_df.iterrows():
        source_file = str(row["파일명"])
        ingest_file = str(row["ingest_file"])
        cached = (
            parsed_by_ingest.get(ingest_file)
            or parsed_by_file.get(ingest_file)
            or parsed_by_file.get(source_file)
        )
        if cached is None:
            raise KeyError(
                "Cached parsed output does not contain a matching row for "
                f"source_file={source_file!r}, ingest_file={ingest_file!r}."
            )

        rebuilt = row.to_dict()
        rebuilt["본문_마크다운"] = cached["본문_마크다운"]
        rebuilt["본문_글자수"] = cached["본문_글자수"]
        rebuilt_rows.append(rebuilt)

    return pd.DataFrame(rebuilt_rows)


def run_ingest_pipeline(
    metadata_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    parse_fn=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    duplicates_map_path: str | Path | None = None,
    parsed_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the full ingest pipeline.

    Steps:
    1. Load metadata CSV
    2. Parse raw documents or reuse cached parsed output
    3. Clean text and enrich metadata
    4. Chunk and persist parquet artifacts
    """
    parser = parse_fn or _default_parse
    raw_root = Path(raw_dir)
    processed_root = Path(output_dir)
    processed_root.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_frame(
        metadata_path,
        duplicates_map_path=duplicates_map_path,
    )
    ingest_df = metadata_df[metadata_df["ingest_enabled"]].copy()

    if parsed_path:
        cached_path = Path(parsed_path)
        print(f"파싱 결과 재사용: {cached_path}")
        parsed_source_df = pd.read_parquet(cached_path)
        parsed_df = _build_parsed_frame_from_cache(ingest_df, parsed_source_df)
    else:
        parsed_rows: list[dict] = []
        total = len(ingest_df)
        for i, (_, row) in enumerate(ingest_df.iterrows(), 1):
            source_file = row["파일명"]
            ingest_file = row["ingest_file"]
            file_path = raw_root / ingest_file
            display_name = (
                source_file
                if source_file == ingest_file
                else f"{source_file} -> {ingest_file}"
            )
            print(f"[{i}/{total}] 파싱 중: {display_name}", end=" ... ")
            try:
                parsed = parser(file_path)
            except Exception as exc:
                parsed = {
                    "파일명": source_file,
                    "파서": "error",
                    "텍스트": "",
                    "글자수": 0,
                    "성공": False,
                    "에러": str(exc),
                }
            else:
                parsed = parsed.copy()
                parsed["파일명"] = source_file

            parsed["ingest_file"] = ingest_file
            status = "OK" if parsed.get("성공") else f"실패({parsed.get('에러', '?')})"
            print(f"{status} ({parsed.get('글자수', 0):,}자)")
            parsed_rows.append(parsed)

        parsed_df = ingest_df.copy()
        parsed_map = {row["파일명"]: row for row in parsed_rows}
        parsed_df["본문_마크다운"] = parsed_df["파일명"].map(
            lambda name: parsed_map[name]["텍스트"]
        )
        parsed_df["본문_글자수"] = parsed_df["파일명"].map(
            lambda name: parsed_map[name]["글자수"]
        )

    parsed_output_path = processed_root / "parsed_documents.parquet"
    parsed_df.to_parquet(parsed_output_path, index=False)

    cleaned_df = parsed_df.copy()
    cleaned_df["본문_정제"] = cleaned_df["본문_마크다운"].fillna("").map(clean_text)
    cleaned_df["정제_글자수"] = cleaned_df["본문_정제"].str.len()
    cleaned_df["기관유형"] = cleaned_df["발주 기관"].map(classify_agency)
    cleaned_df["사업도메인"] = [
        classify_domain(name, text)
        for name, text in zip(cleaned_df["사업명"], cleaned_df["본문_정제"], strict=False)
    ]
    cleaned_df["기술스택"] = cleaned_df["본문_정제"].map(extract_tech_stack)
    cleaned_df["공개연도"] = cleaned_df["공개 일자"].map(_public_year)

    cleaned_path = processed_root / "cleaned_documents.parquet"
    cleaned_df.to_parquet(cleaned_path, index=False)

    skip_cols = {"본문_마크다운", "본문_정제", "본문_글자수", "정제_글자수"}
    all_chunks = []
    for _, row in cleaned_df.iterrows():
        metadata = {k: v for k, v in row.to_dict().items() if k not in skip_cols}
        metadata["doc_id"] = str(metadata.get("공고 번호") or metadata.get("파일명"))
        chunks = chunk_document(
            row["본문_정제"],
            metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunk.to_record() for chunk in chunks)

    chunk_df = pd.DataFrame(all_chunks)
    chunks_path = processed_root / "chunks.parquet"
    chunk_df.to_parquet(chunks_path, index=False)

    return {
        "parsed": parsed_output_path,
        "cleaned": cleaned_path,
        "chunks": chunks_path,
    }
