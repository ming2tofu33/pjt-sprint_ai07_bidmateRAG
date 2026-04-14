"""Build a document quality checklist CSV from cleaned parquet output."""

from __future__ import annotations

import argparse
from pathlib import Path

from bidmate_rag.storage.document_quality import (
    build_document_quality_report_from_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a document quality checklist from cleaned_documents.parquet."
    )
    parser.add_argument(
        "--metadata-path",
        default="data/raw/metadata/data_list.csv",
        help="입력 metadata CSV 경로",
    )
    parser.add_argument(
        "--cleaned-path",
        default="data/processed/cleaned_documents.parquet",
        help="입력 cleaned_documents.parquet 경로",
    )
    parser.add_argument(
        "--duplicates-map-path",
        default="configs/data_quality/duplicates_map.csv",
        help="duplicates_map.csv 경로",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/document_quality_checklist.csv",
        help="출력 CSV 경로",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=500,
        help="정제 본문 길이가 이 값보다 짧으면 short_text로 표시",
    )
    args = parser.parse_args()

    cleaned_path = Path(args.cleaned_path)
    metadata_path = Path(args.metadata_path)
    duplicates_map_path = Path(args.duplicates_map_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_document_quality_report_from_sources(
        metadata_path=metadata_path,
        cleaned_documents_path=cleaned_path,
        duplicates_map_path=duplicates_map_path,
        short_text_threshold=args.short_threshold,
    )
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"metadata: {metadata_path}")
    print(f"input: {cleaned_path}")
    print(f"rows: {len(report)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
