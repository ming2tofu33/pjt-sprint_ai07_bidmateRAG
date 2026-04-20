"""문서 인제스트 CLI.

RFP 원본 문서(HWP/PDF)를 파싱 → 정제 → 청킹하여
parquet 파일로 저장하는 스크립트.

사용 예시::

    uv run python scripts/ingest_data.py
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from bidmate_rag.pipelines.ingest import run_ingest_pipeline


def main() -> None:
    """CLI 인자를 파싱하고 인제스트 파이프라인을 실행"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Parse, clean, and chunk RFP documents.")
    parser.add_argument("--metadata-path", default="data/raw/metadata/data_list.csv")  # 문서 메타 CSV
    parser.add_argument("--raw-dir", default="data/raw/rfp")           # 원본 RFP 문서 폴더
    parser.add_argument("--output-dir", default="data/processed")      # 결과 저장 경로
    parser.add_argument("--chunk-size", type=int, default=1000)        # 청크 크기 (글자 수)
    parser.add_argument("--chunk-overlap", type=int, default=150)      # 청크 간 겹침 (글자 수)
    parser.add_argument("--experiment-config", default=None,
                        help="실험 config (YAML). chunk_size/overlap 설정을 오버라이드")
    parser.add_argument("--parsed-path", default=None,
                    help="기존 파싱 결과 parquet 경로. 지정하면 파싱 단계를 건너뜀")
    args = parser.parse_args()

    # 기본 청킹 설정
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    # 실험 config가 있으면 청킹 설정을 덮어쓰고, 출력 경로도 실험별로 분리
    if args.experiment_config:
        from pathlib import Path

        import yaml
        exp_cfg = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
        chunk_size = exp_cfg.get("chunk_size", chunk_size)
        chunk_overlap = exp_cfg.get("chunk_overlap", chunk_overlap)
        exp_name = exp_cfg.get("name", "default")
        output_dir = f"data/processed/{exp_name}"
        print(f"실험 config: {args.experiment_config}")
    else:
        output_dir = args.output_dir

    print(f"청킹 설정: size={chunk_size}, overlap={chunk_overlap}")
    print(f"출력 경로: {output_dir}")

    # 인제스트 파이프라인 실행: 파싱 → 정제 → 청킹 → parquet 저장
    outputs = run_ingest_pipeline(
        metadata_path=args.metadata_path,
        raw_dir=args.raw_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parsed_path=args.parsed_path,
    )

    # 산출물 경로 출력 (parsed_documents.parquet, chunks.parquet 등)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
