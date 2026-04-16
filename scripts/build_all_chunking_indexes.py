# scripts/build_all_chunking_indexes.py

import subprocess
import sys

PROVIDER = "configs/providers/openai_gpt5mini.yaml"

CHUNKING_CONFIGS = [
    {
        "name": "chunking_500_100",
        "config": "configs/chunking/chunking_500_100.yaml",
        "chunks": "data/processed/chunking-500-100/chunks.parquet",
    },
    {
        "name": "chunking_1000_150",
        "config": "configs/chunking/chunking_1000_150.yaml",
        "chunks": "data/processed/chunks.parquet",  # 기존 베이스라인
    },
    {
        "name": "chunking_1500_200",
        "config": "configs/chunking/chunking_1500_200.yaml",
        "chunks": "data/processed/chunking-1500-200/chunks.parquet",
    },
    {
        "name": "chunking_semantic_percentile",
        "config": "configs/chunking/chunking_semantic_percentile.yaml",
        "chunks": "data/processed/semantic-percentile/chunks.parquet",
    },
    # {
    #     "name": "chunking_semantic_std",
    #     "config": "configs/chunking/chunking_semantic_std.yaml",
    #     "chunks": "data/processed/semantic-std/chunks.parquet",
    # },
    # {
    #     "name": "chunking_semantic_interquartile",
    #     "config": "configs/chunking/chunking_semantic_interquartile.yaml",
    #     "chunks": "data/processed/semantic-interquartile/chunks.parquet",
    # },
]

def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"❌ 실패: {' '.join(cmd)}")
        sys.exit(1)

def main() -> None:
    total = len(CHUNKING_CONFIGS)
    for i, cfg in enumerate(CHUNKING_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {cfg['name']}")
        print(f"{'='*60}")

        # 1. ingest (chunking_1000_150는 이미 있으므로 skip)
        if cfg["name"] != "chunking_1000_150":
            print(f"  → ingest 실행 중...")
            run([
                "python", "scripts/ingest_data.py",
                "--experiment-config", cfg["config"],
                "--parsed-path", "data/processed/parsed_documents.parquet",
            ])

        # 2. build_index
        print(f"  → build_index 실행 중...")
        run([
            "python", "scripts/build_index.py",
            "--provider-config", PROVIDER,
            "--experiment-config", cfg["config"],
            "--chunks-path", cfg["chunks"],
            
        ])

        print(f"  ✅ {cfg['name']} 완료!")

    print(f"\n{'='*60}")
    print("🎉 모든 청킹 벡터 DB 생성 완료!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()