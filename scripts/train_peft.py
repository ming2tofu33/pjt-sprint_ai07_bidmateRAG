"""PEFT 파인튜닝 CLI.

사용 예시::

    # LoRA 학습
    uv run python scripts/train_peft.py \
        --train-jsonl data/train.jsonl \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --method lora

    # QLoRA 학습
    uv run python scripts/train_peft.py \
        --train-jsonl data/train.jsonl \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --method qlora

    # 데이터 변환만 (학습 없이)
    uv run python scripts/train_peft.py \
        --train-jsonl data/train.jsonl \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --prepare-only
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from bidmate_rag.training.peft import (
    build_sft_record,
    default_adapter_dir,
    run_training,
    LoRAConfig,
    TrainingConfig,
)


def prepare_sft_data(train_jsonl: str, output_dir: Path) -> Path:
    """원본 JSONL → SFT 포맷으로 변환하여 저장."""
    formatted_path = output_dir / "sft_train.jsonl"
    count = 0
    with (
        Path(train_jsonl).open("r", encoding="utf-8") as src,
        formatted_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            record = build_sft_record(json.loads(line))
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logger.info(f"SFT 포맷 변환 완료: {count}개 샘플 → {formatted_path}")
    return formatted_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA 파인튜닝 실행")

    # 필수 인자
    parser.add_argument("--train-jsonl", required=True,
                        help="학습 데이터 JSONL 경로 (instruction/output 키 필요)")
    parser.add_argument("--base-model", required=True,
                        help="베이스 모델명 (예: Qwen/Qwen2.5-0.5B-Instruct)")

    # 학습 방식
    parser.add_argument("--method", default="lora", choices=["lora", "qlora"],
                        help="PEFT 방식 (기본: lora)")
    parser.add_argument("--output-root", default="artifacts/training",
                        help="어댑터 저장 루트 경로")

    # 데이터 변환만
    parser.add_argument("--prepare-only", action="store_true",
                        help="SFT 포맷 변환만 수행 (학습 없이)")

    # LoRA 하이퍼파라미터
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (기본: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (기본: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (기본: 0.05)")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=3,
                        help="학습 에폭 수 (기본: 3)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="배치 크기 (기본: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="학습률 (기본: 2e-4)")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="최대 시퀀스 길이 (기본: 512)")

    args = parser.parse_args()

    # 어댑터 저장 경로 설정
    output_dir = default_adapter_dir(args.output_root, args.base_model, args.method)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SFT 포맷 변환
    formatted_path = prepare_sft_data(args.train_jsonl, output_dir)

    # prepare-only면 여기서 종료
    if args.prepare_only:
        print(f"✅ SFT 데이터 변환 완료: {formatted_path}")
        print("학습을 실행하려면 --prepare-only 옵션을 제거하세요.")
        return

    # LoRA 설정
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # 학습 설정
    training_config = TrainingConfig(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_qlora=(args.method == "qlora"),
    )

    # 학습 실행
    adapter_path = run_training(
        base_model=args.base_model,
        train_jsonl=formatted_path,
        output_dir=output_dir,
        method=args.method,
        lora_config=lora_config,
        training_config=training_config,
    )

    print(f"✅ 학습 완료!")
    print(f"어댑터 저장 경로: {adapter_path}")
    print(f"사용 방법: --adapter-path {adapter_path}")


if __name__ == "__main__":
    main()