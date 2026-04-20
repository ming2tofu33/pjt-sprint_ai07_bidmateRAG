"""PEFT 어댑터 평가 CLI.

학습된 PEFT 어댑터의 아티팩트 이름을 출력하고,
어댑터를 적용한 벤치마크 실행 안내를 제공한다.

사용 예시::

    uv run python scripts/eval_peft.py --base-model gpt-5-mini
"""

from __future__ import annotations

import argparse

from bidmate_rag.training.peft import build_training_artifact_name


def main() -> None:
    """PEFT 아티팩트 이름을 생성하고 다음 단계를 안내"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(
        description="Describe the PEFT artifact that should be evaluated."
    )
    parser.add_argument("--base-model", required=True)   # 기반 모델명 (예: gpt-5-mini)
    parser.add_argument("--method", default="lora")      # PEFT 방식 (기본: lora)
    args = parser.parse_args()

    # 아티팩트 이름 생성 후 다음 단계 안내
    print(build_training_artifact_name(args.base_model, args.method))
    print("Run the benchmark pipeline with the adapter-aware provider config after training.")


if __name__ == "__main__":
    main()
