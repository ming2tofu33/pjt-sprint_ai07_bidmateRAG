"""RAG 평가 벤치마크 CLI shim.

실제 구현은 bidmate_rag.cli.eval에 있으며, 이 파일은
기존 호출(uv run python scripts/run_eval.py ...)이 계속 동작하도록 유지한다.

사용 예시::

    uv run python scripts/run_eval.py \\
        --evaluation-path data/eval/eval_v1/eval_batch_01.csv \\
        --provider-config configs/providers/openai_gpt5mini.yaml
"""

from __future__ import annotations

# 실제 CLI 로직은 bidmate_rag.cli.eval.main에 위임
from bidmate_rag.cli.eval import main

if __name__ == "__main__":
    main()
