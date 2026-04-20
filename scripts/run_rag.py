"""단일 질문 RAG CLI.

터미널에서 질문 하나를 던지고 답변을 받는 스크립트.
UI 없이 빠르게 리트리버 + LLM 응답을 확인할 때 사용한다.

사용 예시::

    uv run python scripts/run_rag.py \\
        --provider-config configs/providers/openai_gpt5mini.yaml \\
        --question "국민연금공단 이러닝시스템 요구사항 정리해줘"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def _load_history(*, history_json: str | None, history_file: str | None) -> list[dict] | None:
    """CLI 인자에서 대화 히스토리를 로드한다."""

    if history_file:
        payload = Path(history_file).read_text(encoding="utf-8-sig").lstrip("\ufeff")
    elif history_json:
        payload = history_json
    else:
        return None

    data = json.loads(payload)
    if not isinstance(data, list):
        raise SystemExit("chat history must be a JSON list.")
    return data


def main() -> None:
    """CLI 인자를 파싱하고 RAG 파이프라인으로 질문-답변을 실행"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Run a single RAG query.")
    parser.add_argument("--question", required=True)           # 질문 문자열
    parser.add_argument("--provider-config", required=True)    # LLM/임베딩 설정 YAML
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)   # 실험별 설정 (선택)
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--history-json", default=None)
    parser.add_argument("--history-file", default=None)
    args = parser.parse_args()

    # 설정 로딩 → RAG 파이프라인 구성 (리트리버 + LLM)
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
        retrieval_config_path=args.retrieval_config,
    )

    chat_history = _load_history(
        history_json=args.history_json,
        history_file=args.history_file,
    )

    # ExperimentConfig.retrieval_top_k 우선, 없으면 ProjectConfig 기본값
    top_k = (
        runtime.experiment.retrieval_top_k
        or runtime.project.default_retrieval_top_k
        or 5
    )

    # 질문을 파이프라인에 전달 → 벡터 검색 → LLM 답변 생성
    result = pipeline.answer(
        args.question,
        chat_history=chat_history,
        top_k=top_k,
        question_id=f"q-{uuid4().hex[:8]}",        # 랜덤 질문 ID
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=f"cli-{uuid4().hex[:8]}",            # 랜덤 실행 ID
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
    )

    debug = getattr(result, "debug", {}) or {}
    if debug:
        print(f"원본 질문: {debug.get('original_query', args.question)}")
        print(f"재작성 질문: {debug.get('rewritten_query', args.question)}")
        if debug.get("memory_summary"):
            print(f"메모리 요약: {debug['memory_summary']}")
        if debug.get("memory_slots"):
            print(f"메모리 슬롯: {debug['memory_slots']}")
        print(f"재작성 비용(USD): {float(debug.get('rewrite_cost_usd', 0.0) or 0.0):.6f}")
        print(f"생성 비용(USD): {float(debug.get('generation_cost_usd', result.cost_usd) or 0.0):.6f}")
        print(f"총 비용(USD): {float(debug.get('total_cost_usd', result.cost_usd) or 0.0):.6f}")
        print()

    # 최종 답변 출력
    print(result.answer)


if __name__ == "__main__":
    main()
