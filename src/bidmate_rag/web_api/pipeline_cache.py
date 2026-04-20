"""Cached `build_runtime_pipeline` wrapper.

`build_runtime_pipeline`은 ChromaDB 클라이언트/임베더/LLM 프로바이더를 새로 생성하므로
호출 한 번에 2~3초가 걸릴 수 있다. 웹 API처럼 반복 요청이 들어오는 환경에서는
(provider_config, chunking_config) 조합별로 재사용해야 한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from bidmate_rag.pipelines.runtime import build_runtime_pipeline

_PROVIDER_DIR = Path("configs/providers")
_CHUNKING_DIR = Path("configs/chunking")
_BASE_CONFIG = Path("configs/base.yaml")


@lru_cache(maxsize=8)
def get_pipeline(provider_config: str, chunking_config: str | None = None):
    """캐시된 RAG 파이프라인을 반환한다.

    Args:
        provider_config: provider config 파일명 (확장자 제외). 예: "openai_gpt5mini".
        chunking_config: chunking config 파일명 (확장자 제외). 예: "chunking_1000_150".
            None이면 experiment config 없이 legacy/shared 컬렉션을 사용.

    Returns:
        (pipeline, runtime, embedder, llm) 튜플 — `build_runtime_pipeline`과 동일.
    """
    provider_path = _PROVIDER_DIR / f"{provider_config}.yaml"
    if not provider_path.exists():
        raise FileNotFoundError(f"provider config not found: {provider_path}")
    chunking_path: Path | None = None
    if chunking_config:
        chunking_path = _CHUNKING_DIR / f"{chunking_config}.yaml"
        if not chunking_path.exists():
            raise FileNotFoundError(f"chunking config not found: {chunking_path}")
    return build_runtime_pipeline(
        base_config_path=_BASE_CONFIG,
        provider_config_path=provider_path,
        experiment_config_path=chunking_path,
    )


def clear_cache() -> None:
    """테스트에서 사용."""
    get_pipeline.cache_clear()
