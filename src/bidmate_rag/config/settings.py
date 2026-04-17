"""Runtime settings helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """프로젝트 기본 설정."""

    project_name: str = "bidmate-rag"
    default_retrieval_top_k: int = 5
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 150


class BoostConfig(BaseModel):
    """섹션/테이블 부스팅 가중치 설정."""

    section: float = 0.12    # 섹션 힌트 일치 시 가산
    table: float = 0.08      # 표(table) 청크 가산
    max_total: float = 0.15  # 부스팅 합산 상한


class HybridConfig(BaseModel):
    """하이브리드 검색 (Dense + Sparse RRF) 설정."""

    enabled: bool = False              # 하이브리드 검색 활성화 여부
    dense_pool_multiplier: int = 3     # Dense 후보 풀 배수
    sparse_pool_multiplier: int = 3    # Sparse 후보 풀 배수
    rrf_k: int = 60                    # RRF 순위 융합 상수


class RewriteConfig(BaseModel):
    """멀티턴 쿼리 재작성 설정."""

    mode: str = "llm_with_rule_fallback"
    max_completion_tokens: int = Field(16000, gt=0)
    timeout_seconds: int = Field(30, gt=0)


class SummaryBufferConfig(BaseModel):
    """오래된 대화 요약 버퍼 설정."""

    max_recent_turns: int = 4
    max_summary_chars: int = 400


class SlotMemoryConfig(BaseModel):
    """구조화 슬롯 메모리 설정."""

    enabled: bool = True


class MemoryConfig(BaseModel):
    """멀티턴 메모리 설정."""

    enabled: bool = True
    summary_buffer: SummaryBufferConfig = Field(default_factory=SummaryBufferConfig)
    slot_memory: SlotMemoryConfig = Field(default_factory=SlotMemoryConfig)


class DebugTraceConfig(BaseModel):
    """디버그 추적 설정."""

    enabled: bool = True


class RetrievalConfig(BaseModel):
    """검색 전략 설정 (리랭커, 멀티턴, 부스팅, 하이브리드)."""

    reranker_model: str | None = None  # Cross-Encoder 모델명 (null이면 비활성화)
    enable_multiturn: bool = True      # 멀티턴 검색 보강 사용 여부
    boost: BoostConfig = Field(default_factory=BoostConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    debug_trace: DebugTraceConfig = Field(default_factory=DebugTraceConfig)


class ProviderConfig(BaseModel):
    """LLM/임베딩 프로바이더 설정."""

    provider: str
    model: str
    api_base: str | None = None
    scenario: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """실험 구성 설정."""

    name: str
    mode: str = "full_rag"
    notes_path: str | None = None
    retrieval_top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    provider_configs: list[str] = Field(default_factory=list)
    # Grid search 매트릭스 — 키는 ExperimentConfig 필드명(chunk_size,
    # retrieval_top_k 등), 값은 값 리스트. 예:
    #   matrix:
    #     chunk_size: [500, 1000]
    #     retrieval_top_k: [3, 5, 8]
    # → 6개 sub-experiment로 자동 expand (run_experiment.py에서)
    matrix: dict[str, list[Any]] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """프로젝트 + 프로바이더 + 실험 + 검색 설정을 통합한 런타임 설정."""

    project: ProjectConfig
    provider: ProviderConfig
    experiment: ExperimentConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    """YAML 파일을 읽어 dict로 반환한다.

    Args:
        path: YAML 파일 경로. None이면 빈 dict 반환.

    Returns:
        파싱된 dict. 파일이 비어있으면 빈 dict.
    """
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


def load_runtime_config(
    base_config_path: str | Path,
    provider_config_path: str | Path,
    experiment_config_path: str | Path | None = None,
    retrieval_config_path: str | Path | None = None,
) -> RuntimeConfig:
    """여러 YAML 설정 파일을 로딩·병합하여 RuntimeConfig를 생성한다.

    Args:
        base_config_path: 기본 프로젝트 설정 YAML 경로.
        provider_config_path: 프로바이더 설정 YAML 경로.
        experiment_config_path: 실험 설정 YAML 경로 (선택).
        retrieval_config_path: 검색 전략 YAML 경로 (선택).

    Returns:
        통합된 RuntimeConfig 객체.
    """
    base_data = _load_yaml(base_config_path)
    provider_data = _load_yaml(provider_config_path)
    experiment_data = _load_yaml(experiment_config_path)
    retrieval_data = _load_yaml(retrieval_config_path)

    project = ProjectConfig.model_validate(base_data)
    retrieval = RetrievalConfig.model_validate(retrieval_data or {})
    experiment = ExperimentConfig.model_validate(experiment_data or {"name": "ad-hoc"})
    if experiment.retrieval_top_k is None:
        experiment.retrieval_top_k = project.default_retrieval_top_k
    if experiment.chunk_size is None:
        experiment.chunk_size = project.default_chunk_size
    if experiment.chunk_overlap is None:
        experiment.chunk_overlap = project.default_chunk_overlap

    provider_known_keys = {
        "provider",
        "model",
        "api_base",
        "scenario",
        "embedding_model",
        "collection_name",
    }
    provider_extra = {
        key: value for key, value in provider_data.items() if key not in provider_known_keys
    }
    provider = ProviderConfig.model_validate({**provider_data, "extra": provider_extra})

    return RuntimeConfig(
        project=project, provider=provider, experiment=experiment, retrieval=retrieval
    )
