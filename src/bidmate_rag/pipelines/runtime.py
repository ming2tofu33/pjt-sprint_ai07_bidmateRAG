"""런타임 조립 헬퍼.

CLI 스크립트와 Streamlit UI가 공유하는 파이프라인 조립 로직.
설정 파일 -> 프로바이더/리트리버/LLM 생성 -> RAGChatPipeline 반환.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bidmate_rag.config.settings import RuntimeConfig, load_runtime_config
from bidmate_rag.generation.calculation_engine import CalculationEngine
from bidmate_rag.pipelines.chat import RAGChatPipeline
from bidmate_rag.providers.llm.registry import build_embedding_provider, build_llm_provider
from bidmate_rag.retrieval.memory import ConversationMemory
from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.retrieval.sparse_store import BM25SparseStore
from bidmate_rag.retrieval.vector_store import ChromaVectorStore
from bidmate_rag.storage.calculation_store import CalculationStore
from bidmate_rag.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


def _load_reranker(model_name: str | None):
    """선택적 실험용 Cross-Encoder 리랭킹 모델을 로드한다.

    Args:
        model_name: HuggingFace 모델명. None이면 기본 운영 경로를 사용한다.

    Returns:
        CrossEncoder 모델 인스턴스. model_name이 None이거나 로드 실패 시 None.
    """
    if not model_name:
        return None
    try:
        from sentence_transformers import CrossEncoder

        logger.info("실험용 Cross-Encoder 리랭킹 모델 로딩: %s", model_name)
        model = CrossEncoder(model_name)
        logger.info("실험용 Cross-Encoder 로딩 완료")
        return model
    except Exception as e:
        logger.warning("실험용 Cross-Encoder 로딩 실패 (기본 운영 경로 계속 사용): %s", e)
        return None


def collection_name_for_config(runtime: RuntimeConfig) -> str:
    """RuntimeConfig에서 ChromaDB 컬렉션 이름을 생성한다.

    격리 규칙:
      - ``experiment.mode == "full_rag"`` (default): chunking을 바꿔가며 실험할
        가능성이 있으므로 ``실험명-...`` prefix로 격리.
      - ``experiment.mode == "generation_only"``: 동일 인덱스에 다른 LLM만
        붙이는 실험이므로 collection을 공유한다.
      - 실험 config가 없는 경우 (``ad-hoc``): legacy 동작 보존.

    ``provider.collection_name``이 명시된 경우:
      - 격리가 필요 없는 모드(generation_only / ad-hoc)에서는 그것을 그대로 사용
      - 격리가 필요한 모드(full_rag)에서는 ``{실험명}-{명시이름}`` 형식으로 prefix
    """
    model = (
        (runtime.provider.embedding_model or runtime.provider.model)
        .replace("/", "-")
        .replace(" ", "-")
    )
    exp_name = (runtime.experiment.name or "ad-hoc").replace("/", "-").replace(" ", "-")
    mode = runtime.experiment.mode or "full_rag"
    is_shared = mode == "generation_only" or exp_name in ("ad-hoc", "default", "")

    explicit = runtime.provider.collection_name
    if explicit:
        if is_shared:
            return explicit
        return f"{exp_name}-{explicit}".lower()

    if is_shared:
        return f"bidmate-{runtime.provider.provider}-{model}".lower()
    return f"bidmate-{exp_name}-{runtime.provider.provider}-{model}".lower()


def _resolve_metadata_path(runtime: RuntimeConfig, explicit: str | Path | None) -> Path:
    """실험별 metadata parquet 경로를 자동 결정한다.

    우선순위:
      1. ``explicit``가 명시되어 있고 파일이 존재 -> 그것 사용
      2. 실험별 sub-dir ``data/processed/{exp_name}/cleaned_documents.parquet``
      3. 공용 ``data/processed/cleaned_documents.parquet`` (legacy fallback)
    """
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path

    exp_name = runtime.experiment.name or "ad-hoc"
    if exp_name not in ("ad-hoc", "default", ""):
        sub = Path(f"data/processed/{exp_name}/cleaned_documents.parquet")
        if sub.exists():
            return sub

    return Path("data/processed/cleaned_documents.parquet")


def _resolve_chunks_path(runtime: RuntimeConfig, explicit: str | Path | None = None) -> Path:
    """실험별 chunks parquet 경로를 자동 결정한다.

    Args:
        runtime: 런타임 설정.
        explicit: 명시 경로. None이면 자동 탐지.

    Returns:
        chunks.parquet 경로.
    """
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path

    exp_name = runtime.experiment.name or "ad-hoc"
    if exp_name not in ("ad-hoc", "default", ""):
        sub = Path(f"data/processed/{exp_name}/chunks.parquet")
        if sub.exists():
            return sub

    return Path("data/processed/chunks.parquet")


def build_runtime_pipeline(
    base_config_path: str | Path,
    provider_config_path: str | Path,
    experiment_config_path: str | Path | None = None,
    retrieval_config_path: str | Path | None = "configs/retrieval.yaml",
    persist_dir: str | Path = "artifacts/chroma_db",
    metadata_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
):
    """설정 파일들로부터 RAGChatPipeline을 조립한다.

    Args:
        base_config_path: 기본 설정 YAML 경로.
        provider_config_path: 프로바이더 설정 YAML 경로.
        experiment_config_path: 실험 설정 YAML 경로 (선택).
        retrieval_config_path: 리트리벌 전략 YAML 경로 (기본: configs/retrieval.yaml).
        persist_dir: ChromaDB 저장 디렉터리.
        metadata_path: 정제된 문서 메타데이터 parquet 경로. None이면
            실험별 sub-dir -> 공용 순서로 자동 탐지.
        chunks_path: chunks parquet 경로. None이면 실험 설정 기준으로 자동 탐지.
        adapter_path: 로컬 LLM 어댑터 경로. 지정 시 LLM 생성에 함께 전달.

    Returns:
        (pipeline, runtime, embedder, llm) 튜플.
    """
    runtime = load_runtime_config(
        base_config_path,
        provider_config_path,
        experiment_config_path,
        retrieval_config_path,
    )
    embedder = build_embedding_provider(runtime.provider)

    if adapter_path:
        llm = build_llm_provider(runtime.provider, adapter_path=adapter_path)
    else:
        llm = build_llm_provider(runtime.provider)

    vector_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name_for_config(runtime),
    )
    resolved_chunks_path = _resolve_chunks_path(runtime, chunks_path)

    # collection이 비어있으면 자동으로 DB 생성
    if hasattr(vector_store, "count") and vector_store.count() == 0:
        from bidmate_rag.pipelines.build_index import build_index_from_parquet

        if resolved_chunks_path.exists():
            print(f"[자동 빌드] {collection_name_for_config(runtime)}")
            build_index_from_parquet(
                str(resolved_chunks_path),
                embedder=embedder,
                vector_store=vector_store,
            )
        else:
            print("[경고] chunks 파일을 찾을 수 없습니다.")

    resolved_path = _resolve_metadata_path(runtime, metadata_path)
    metadata_store = (
        MetadataStore.from_parquet(resolved_path)
        if resolved_path.exists()
        else MetadataStore(pd.DataFrame())
    )
    calculation_engine = None
    if resolved_path.exists():
        calculation_engine = CalculationEngine(
            CalculationStore.from_parquet(resolved_path, db_path=":memory:")
        )

    sparse_store = None
    if runtime.retrieval.hybrid.enabled and resolved_chunks_path.exists():
        sparse_store = BM25SparseStore.from_parquet(resolved_chunks_path)

    memory = None
    if runtime.retrieval.memory.enabled:
        memory = ConversationMemory(
            max_recent_turns=runtime.retrieval.memory.summary_buffer.max_recent_turns,
            max_summary_chars=runtime.retrieval.memory.summary_buffer.max_summary_chars,
            agency_list=getattr(metadata_store, "agency_list", []),
            slot_enabled=runtime.retrieval.memory.slot_memory.enabled,
        )
    reranker = _load_reranker(runtime.retrieval.reranker_model)
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=metadata_store,
        sparse_store=sparse_store,
        reranker_model=reranker,
        enable_multiturn=runtime.retrieval.enable_multiturn,
        boost_config=runtime.retrieval.boost.model_dump(),
        hybrid_config=runtime.retrieval.hybrid.model_dump(),
        rewrite_llm=llm if runtime.retrieval.enable_multiturn else None,
        rewrite_mode=runtime.retrieval.rewrite.mode,
        rewrite_max_completion_tokens=runtime.retrieval.rewrite.max_completion_tokens,
        rewrite_timeout_seconds=runtime.retrieval.rewrite.timeout_seconds,
        memory=memory,
        debug_trace_enabled=runtime.retrieval.debug_trace.enabled,
    )
    pipeline = RAGChatPipeline(
        retriever=retriever,
        llm=llm,
        memory=memory,
        debug_trace_enabled=runtime.retrieval.debug_trace.enabled,
        calculation_engine=calculation_engine,
    )

    return pipeline, runtime, embedder, llm
