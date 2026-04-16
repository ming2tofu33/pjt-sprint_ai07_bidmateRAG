"""Streamlit UI 공용 헬퍼 함수."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from bidmate_rag.evaluation.dataset import (
    find_latest_metadata_path,
    load_eval_samples,
)
from bidmate_rag.evaluation.pipeline import EvaluationArtifacts, execute_evaluation
from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def list_provider_configs(config_dir: str | Path = "configs/providers") -> list[Path]:
    """프로바이더 설정 YAML 파일 목록을 반환한다.

    Args:
        config_dir: 프로바이더 설정 디렉터리 경로.

    Returns:
        정렬된 YAML 파일 경로 리스트.
    """
    return sorted(Path(config_dir).glob("*.yaml"))

#청킹 설정
def list_chunking_configs(config_dir: str | Path = "configs/chunking") -> list[Path]:
    return sorted(Path(config_dir).glob("*.yaml"))

# 학습된 어댑터 목록
def list_adapters() -> list[Path]:
    """artifacts/training/ 폴더에서 학습된 어댑터 목록 반환."""
    adapter_root = Path("artifacts/training")
    if not adapter_root.exists():
        return []
    return [p for p in sorted(adapter_root.iterdir()) if p.is_dir()]

# 시나리오 A 임베딩 모델 목록
def list_scenario_a_embeddings(
    config_dir: str | Path = "configs/providers/scenario_a/embeddings"
) -> list[Path]:
    return sorted(Path(config_dir).glob("*.yaml"))

# 시나리오 A LLM 모델 목록
def list_scenario_a_llms(
    config_dir: str | Path = "configs/providers/scenario_a/llms"
) -> list[Path]:
    return sorted(Path(config_dir).glob("*.yaml"))

# 시나리오 A용 프로바이더 설정 YAML 동적 생성 함수
def build_scenario_a_provider_config(
    embedding_config_path: str | Path,
    llm_config_path: str | Path,
    tmp_dir: str | Path = "/tmp",
) -> Path:
    """시나리오 A용 임베딩 + LLM yaml을 합쳐서 임시 provider yaml 생성."""
    import yaml
    import tempfile

    embedding_config = yaml.safe_load(Path(embedding_config_path).read_text())
    llm_config = yaml.safe_load(Path(llm_config_path).read_text())

    # 두 yaml 합쳐서 provider yaml 동적 생성
    provider_config = {
        "provider": llm_config.get("provider", "huggingface"),
        "scenario": "scenario_a",
        "model": llm_config.get("model", ""),
        "embedding_model": embedding_config.get("embedding_model", ""),
    }

    if llm_config.get("api_base"):
        provider_config["api_base"] = llm_config["api_base"]

    # 임시 파일로 저장
    tmp_path = Path(tmp_dir) / f"scenario_a_{Path(embedding_config_path).stem}_{Path(llm_config_path).stem}.yaml"
    tmp_path.write_text(yaml.dump(provider_config, allow_unicode=True))
    return tmp_path


def load_benchmark_frames(benchmarks_dir: str | Path = "artifacts/logs/benchmarks") -> pd.DataFrame:
    """벤치마크 결과 parquet 파일들을 하나의 DataFrame으로 합친다.

    Args:
        benchmarks_dir: 벤치마크 결과 저장 디렉터리.

    Returns:
        모든 벤치마크 결과를 합친 DataFrame (없으면 빈 DataFrame).
    """
    benchmark_dir = Path(benchmarks_dir)
    files = sorted(benchmark_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        # 어떤 파일에서 왔는지 추적
        frame["source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_run_records(run_file: str | Path) -> list[dict]:
    """실행 로그 JSONL 파일을 파싱하여 딕셔너리 리스트로 반환한다.

    Args:
        run_file: JSONL 실행 로그 파일 경로.

    Returns:
        각 행을 파싱한 딕셔너리 리스트 (파일 없으면 빈 리스트).
    """
    path = Path(run_file)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_metadata_options(parquet_path: str | Path | None = None) -> dict:
    """사이드바 필터용 메타데이터 옵션을 로딩한다.

    Args:
        parquet_path: cleaned_documents.parquet 경로. None이면 최신 파일 자동 탐지.

    Returns:
        agencies, domains, agency_types 키를 가진 딕셔너리.
    """
    # 경로 미지정 시 최신 실험별 parquet 자동 탐지
    if parquet_path is None:
        parquet_path = find_latest_metadata_path()
    path = Path(parquet_path)
    if not path.exists():
        return {"agencies": [], "domains": [], "agency_types": []}
    df = pd.read_parquet(path)
    return {
        # 발주 기관 목록
        "agencies": sorted(df["발주 기관"].dropna().unique().tolist()),
        # 사업명 기반 도메인 분류
        "domains": sorted(
            df.apply(lambda r: _classify_domain_simple(r.get("사업명", "")), axis=1)
            .unique()
            .tolist()
        )
        if "사업명" in df.columns
        else [],
        # 발주 기관 기반 기관유형 분류
        "agency_types": sorted(
            df.apply(lambda r: _classify_agency_simple(r.get("발주 기관", "")), axis=1)
            .unique()
            .tolist()
        )
        if "발주 기관" in df.columns
        else [],
    }


def _classify_agency_simple(name: str) -> str:
    """기관명으로 기관유형을 간이 분류한다.

    Args:
        name: 발주 기관명.

    Returns:
        기관유형 문자열 (대학교, 공기업/준정부기관, 지방자치단체 등).
    """
    name = str(name)
    if any(k in name for k in ["대학", "학교"]):
        return "대학교"
    if any(k in name for k in ["공사", "공단", "진흥원", "진흥회", "평가원", "정보원"]):
        return "공기업/준정부기관"
    if any(k in name for k in ["시", "도", "군", "구", "광역"]):
        return "지방자치단체"
    if any(k in name for k in ["연구원", "연구소", "과학"]):
        return "연구기관"
    if any(k in name for k in ["부 ", "처 ", "청 ", "위원회"]):
        return "중앙행정기관"
    return "기타"


def _classify_domain_simple(name: str) -> str:
    """사업명으로 사업도메인을 간이 분류한다.

    Args:
        name: 사업명.

    Returns:
        사업도메인 문자열 (교육/학습, 안전/재난, 경영/행정 등).
    """
    name = str(name)
    if any(k in name for k in ["교육", "이러닝", "학습", "학사"]):
        return "교육/학습"
    if any(k in name for k in ["안전", "재난", "관제", "선량"]):
        return "안전/재난"
    if any(k in name for k in ["홈페이지", "포털", "웹"]):
        return "웹/포털"
    if any(k in name for k in ["ERP", "그룹웨어", "경영"]):
        return "경영/행정"
    if any(k in name for k in ["GIS", "지도", "수문"]):
        return "공간정보/GIS"
    if any(k in name for k in ["의료", "바이오", "병원"]):
        return "의료/바이오"
    return "기타 정보시스템"


def run_live_query(
    question: str,
    provider_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
    experiment_config_path: str | Path | None = None,
    top_k: int = 5,
    manual_filters: dict | None = None,
    metadata_filter: dict | None = None,
    chat_history: list[dict] | None = None,
    system_prompt: str | None = None,
    max_context_chars: int = 8000,
    embedding_config_path: str | Path | None = None, # 시나리오 A 임베딩 설정 경로 (선택)
    llm_config_path: str | Path | None = None, # 시나리오 A LLM 설정 경로 (선택)
    adapter_path: str | Path | None = None, # 어댑터 경로 (선택)
):
    """단일 RAG 쿼리를 실행한다 (라이브 데모 + 디버그 탭 공용).

    Args:
        question: 사용자 질문.
        provider_config_path: 프로바이더 설정 YAML 경로.
        base_config_path: 기본 설정 YAML 경로.
        experiment_config_path: 실험 설정 YAML 경로 (선택).
        top_k: 검색할 청크 수.
        manual_filters: 사이드바 수동 필터 (한국어 키).
        metadata_filter: 평가셋의 explicit override 필터.
        chat_history: 멀티턴 대화 이력.
        system_prompt: 시스템 프롬프트 오버라이드.
        max_context_chars: LLM에 전달할 최대 컨텍스트 길이.

    Returns:
        GenerationResult (답변, 검색 청크, 토큰 사용량 등).
    """
    # 시나리오 A: 임베딩 + LLM yaml로 provider yaml 동적 생성
    if embedding_config_path and llm_config_path:
        provider_config_path = build_scenario_a_provider_config(
            embedding_config_path, llm_config_path
        )
    # 1. 런타임 파이프라인 조립
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
        adapter_path=adapter_path, # 어댑터 경로 전달
    )
    # 2. 시스템 프롬프트 오버라이드 적용
    if system_prompt:
        pipeline.system_prompt = system_prompt

    # 3. LLM 생성 설정
    gen_config = {"max_context_chars": max_context_chars}

    # 4. 메타데이터 필터 결정
    #    explicit metadata_filter 우선, 없으면 manual_filters에서 변환
    resolved_filter: dict | None = None
    if metadata_filter is not None:
        # explicit (debug_tab 등) — 빈 dict는 "필터 없음" 명시
        resolved_filter = dict(metadata_filter) if metadata_filter else {}
    elif manual_filters:
        # 라이브 데모 사이드바 필터
        if manual_filters.get("_no_filter"):
            resolved_filter = {}
        else:
            resolved_filter = dict(manual_filters)
        
    run_id = f"live-{uuid4().hex[:8]}"

    # 5. RAG 파이프라인 실행 (검색 → LLM 생성)
    return pipeline.answer(
        question,
        top_k=top_k,
        metadata_filter=resolved_filter,
        chat_history=chat_history,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=run_id,
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
        generation_config=gen_config,
    )


def run_benchmark_experiment(
    evaluation_path: str | Path,
    provider_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
    experiment_config_path: str | Path | None = None,
    runs_dir: str | Path = "artifacts/logs/runs",
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
    *,
    run_id: str | None = None,
    skip_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    judge_v2: bool = False,
    progress_callback=None,
    embedding_config_path: str | Path | None = None, # 시나리오 A 임베딩 설정 경로 (선택)
    llm_config_path: str | Path | None = None, # 시나리오
    adapter_path: str | Path | None = None, # 어댑터 경로 
    top_k: int = 5, # 검색할 청크 수 (기본값 5, ExperimentConfig.retrieval_top_k로 오버라이드 가능)
) -> EvaluationArtifacts:
    """런타임 파이프라인을 조립하고 전체 평가를 실행한다.

    Args:
        evaluation_path: 평가셋 파일 경로.
        provider_config_path: 프로바이더 설정 YAML 경로.
        base_config_path: 기본 설정 YAML 경로.
        experiment_config_path: 실험 설정 YAML 경로 (선택).
        runs_dir: 실행 로그 저장 디렉터리.
        benchmarks_dir: 벤치마크 결과 저장 디렉터리.
        run_id: 실행 ID 오버라이드 (미지정 시 자동 생성).
        skip_judge: LLM 판정 평가 건너뛰기 여부.
        judge_model: LLM 판정에 사용할 모델명.
        progress_callback: 진행 상황 콜백 (Streamlit 프로그레스 바 등).

    Returns:
        EvaluationArtifacts (실행 결과, 산출물 경로, 지표 등).
    """
    # 시나리오 A: 임베딩 + LLM yaml로 provider yaml 동적 생성
    if embedding_config_path and llm_config_path:
        provider_config_path = build_scenario_a_provider_config(
            embedding_config_path, llm_config_path
        )

    # 1. 런타임 파이프라인 조립
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
        adapter_path=adapter_path, # 어댑터 경로 전달
    )
    # 2. 평가셋 로딩 ("다중" 필터 → $in 변환을 위해 기관 목록 전달)
    agency_list = getattr(pipeline.retriever.metadata_store, "agency_list", [])
    samples = load_eval_samples(evaluation_path, agency_list=agency_list)

    # 3. 평가 실행 (검색 → LLM 생성 → 지표 계산 → 산출물 저장)
    return execute_evaluation(
        samples,
        pipeline=pipeline,
        runtime=runtime,
        embedder=embedder,
        eval_path=str(evaluation_path),
        config_paths={
            "base": str(base_config_path),
            "provider": str(provider_config_path),
            "experiment": str(experiment_config_path) if experiment_config_path else None,
        },
        runs_dir=runs_dir,
        benchmarks_dir=benchmarks_dir,
        run_id=run_id,
        skip_judge=skip_judge,
        judge_model=judge_model,
        judge_v2=judge_v2,
        progress_callback=progress_callback,
        top_k=top_k, # 검색할 청크 수 전달
    )
