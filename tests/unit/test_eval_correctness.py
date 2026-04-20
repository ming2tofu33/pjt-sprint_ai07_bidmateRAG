"""Tests for eval correctness fixes (top_k / metadata_filter / history /
chunking isolation / parquet append).

Each test targets a single regression that the fix branch addressed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from bidmate_rag.config.settings import (
    ExperimentConfig,
    ProjectConfig,
    ProviderConfig,
    RuntimeConfig,
)
from bidmate_rag.evaluation.dataset import (
    EVAL_FILTER_KEY_MAP,
    load_eval_samples,
    normalize_metadata_filter,
)
from bidmate_rag.evaluation.runner import persist_benchmark_summary
from bidmate_rag.pipelines.runtime import collection_name_for_config
from bidmate_rag.retrieval.agency_matching import extract_agencies_from_text
from bidmate_rag.retrieval.retriever import RAGRetriever

# ---------------------------------------------------------------------------
# Phase 2: normalize_metadata_filter
# ---------------------------------------------------------------------------


def test_normalize_metadata_filter_maps_english_to_korean_keys():
    out = normalize_metadata_filter({"agency": "한국가스공사", "year": "2024"})
    assert out == {"발주 기관": "한국가스공사", "공개연도": 2024}


def test_normalize_metadata_filter_institution_aliased_to_agency():
    out = normalize_metadata_filter({"institution": "한국철도공사"})
    assert out == {"발주 기관": "한국철도공사"}


def test_extract_agencies_from_text_supports_mixed_legal_prefix_parenthesis():
    matched = extract_agencies_from_text(
        "한국대학스포츠협의회",
        ["(사）한국대학스포츠협의회"],
    )
    assert matched == ["(사）한국대학스포츠협의회"]


def test_extract_agencies_from_text_strips_trailing_parenthetical_suffix():
    matched = extract_agencies_from_text(
        "한국철도공사",
        ["한국철도공사 (용역)"],
    )
    assert matched == ["한국철도공사 (용역)"]


def test_extract_agencies_from_text_strips_leading_year_prefix():
    matched = extract_agencies_from_text(
        "구미 아시아육상경기선수권대회 조직위원회",
        ["2025 구미 아시아육상경기선수권대회 조직위원회"],
    )
    assert matched == ["2025 구미 아시아육상경기선수권대회 조직위원회"]


def test_normalize_metadata_filter_year_string_coerced_to_int():
    out = normalize_metadata_filter({"year": "2023"})
    assert out == {"공개연도": 2023}
    assert isinstance(out["공개연도"], int)


def test_normalize_metadata_filter_legacy_domain_value_is_reinterpreted_as_agency():
    out = normalize_metadata_filter(
        {"domain": "(사)벤처기업협회", "year": "2024"},
        agency_list=["(사)벤처기업협회", "(사)부산국제영화제"],
    )
    assert out == {"발주 기관": "(사)벤처기업협회", "공개연도": 2024}


def test_normalize_metadata_filter_multi_year_string_uses_in_operator():
    out = normalize_metadata_filter({"year": "2024, 2025"})
    assert out == {"공개연도": {"$in": [2024, 2025]}}


def test_normalize_metadata_filter_drops_unrecognized_legacy_domain_value():
    out = normalize_metadata_filter(
        {"domain": "평택시", "year": "2024"},
        agency_list=["경기도 평택시", "경기도 안양시"],
    )
    assert out == {"공개연도": 2024}


def test_normalize_metadata_filter_legacy_domain_uses_metadata_alias_map():
    out = normalize_metadata_filter(
        {"domain": "평택시", "year": "2024"},
        agency_list=["경기도 평택시", "경기도 안양시"],
        agency_alias_map={"평택시": "경기도 평택시"},
    )
    assert out == {"발주 기관": "경기도 평택시", "공개연도": 2024}


def test_normalize_metadata_filter_agency_field_uses_metadata_alias_map():
    out = normalize_metadata_filter(
        {"institution": "한국철도공사"},
        agency_list=["한국철도공사 (용역)"],
        agency_alias_map={"한국철도공사": "한국철도공사 (용역)"},
    )
    assert out == {"발주 기관": "한국철도공사 (용역)"}


def test_normalize_metadata_filter_none_or_empty_returns_none():
    assert normalize_metadata_filter(None) is None
    assert normalize_metadata_filter({}) is None


def test_normalize_metadata_filter_unknown_key_passes_through():
    # 매핑에 없는 키는 그대로 보존 (warning 로그만 발생)
    out = normalize_metadata_filter({"weird_key": "x"})
    assert out == {"weird_key": "x"}


def test_normalize_metadata_filter_multi_agency_uses_legal_name_normalization():
    out = normalize_metadata_filter(
        {"agency": "다중"},
        question="나노종합기술원의 스마트 팹 서비스 사업과 부산국제영화제의 BIFF 온라인서비스 재개발 사업을 비교해 주세요.",
        agency_list=["나노종합기술원", "(사)부산국제영화제"],
    )
    assert out == {"발주 기관": {"$in": ["나노종합기술원", "(사)부산국제영화제"]}}


def test_normalize_metadata_filter_multi_agency_warns_when_under_matched(caplog):
    with caplog.at_level("WARNING"):
        out = normalize_metadata_filter(
            {"agency": "다중"},
            question="나노종합기술원 사업을 비교해 주세요.",
            agency_list=["나노종합기술원", "(사)부산국제영화제"],
        )

    assert out == {"발주 기관": {"$in": ["나노종합기술원"]}}
    assert "matched fewer than 2 agencies" in caplog.text


def test_normalize_metadata_filter_matches_korean_pronunciation_alias_for_koica():
    out = normalize_metadata_filter(
        {"agency": "다중"},
        question="코이카 전자조달의 우즈베키스탄 방송시스템 사업과 아시아물위원회 사무국의 스마트 관개시스템 사업을 비교해 주세요.",
        agency_list=["KOICA 전자조달", "사단법인아시아물위원회사무국"],
    )
    assert out == {
        "발주 기관": {"$in": ["KOICA 전자조달", "사단법인아시아물위원회사무국"]}
    }


def test_eval_filter_key_map_covers_known_eval_csv_keys():
    # 평가셋 580개 샘플에서 발견된 키 (agency/institution/project/domain/year)
    for key in ("agency", "institution", "project", "domain", "year"):
        assert key in EVAL_FILTER_KEY_MAP


def test_load_eval_samples_resolves_json_ground_truth_to_canonical_ingest_file(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    duplicates_map_path = tmp_path / "duplicates_map.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {"공고 번호": "1", "발주 기관": "Agency", "파일명": "BioIN_project.hwp"},
            {"공고 번호": "2", "발주 기관": "Agency", "파일명": "Agency_project.hwp"},
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "duplicate_group_id": "DUP-001",
                "source_file": "BioIN_project.hwp",
                "canonical_file": "Agency_project.hwp",
                "is_duplicate": True,
                "resolved_agency": "Agency",
            },
            {
                "duplicate_group_id": "DUP-001",
                "source_file": "Agency_project.hwp",
                "canonical_file": "Agency_project.hwp",
                "is_duplicate": False,
                "resolved_agency": "Agency",
            },
        ]
    ).to_csv(duplicates_map_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "ground_truth_docs": '["BioIN_project.json"]',
                "metadata_filter": '{"domain": "Agency", "year": "2024"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["Agency"],
        metadata_path=metadata_path,
        duplicates_map_path=duplicates_map_path,
    )

    assert samples[0].expected_doc_titles == ["Agency_project.hwp"]
    assert samples[0].metadata["metadata_filter"] == {
        "발주 기관": "Agency",
        "공개연도": 2024,
    }


def test_load_eval_samples_resolves_legacy_filename_prefix_to_canonical_agency(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국한의학연구원",
                "파일명": "국가과학기술지식정보서비스_통합정보시스템 고도화 용역.hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"domain": "국가과학기술지식정보서비스"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국한의학연구원"],
        metadata_path=metadata_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "발주 기관": "한국한의학연구원"
    }


def test_load_eval_samples_accepts_plain_string_ground_truth_doc(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국철도공사 (용역)",
                "파일명": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "ground_truth_docs": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역",
                "metadata_filter": '{"institution": "한국철도공사"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국철도공사 (용역)"],
        metadata_path=metadata_path,
    )

    assert samples[0].expected_doc_titles == [
        "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp"
    ]
    assert samples[0].metadata["metadata_filter"] == {
        "발주 기관": "한국철도공사 (용역)"
    }


def test_load_eval_samples_resolves_project_alias_from_manual_alias_map(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    project_alias_path = tmp_path / "project_alias_map.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국어촌어항공단",
                "사업명": "한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역",
                "파일명": "한국어촌어항공단_한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역.hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "canonical_agency": "한국어촌어항공단",
                "canonical_project": "한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역",
                "project_alias": "경영관리시스템 기능 고도화",
                "alias_type": "핵심축약",
                "source": "manual",
                "enabled": True,
                "note": "",
            }
        ]
    ).to_csv(project_alias_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"project": "경영관리시스템 기능 고도화", "institution": "한국어촌어항공단"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국어촌어항공단"],
        metadata_path=metadata_path,
        project_alias_path=project_alias_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "사업명": "한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역",
        "발주 기관": "한국어촌어항공단",
    }


def test_load_eval_samples_resolves_project_alias_from_metadata_substring(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "대전대학교",
                "사업명": "대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전산 유지보수 용역",
                "파일명": "대전대학교_대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전산 유지보수 용역.hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"project": "다층적 융합 학습경험 플랫폼(MILE)", "institution": "대전대학교"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["대전대학교"],
        metadata_path=metadata_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "사업명": "대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전산 유지보수 용역",
        "발주 기관": "대전대학교",
    }


def test_load_eval_samples_resolves_project_alias_from_metadata_tokens(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국어촌어항공단",
                "사업명": "한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역",
                "파일명": "한국어촌어항공단_한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역.hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"project": "경영관리시스템 기능 고도화", "institution": "한국어촌어항공단"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국어촌어항공단"],
        metadata_path=metadata_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "사업명": "한국어촌어항공단 경영관리시스템(ERP·GW) 기능 고도화 용역",
        "발주 기관": "한국어촌어항공단",
    }


def test_load_eval_samples_prefers_shorter_project_when_fuzzy_matches_overlap(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국보건산업진흥원",
                "사업명": "의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업",
                "파일명": "한국보건산업진흥원_의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업.hwp",
            },
            {
                "공고 번호": "2",
                "발주 기관": "한국보건산업진흥원",
                "사업명": "의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업(2차)",
                "파일명": "BioIN_의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업(2차).hwp",
            },
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"project": "의료기기산업 종합정보시스템(정보관리기관) 기능개선", "institution": "한국보건산업진흥원"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국보건산업진흥원"],
        metadata_path=metadata_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "사업명": "의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업",
        "발주 기관": "한국보건산업진흥원",
    }


def test_load_eval_samples_drops_unresolved_project_filter(tmp_path):
    metadata_path = tmp_path / "data_list.csv"
    eval_path = tmp_path / "eval.csv"

    pd.DataFrame(
        [
            {
                "공고 번호": "1",
                "발주 기관": "한국보건산업진흥원",
                "사업명": "의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업(2차)",
                "파일명": "한국보건산업진흥원_의료기기산업 종합정보시스템(정보관리기관) 기능개선 사업(2차).hwp",
            }
        ]
    ).to_csv(metadata_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "id": "Q001",
                "question": "질문",
                "metadata_filter": '{"project": "전혀 없는 사업명", "institution": "한국보건산업진흥원"}',
            }
        ]
    ).to_csv(eval_path, index=False, encoding="utf-8-sig")

    samples = load_eval_samples(
        eval_path,
        agency_list=["한국보건산업진흥원"],
        metadata_path=metadata_path,
    )

    assert samples[0].metadata["metadata_filter"] == {
        "발주 기관": "한국보건산업진흥원"
    }


# ---------------------------------------------------------------------------
# Phase 2: RAGRetriever.retrieve metadata_filter override semantics
# ---------------------------------------------------------------------------


def _make_retriever_with_mock_store():
    vector_store = MagicMock()
    vector_store.query.return_value = []
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8
    metadata_store = MagicMock()
    metadata_store.agency_list = ["한국가스공사"]
    metadata_store.find_relevant_docs.return_value = []
    return (
        RAGRetriever(vector_store=vector_store, embedder=embedder, metadata_store=metadata_store),
        vector_store,
    )


def test_retriever_explicit_metadata_filter_bypasses_auto_extraction():
    retriever, vector_store = _make_retriever_with_mock_store()
    explicit = {"발주 기관": "한국가스공사", "공개연도": 2024}
    retriever.retrieve("아무 질문", metadata_filter=explicit)
    call = vector_store.query.call_args
    assert call.kwargs["where"] == explicit


def test_retriever_explicit_metadata_filter_adds_history_doc_shortlist_when_needed():
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.metadata_store.find_relevant_docs.return_value = ["doc-1.hwp", "doc-2.hwp"]
    retriever.retrieve(
        "그렇다면 이 사업의 예산은 얼마야?",
        metadata_filter={"발주 기관": "한국가스공사"},
        chat_history=[{"role": "user", "content": "한국가스공사 ERP 구축 사업 설명해줘"}],
    )
    assert len(vector_store.query.call_args_list) >= 2
    first_call = vector_store.query.call_args_list[0]
    fallback_call = vector_store.query.call_args_list[-1]
    assert first_call.kwargs["where"] == {
        "발주 기관": "한국가스공사",
        "파일명": {"$in": ["doc-1.hwp", "doc-2.hwp"]},
    }
    assert fallback_call.kwargs["where"] == {"발주 기관": "한국가스공사"}


def test_retriever_keeps_existing_doc_constraint_when_history_present():
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve(
        "그렇다면 이 사업의 예산은 얼마야?",
        metadata_filter={"발주 기관": "한국가스공사", "파일명": {"$in": ["erp.hwp"]}},
        chat_history=[{"role": "user", "content": "한국가스공사 ERP 구축 사업 설명해줘"}],
    )
    call = vector_store.query.call_args
    assert call.kwargs["where"] == {
        "발주 기관": "한국가스공사",
        "파일명": {"$in": ["erp.hwp"]},
    }


def test_retriever_empty_dict_filter_means_no_filter():
    """metadata_filter={}는 자동 추출까지 비활성화 (Streamlit '필터 없음' 모드)."""
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("한국가스공사 사업 알려줘", metadata_filter={})
    call = vector_store.query.call_args
    assert call.kwargs["where"] is None


def test_retriever_none_filter_falls_back_to_auto_extraction():
    """metadata_filter=None은 legacy 자동 추출 동작."""
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("한국가스공사 사업", metadata_filter=None)
    call = vector_store.query.call_args
    # 자동 추출이 발주 기관을 잡았어야 함
    where = call.kwargs["where"]
    assert where is not None
    assert where.get("발주 기관") == "한국가스공사"


def test_retriever_without_reranker_expands_query_pool():
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("질문", top_k=12)
    assert vector_store.query.call_args.kwargs["top_k"] == 36


# ---------------------------------------------------------------------------
# Phase 3: collection_name_for_config mode-aware isolation
# ---------------------------------------------------------------------------


def _make_runtime(exp_name: str, mode: str, explicit: str | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(
            provider="openai",
            model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
            collection_name=explicit,
        ),
        experiment=ExperimentConfig(name=exp_name, mode=mode),
    )


def test_collection_generation_only_shares_explicit_collection():
    rt = _make_runtime("generation-compare", "generation_only", "bidmate-shared")
    assert collection_name_for_config(rt) == "bidmate-shared"


def test_collection_full_rag_isolates_with_prefix_when_explicit():
    rt = _make_runtime("chunk_500_100", "full_rag", "bidmate-shared")
    assert collection_name_for_config(rt) == "chunk_500_100-bidmate-shared"


def test_collection_full_rag_isolates_via_default_format():
    rt = _make_runtime("chunk_500_100", "full_rag", None)
    name = collection_name_for_config(rt)
    assert "chunk_500_100" in name
    assert name == "bidmate-chunk_500_100-openai-text-embedding-3-small"


def test_collection_ad_hoc_preserves_legacy_explicit_name():
    rt = _make_runtime("ad-hoc", "full_rag", "bidmate-legacy")
    assert collection_name_for_config(rt) == "bidmate-legacy"


# ---------------------------------------------------------------------------
# Phase 4: persist_benchmark_summary append-or-replace
# ---------------------------------------------------------------------------


def _summary(run_id: str, hit: float) -> dict:
    return {
        "experiment_name": "exp",
        "run_id": run_id,
        "scenario": "openai",
        "provider_label": "openai:gpt-5-mini",
        "num_samples": 2,
        "avg_latency_ms": 1000.0,
        "total_cost_usd": 0.001,
        "hit_rate@5": hit,
    }


def test_persist_benchmark_summary_first_write_creates_file(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run-1"


def test_persist_benchmark_summary_appends_new_run_ids(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-2", 0.8)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-3", 1.0)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert set(df["run_id"]) == {"run-1", "run-2", "run-3"}
    assert len(df) == 3


def test_persist_benchmark_summary_replaces_existing_run_id(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-1", 0.9)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert len(df) == 1
    assert df.iloc[0]["hit_rate@5"] == 0.9  # 마지막 값으로 replace


def test_persist_benchmark_summary_provider_compare_scenario(tmp_path):
    """generation_compare처럼 같은 experiment에 여러 provider를 순차로 평가:
    이전에는 마지막 provider만 남았지만 이제 전부 보존되어야 함.
    """
    persist_benchmark_summary([_summary("nano-run", 0.6)], tmp_path, "generation-compare")
    persist_benchmark_summary([_summary("mini-run", 0.7)], tmp_path, "generation-compare")
    persist_benchmark_summary([_summary("full-run", 0.8)], tmp_path, "generation-compare")
    df = pd.read_parquet(tmp_path / "generation-compare.parquet")
    assert len(df) == 3
    assert set(df["run_id"]) == {"nano-run", "mini-run", "full-run"}
