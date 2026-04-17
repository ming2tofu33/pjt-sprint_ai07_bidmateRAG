"""Tests for evaluation/schema_validator.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.evaluation.schema_validator import (
    render_validation_report,
    validate_eval_samples,
)
from bidmate_rag.schema import EvalSample


def _make_cleaned_documents(tmp_path: Path) -> Path:
    """검증에 쓸 fake cleaned_documents.parquet 생성."""
    df = pd.DataFrame(
        [
            {
                "발주 기관": "한국가스공사",
                "사업명": "차세대 ERP 구축",
                "사업도메인": "경영/행정",
                "기관유형": "공기업/준정부기관",
                "공개연도": 2024,
                "사업 금액": 14_107_009_000,
                "파일명": "한국가스공사_차세대 ERP 구축.hwp",
            },
            {
                "발주 기관": "고려대학교",
                "사업명": "차세대 포털·학사 정보시스템",
                "사업도메인": "교육/학습",
                "기관유형": "대학교",
                "공개연도": 2024,
                "사업 금액": 11_270_000_000,
                "파일명": "고려대학교_차세대 포털·학사 정보시스템.pdf",
            },
        ]
    )
    out = tmp_path / "cleaned_documents.parquet"
    df.to_parquet(out, index=False)
    return out


def _make_sample(
    qid: str,
    question: str = "dummy",
    metadata_filter: dict | None = None,
    expected_doc_titles: list[str] | None = None,
    history: object = None,
) -> EvalSample:
    metadata: dict = {}
    if metadata_filter is not None:
        metadata["metadata_filter"] = metadata_filter
    if history is not None:
        metadata["history"] = history
    return EvalSample(
        question_id=qid,
        question=question,
        expected_doc_titles=expected_doc_titles or [],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 유효한 케이스
# ---------------------------------------------------------------------------


def test_valid_sample_yields_zero_issues(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(
            "Q1",
            metadata_filter={"발주 기관": "한국가스공사"},
            expected_doc_titles=["한국가스공사_차세대 ERP 구축.hwp"],
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert report.errors == []
    assert report.warnings == []
    assert report.is_valid(strict=True)


# ---------------------------------------------------------------------------
# metadata_filter 키/값 검증
# ---------------------------------------------------------------------------


def test_metadata_filter_unknown_key_warns(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", metadata_filter={"vendor": "어떤사"})]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.warnings) == 1
    assert "vendor" in report.warnings[0].message
    assert report.errors == []


def test_metadata_filter_unknown_value_warns(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", metadata_filter={"발주 기관": "다중"})]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.warnings) == 1
    assert "다중" in report.warnings[0].message
    assert "발주 기관" in report.warnings[0].message


def test_metadata_filter_operator_value_skipped(tmp_path):
    """$gte/$in 같은 operator 값은 매칭 검증을 skip해야 함."""
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample("Q1", metadata_filter={"사업 금액": {"$gte": 1_000_000_000}})
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    # operator 값은 검증 skip → warning 0
    assert report.warnings == []


def test_metadata_filter_multiple_keys_each_validated(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(
            "Q1",
            metadata_filter={
                "발주 기관": "한국가스공사",  # OK
                "사업도메인": "없는도메인",  # warning
            },
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.warnings) == 1
    assert "사업도메인" in report.warnings[0].message


# ---------------------------------------------------------------------------
# ground_truth_docs 검증
# ---------------------------------------------------------------------------


def test_ground_truth_doc_unknown_warns(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample("Q1", expected_doc_titles=["없는 파일.pdf"])
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.warnings) == 1
    assert "없는 파일.pdf" in report.warnings[0].message


def test_ground_truth_doc_matches_사업명(tmp_path):
    """ground_truth_docs가 사업명과 매칭되어도 OK."""
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", expected_doc_titles=["차세대 ERP 구축"])]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert report.warnings == []


def test_multidoc_question_without_anchors_warns(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(
            "Q1",
            question="세 사업 중 문서 내에 법제도 준수 여부 점검표를 포함한 사업은 무엇입니까?",
            expected_doc_titles=[
                "한국가스공사_차세대 ERP 구축.hwp",
                "고려대학교_차세대 포털·학사 정보시스템.pdf",
            ],
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert any(
        issue.field == "question" and "underspecified" in issue.message
        for issue in report.warnings
    )


def test_multidoc_question_with_visible_anchors_does_not_warn(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(
            "Q1",
            question="한국가스공사와 고려대학교 사업을 비교할 때 예산 규모가 큰 곳은 어디입니까?",
            expected_doc_titles=[
                "한국가스공사_차세대 ERP 구축.hwp",
                "고려대학교_차세대 포털·학사 정보시스템.pdf",
            ],
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert not any(issue.field == "question" for issue in report.warnings)


def test_multidoc_question_anchor_matches_ignoring_fullwidth_parens(tmp_path):
    """파일명/기관명에 전각 괄호(）, 질문에 반각 괄호()가 섞여도 매칭되어야 함 (Q028 회귀)."""
    df = pd.DataFrame(
        [
            {
                "발주 기관": "(사)벤처기업협회",
                "사업명": "벤처확인종합관리시스템",
                "사업도메인": "경영/행정",
                "기관유형": "협회",
                "공개연도": 2024,
                "사업 금액": 352_000_000,
                "파일명": "(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .hwp",
            },
            {
                "발주 기관": "(사）한국대학스포츠협의회",
                "사업명": "KUSF 체육특기자 경기기록 관리시스템",
                "사업도메인": "교육/학습",
                "기관유형": "협회",
                "공개연도": 2024,
                "사업 금액": 150_000_000,
                "파일명": "(사）한국대학스포츠협의회_KUSF 체육특기자 경기기록 관리시스템 개발.hwp",
            },
        ]
    )
    cleaned = tmp_path / "cleaned_documents.parquet"
    df.to_parquet(cleaned, index=False)

    samples = [
        _make_sample(
            "Q028",
            question=(
                "(사)벤처기업협회와 (사)한국대학스포츠협의회의 사업은 특정 목적을 위한 "
                "세부 제도 지원 또는 평가 체계 구축을 포함합니다."
            ),
            expected_doc_titles=[
                "(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .hwp",
                "(사）한국대학스포츠협의회_KUSF 체육특기자 경기기록 관리시스템 개발.hwp",
            ],
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert not any(
        issue.field == "question" and "underspecified" in issue.message
        for issue in report.warnings
    )


# ---------------------------------------------------------------------------
# history 형식 검증
# ---------------------------------------------------------------------------


def test_history_must_be_list(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", history="not a list")]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.errors) == 1
    assert "must be a list" in report.errors[0].message


def test_history_turn_must_be_dict(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", history=[{"role": "user"}, "wrong"])]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert len(report.errors) == 1
    assert "turn[1]" in report.errors[0].message


def test_history_valid_list_of_dicts(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(
            "Q1",
            history=[
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "안녕하세요"},
            ],
        )
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert report.errors == []


# ---------------------------------------------------------------------------
# is_valid strict semantics
# ---------------------------------------------------------------------------


def test_is_valid_warning_only_loose_passes(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", metadata_filter={"발주 기관": "다중"})]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert report.is_valid(strict=False) is True
    assert report.is_valid(strict=True) is False


def test_is_valid_error_fails_both(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [_make_sample("Q1", history="not a list")]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    assert report.is_valid(strict=False) is False
    assert report.is_valid(strict=True) is False


# ---------------------------------------------------------------------------
# fallback: cleaned_documents.parquet 없음
# ---------------------------------------------------------------------------


def test_missing_parquet_skips_value_checks_but_validates_history(tmp_path):
    """parquet 없으면 키/값 검증은 skip, history 형식 검증은 계속."""
    samples = [
        _make_sample(
            "Q1",
            metadata_filter={"발주 기관": "anything"},  # 매칭 검증 skip
            history="invalid",  # error
        )
    ]
    report = validate_eval_samples(
        samples, cleaned_documents_path=tmp_path / "missing.parquet"
    )
    # metadata_filter 검증 0, history error 1
    assert len(report.errors) == 1
    assert report.errors[0].field == "history"


# ---------------------------------------------------------------------------
# render_validation_report
# ---------------------------------------------------------------------------


def test_render_no_issues(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample("Q1", metadata_filter={"발주 기관": "한국가스공사"})
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    text = render_validation_report(report)
    assert "✓ All samples valid" in text


def test_render_with_issues_shows_icons(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample("Q1", metadata_filter={"발주 기관": "다중"}),
        _make_sample("Q2", history="bad"),
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    text = render_validation_report(report)
    assert "❌" in text
    assert "⚠️" in text
    assert "Q1" in text
    assert "Q2" in text


def test_render_truncates_long_list(tmp_path):
    cleaned = _make_cleaned_documents(tmp_path)
    samples = [
        _make_sample(f"Q{i}", metadata_filter={"발주 기관": "다중"})
        for i in range(30)
    ]
    report = validate_eval_samples(samples, cleaned_documents_path=cleaned)
    text = render_validation_report(report, max_lines=5)
    assert "and 25 more" in text
