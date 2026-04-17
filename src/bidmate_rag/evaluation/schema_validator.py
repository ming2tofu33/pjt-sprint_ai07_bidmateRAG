"""평가셋 스키마 검증.

평가셋 작성자가 만든 metadata_filter / ground_truth_docs / history가 실제
ChromaDB 메타데이터와 매칭되는지 사전에 확인합니다. 어제 발견한 케이스 (예:
``metadata_filter={"domain": "(사)벤처기업협회"}``처럼 코드는 통과하지만 매칭 0건)
같은 침묵 실패를 평가 시작 전에 잡습니다.

기본 정책: ``warn`` (경고만 표시하고 평가 진행). CI나 안전한 실험에서는
``--strict`` 옵션으로 fail.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bidmate_rag.retrieval.agency_matching import extract_agencies_from_text
from bidmate_rag.schema import EvalSample

logger = logging.getLogger(__name__)


# cleaned_documents.parquet에서 검증에 사용할 컬럼들
_KNOWN_METADATA_COLUMNS = (
    "발주 기관",
    "사업명",
    "사업도메인",
    "기관유형",
    "공개연도",
    "사업 금액",
    "파일명",
)


@dataclass
class ValidationIssue:
    sample_id: str
    severity: str  # "warning" | "error"
    field: str
    message: str


@dataclass
class ValidationReport:
    total_samples: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def is_valid(self, strict: bool = False) -> bool:
        """기본 — error가 없으면 True. ``strict=True`` 시 warning도 차단."""
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


def _load_known_metadata(
    cleaned_documents_path: Path | str = "data/processed/cleaned_documents.parquet",
) -> dict[str, set]:
    """ChromaDB에 들어갈 청크 메타데이터의 unique 값을 사전 로드.

    평가셋의 metadata_filter / ground_truth_docs 값이 실제 데이터에 존재하는지
    확인하는 데 사용됩니다. 파일이 없으면 빈 dict를 반환해 키/값 검증을 skip
    (history 형식 검증 등은 계속 동작).
    """
    path = Path(cleaned_documents_path)
    if not path.exists():
        logger.warning(
            "cleaned_documents.parquet not found at %s — skipping value-level checks",
            path,
        )
        return {}
    df = pd.read_parquet(path)
    known: dict[str, set] = {}
    for col in _KNOWN_METADATA_COLUMNS:
        if col in df.columns:
            known[col] = set(df[col].dropna().tolist())
    return known


def _load_doc_to_agency_map(
    cleaned_documents_path: Path | str = "data/processed/cleaned_documents.parquet",
) -> dict[str, str]:
    """Build a filename/stem to agency lookup for eval-set diagnostics."""

    path = Path(cleaned_documents_path)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "파일명" not in df.columns or "발주 기관" not in df.columns:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        file_name = str(row.get("파일명") or "").strip()
        agency = str(row.get("발주 기관") or "").strip()
        if not file_name or not agency:
            continue
        mapping[file_name] = agency
        mapping[Path(file_name).stem] = agency
    return mapping


def _warn_if_multidoc_question_is_underspecified(
    *,
    sample: EvalSample,
    issues: list[ValidationIssue],
    doc_to_agency: dict[str, str],
) -> None:
    """Warn when a multi-doc eval question lacks visible anchors."""

    titles = sample.expected_doc_titles or []
    metadata = sample.metadata or {}
    if len(titles) < 2:
        return
    if metadata.get("metadata_filter") or metadata.get("history"):
        return

    agencies = {
        doc_to_agency.get(title) or doc_to_agency.get(Path(title).stem, "")
        for title in titles
    }
    agencies.discard("")

    question = sample.question or ""
    mentioned_agencies = set(extract_agencies_from_text(question, list(agencies)))
    quoted_segments = re.findall(r'"[^"]+"|\'[^\']+\'', question)

    if len(mentioned_agencies) >= 2 or len(quoted_segments) >= 2:
        return

    issues.append(
        ValidationIssue(
            sample.question_id,
            "warning",
            "question",
            "multi-doc question may be underspecified: no metadata_filter/history and not enough document anchors in the question",
        )
    )


def validate_eval_samples(
    samples: list[EvalSample],
    cleaned_documents_path: Path | str = "data/processed/cleaned_documents.parquet",
) -> ValidationReport:
    """평가셋 샘플 리스트의 metadata_filter / ground_truth_docs / history 검증.

    검증 항목:

    1. **metadata_filter 키** — `EVAL_FILTER_KEY_MAP`에 의해 정규화된 후 ChromaDB
       메타데이터 컬럼에 실제 존재하는지 (`발주 기관`, `사업도메인` 등)
    2. **metadata_filter 값** — 단일 값(str/int/float)이면 해당 컬럼의 unique
       값에 존재하는지 ($in / $gte 등 operator는 값 매칭을 skip)
    3. **ground_truth_docs** (= ``expected_doc_titles``) — 각 값이 ``파일명``
       또는 ``사업명`` set에 존재하는지
    4. **history 형식** — list[dict] 형태인지 (어긋나면 error)
    """
    known = _load_known_metadata(cleaned_documents_path)
    doc_to_agency = _load_doc_to_agency_map(cleaned_documents_path)
    issues: list[ValidationIssue] = []
    file_set = known.get("파일명", set())
    biz_set = known.get("사업명", set())

    for sample in samples:
        sid = sample.question_id
        meta = sample.metadata or {}

        # 1+2. metadata_filter 키/값 검증
        mf = meta.get("metadata_filter")
        if isinstance(mf, dict):
            for key, value in mf.items():
                if known and key not in known:
                    issues.append(
                        ValidationIssue(
                            sid,
                            "warning",
                            "metadata_filter",
                            f"unknown key {key!r} (not in cleaned_documents columns)",
                        )
                    )
                    continue
                # operator (dict) 값은 매칭 skip
                if isinstance(value, (str, int, float)) and known.get(key):
                    if value not in known[key]:
                        issues.append(
                            ValidationIssue(
                                sid,
                                "warning",
                                "metadata_filter",
                                f"value {value!r} for {key!r} not found in data",
                            )
                        )

        # 3. ground_truth_docs 검증
        titles = sample.expected_doc_titles
        if titles and (file_set or biz_set):
            for title in titles:
                if title not in file_set and title not in biz_set:
                    issues.append(
                        ValidationIssue(
                            sid,
                            "warning",
                            "ground_truth_docs",
                            f"{title!r} not found in 파일명 or 사업명",
                        )
                    )

        # 4. history 형식 검증
        history = meta.get("history")
        if history is not None:
            if not isinstance(history, list):
                issues.append(
                    ValidationIssue(
                        sid, "error", "history", "must be a list of message dicts"
                    )
                )
            else:
                for i, turn in enumerate(history):
                    if not isinstance(turn, dict):
                        issues.append(
                            ValidationIssue(
                                sid,
                                "error",
                                "history",
                                f"turn[{i}] is not a dict",
                            )
                        )

        _warn_if_multidoc_question_is_underspecified(
            sample=sample,
            issues=issues,
            doc_to_agency=doc_to_agency,
        )

    return ValidationReport(total_samples=len(samples), issues=issues)


def render_validation_report(report: ValidationReport, max_lines: int = 20) -> str:
    """콘솔 친화적인 검증 리포트 텍스트 생성."""
    lines = [
        f"=== Eval set validation: {report.total_samples} samples ===",
        f"errors: {len(report.errors)}, warnings: {len(report.warnings)}",
    ]
    if not report.issues:
        lines.append("✓ All samples valid")
        return "\n".join(lines)
    shown = report.errors + report.warnings
    for issue in shown[:max_lines]:
        icon = "❌" if issue.severity == "error" else "⚠️"
        lines.append(f"{icon} [{issue.sample_id}] {issue.field}: {issue.message}")
    if len(shown) > max_lines:
        lines.append(f"... and {len(shown) - max_lines} more")
    return "\n".join(lines)
