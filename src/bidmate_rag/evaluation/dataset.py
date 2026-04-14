"""평가 데이터셋 로딩 헬퍼."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.loaders.metadata_loader import load_metadata_frame
from bidmate_rag.retrieval.filters import DOMAIN_KEYWORDS
from bidmate_rag.retrieval.agency_matching import (
    extract_agencies_from_text,
    normalize_agency_name,
)
from bidmate_rag.schema import EvalSample

logger = logging.getLogger(__name__)

_DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "metadata" / "data_list.csv"
)
_DEFAULT_PROJECT_ALIAS_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_quality"
    / "project_alias_map.csv"
)
_KNOWN_DOMAIN_VALUES = set(DOMAIN_KEYWORDS) | {"기타 정보시스템"}
_LOCAL_GOVERNMENT_AGENCY_PATTERN = re.compile(
    r"^(?P<region>[가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도))\s+(?P<local>[가-힣]+(?:시|군|구))$"
)
_PROJECT_COMPACT_KEY_PATTERN = re.compile(r"[^0-9a-z가-힣]+")
_PROJECT_LEADING_YEAR_PATTERN = re.compile(r"^(?:\d{4}\s*(?:년|년도)?\s*)+")


# 평가셋 CSV의 metadata_filter 컬럼은 영문 키를 사용하지만, ChromaDB에 저장된
# 청크 메타데이터는 한국어 키(공백 포함)를 사용합니다. 평가 시 retrieval에
# 적용하려면 영문 → 한국어 매핑이 필요합니다.
EVAL_FILTER_KEY_MAP: dict[str, str] = {
    "agency": "발주 기관",
    "institution": "발주 기관",
    "project": "사업명",
    "domain": "사업도메인",
    "agency_type": "기관유형",
    "tech_stack": "기술스택",
    "year": "공개연도",
    "budget": "사업 금액",
}

# `공개연도`는 ChromaDB에 int로 저장되므로 문자열을 변환해야 매칭됩니다.
_NUMERIC_FILTER_KEYS = {"공개연도", "사업 금액"}


def _extract_agencies_from_question(
    question: str,
    agency_list: list[str],
) -> list[str]:
    """질문 텍스트에서 발주 기관명을 추출한다.

    Args:
        question: 평가 질문 문자열.
        agency_list: 전체 발주기관 목록.

    Returns:
        매칭된 기관명 리스트.
    """
    return extract_agencies_from_text(question, agency_list)


def _normalize_filename_key(value: str) -> str:
    """Normalize a filename/stem so legacy eval doc references can be resolved."""
    stem = Path(value or "").stem
    normalized = unicodedata.normalize("NFKC", stem)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _filename_prefix(value: str) -> str:
    stem = Path(value or "").stem
    return stem.split("_", 1)[0].strip()


def _project_title_from_filename(value: str) -> str:
    stem = Path(value or "").stem.strip()
    if not stem:
        return ""
    return stem.split("_", 1)[1].strip() if "_" in stem else stem


def _normalize_project_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _strip_leading_project_year(value: str) -> str:
    return _PROJECT_LEADING_YEAR_PATTERN.sub("", value).strip()


def _compact_project_key(value: str) -> str:
    normalized = _strip_leading_project_year(_normalize_project_text(value))
    return _PROJECT_COMPACT_KEY_PATTERN.sub("", normalized)


def _project_lookup_keys(value: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    normalized = _normalize_project_text(value)
    stripped = _strip_leading_project_year(normalized)
    compact = _compact_project_key(value)

    for candidate in (normalized, stripped, compact):
        if candidate and candidate not in seen:
            keys.append(candidate)
            seen.add(candidate)
    return keys


def _project_tokens(value: str) -> list[str]:
    normalized = _strip_leading_project_year(_normalize_project_text(value))
    return [token for token in re.findall(r"[0-9a-z가-힣]+", normalized) if len(token) >= 2]


def _collapse_unique_aliases(
    alias_candidates: dict[str, set[str]],
) -> dict[str, str]:
    return {
        alias: next(iter(projects))
        for alias, projects in alias_candidates.items()
        if len(projects) == 1
    }


@lru_cache(maxsize=4)
def _load_expected_doc_title_map(
    metadata_path: str | None = None,
    duplicates_map_path: str | None = None,
) -> dict[str, str]:
    """Build a legacy eval doc-title resolver from metadata + duplicate policy."""
    path = Path(metadata_path) if metadata_path else _DEFAULT_METADATA_PATH
    if not path.exists():
        return {}

    frame = load_metadata_frame(path, duplicates_map_path=duplicates_map_path)
    if "파일명" not in frame.columns:
        return {}

    extension_priority = {"pdf": 0, "hwp": 1, "hwpx": 2, "docx": 3, "docs": 4}
    resolver: dict[str, str] = {}
    records: list[tuple[tuple[int, int, str], str, str]] = []

    for _, row in frame.iterrows():
        source_file = str(row.get("파일명", "") or "")
        ingest_file = str(row.get("ingest_file", "") or source_file)
        if not source_file or not ingest_file:
            continue

        priority = (
            0 if bool(row.get("ingest_enabled", True)) else 1,
            extension_priority.get(Path(ingest_file).suffix.lower().lstrip("."), 99),
            ingest_file,
        )
        records.append((priority, source_file, ingest_file))

    for _, source_file, ingest_file in sorted(records, key=lambda item: item[0]):
        for candidate in (source_file, ingest_file):
            key = _normalize_filename_key(candidate)
            if key:
                resolver.setdefault(key, ingest_file)

    return resolver


@lru_cache(maxsize=4)
def _load_agency_alias_map(
    metadata_path: str | None = None,
    duplicates_map_path: str | None = None,
) -> dict[str, str]:
    """Build a legacy agency-alias resolver from metadata/file-name prefixes."""
    path = Path(metadata_path) if metadata_path else _DEFAULT_METADATA_PATH
    if not path.exists():
        return {}

    frame = load_metadata_frame(path, duplicates_map_path=duplicates_map_path)
    if "resolved_agency" not in frame.columns:
        return {}

    alias_candidates: dict[str, set[str]] = {}

    for _, row in frame.iterrows():
        resolved_agency = str(row.get("resolved_agency", "") or "").strip()
        if not resolved_agency:
            continue

        candidates = {
            resolved_agency,
            str(row.get("original_agency", "") or "").strip(),
            _filename_prefix(str(row.get("파일명", "") or "")),
            _filename_prefix(str(row.get("canonical_file", "") or "")),
            _filename_prefix(str(row.get("ingest_file", "") or "")),
        }

        local_government_match = _LOCAL_GOVERNMENT_AGENCY_PATTERN.match(resolved_agency)
        if local_government_match:
            candidates.add(local_government_match.group("local"))

        for candidate in candidates:
            normalized = normalize_agency_name(candidate)
            if len(normalized) < 3:
                continue
            alias_candidates.setdefault(normalized, set()).add(resolved_agency)

    return {
        alias: next(iter(agencies))
        for alias, agencies in alias_candidates.items()
        if len(agencies) == 1
    }


def _lookup_agency_alias(value: str, agency_alias_map: dict[str, str] | None) -> list[str]:
    if not agency_alias_map:
        return []

    normalized_value = normalize_agency_name(value)
    if not normalized_value:
        return []

    exact_match = agency_alias_map.get(normalized_value)
    if exact_match:
        return [exact_match]

    partial_matches: list[str] = []
    seen: set[str] = set()
    for alias_key, agency in agency_alias_map.items():
        if len(alias_key) < 3:
            continue
        if alias_key in normalized_value or normalized_value in alias_key:
            if agency not in seen:
                partial_matches.append(agency)
                seen.add(agency)

    return partial_matches if len(partial_matches) == 1 else []


@lru_cache(maxsize=4)
def _load_project_alias_map(
    metadata_path: str | None = None,
    project_alias_path: str | None = None,
    duplicates_map_path: str | None = None,
) -> dict[str, Any]:
    """Build a project-title alias resolver from metadata + project alias policy."""
    metadata_file = Path(metadata_path) if metadata_path else _DEFAULT_METADATA_PATH
    alias_file = Path(project_alias_path) if project_alias_path else _DEFAULT_PROJECT_ALIAS_PATH

    alias_candidates_by_agency: dict[str, dict[str, set[str]]] = {}
    global_alias_candidates: dict[str, set[str]] = {}
    agency_projects: dict[str, set[str]] = {}

    def register_project_aliases(agency: str, canonical_project: str, aliases: set[str]) -> None:
        agency_name = str(agency or "").strip()
        project_name = str(canonical_project or "").strip()
        agency_key = normalize_agency_name(agency_name)
        if not agency_key or not project_name:
            return

        agency_projects.setdefault(agency_key, set()).add(project_name)
        for alias in aliases | {project_name}:
            for alias_key in _project_lookup_keys(alias):
                if len(alias_key) < 4:
                    continue
                alias_candidates_by_agency.setdefault(agency_key, {}).setdefault(
                    alias_key,
                    set(),
                ).add(project_name)
                global_alias_candidates.setdefault(alias_key, set()).add(project_name)

    if metadata_file.exists():
        frame = load_metadata_frame(metadata_file, duplicates_map_path=duplicates_map_path)
        if {"사업명", "발주 기관"}.issubset(frame.columns):
            for _, row in frame.iterrows():
                resolved_agency = str(
                    row.get("resolved_agency", "") or row.get("발주 기관", "") or ""
                ).strip()
                canonical_project = str(row.get("사업명", "") or "").strip()
                aliases = {
                    _project_title_from_filename(str(row.get("파일명", "") or "")),
                    _project_title_from_filename(str(row.get("canonical_file", "") or "")),
                    _project_title_from_filename(str(row.get("ingest_file", "") or "")),
                }
                register_project_aliases(resolved_agency, canonical_project, aliases)

    if alias_file.exists():
        alias_frame = pd.read_csv(alias_file, encoding="utf-8-sig").fillna("")
        required_columns = {"canonical_agency", "canonical_project", "project_alias"}
        if required_columns.issubset(alias_frame.columns):
            for _, row in alias_frame.iterrows():
                enabled = row.get("enabled", True)
                enabled_flag = enabled if isinstance(enabled, bool) else str(enabled).strip().lower()
                if enabled_flag in {"false", "0", "no", "n"}:
                    continue

                register_project_aliases(
                    str(row.get("canonical_agency", "") or "").strip(),
                    str(row.get("canonical_project", "") or "").strip(),
                    {
                        str(row.get("project_alias", "") or "").strip(),
                    },
                )

    return {
        "agency_aliases": {
            agency_key: _collapse_unique_aliases(alias_candidates)
            for agency_key, alias_candidates in alias_candidates_by_agency.items()
        },
        "global_aliases": _collapse_unique_aliases(global_alias_candidates),
        "agency_projects": {
            agency_key: {
                project: _compact_project_key(project)
                for project in sorted(projects)
                if _compact_project_key(project)
            }
            for agency_key, projects in agency_projects.items()
        },
    }


def _split_multi_value_text(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _resolve_agency_values(
    raw_value: str,
    agency_list: list[str] | None = None,
    agency_alias_map: dict[str, str] | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for piece in _split_multi_value_text(raw_value):
        matched = extract_agencies_from_text(piece, agency_list or [])
        if not matched:
            matched = _lookup_agency_alias(piece, agency_alias_map)
        for agency in matched:
            if agency not in seen:
                resolved.append(agency)
                seen.add(agency)

    return resolved


def _resolve_single_project_value(
    raw_value: str,
    scoped_agencies: list[str] | None = None,
    project_alias_map: dict[str, Any] | None = None,
) -> str | None:
    if not project_alias_map:
        return None

    value = str(raw_value or "").strip()
    if not value:
        return None

    agency_aliases = project_alias_map.get("agency_aliases", {})
    global_aliases = project_alias_map.get("global_aliases", {})
    agency_projects = project_alias_map.get("agency_projects", {})

    lookup_keys = _project_lookup_keys(value)
    scoped_agency_keys = [
        agency_key
        for agency in (scoped_agencies or [])
        if (agency_key := normalize_agency_name(agency))
    ]

    exact_matches: list[str] = []
    seen_exact: set[str] = set()
    if scoped_agency_keys:
        for agency_key in scoped_agency_keys:
            scoped_aliases = agency_aliases.get(agency_key, {})
            for lookup_key in lookup_keys:
                matched_project = scoped_aliases.get(lookup_key)
                if matched_project and matched_project not in seen_exact:
                    exact_matches.append(matched_project)
                    seen_exact.add(matched_project)
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            logger.warning(
                "Ambiguous scoped project alias value=%r agencies=%s matched=%s",
                value,
                scoped_agencies,
                exact_matches,
            )
            return None
    else:
        for lookup_key in lookup_keys:
            matched_project = global_aliases.get(lookup_key)
            if matched_project and matched_project not in seen_exact:
                exact_matches.append(matched_project)
                seen_exact.add(matched_project)
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            logger.warning(
                "Ambiguous global project alias value=%r matched=%s",
                value,
                exact_matches,
            )
            return None

    compact_value = _compact_project_key(value)
    if not compact_value:
        return None

    fuzzy_matches: list[tuple[int, int, str]] = []
    seen_fuzzy: set[str] = set()
    query_tokens = _project_tokens(value)
    if scoped_agency_keys:
        candidate_sets = [agency_projects.get(agency_key, {}) for agency_key in scoped_agency_keys]
    else:
        candidate_sets = list(agency_projects.values())

    for project_map in candidate_sets:
        for project_name, project_key in project_map.items():
            normalized_project = _strip_leading_project_year(_normalize_project_text(project_name))
            token_match = query_tokens and all(
                token in normalized_project or token in project_key for token in query_tokens
            )
            substring_match = compact_value in project_key or project_key in compact_value
            if not substring_match and not token_match:
                continue
            if project_name not in seen_fuzzy:
                fuzzy_matches.append((abs(len(project_key) - len(compact_value)), len(project_key), project_name))
                seen_fuzzy.add(project_name)

    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0][2]
    if len(fuzzy_matches) > 1:
        fuzzy_matches.sort()
        best_match = fuzzy_matches[0]
        runner_up = fuzzy_matches[1]
        if best_match[:2] != runner_up[:2]:
            return best_match[2]
        logger.warning(
            "Ambiguous fuzzy project match value=%r agencies=%s matched=%s",
            value,
            scoped_agencies,
            [project_name for _, _, project_name in fuzzy_matches],
        )
    return None


def _resolve_project_values(
    raw_value: str,
    scoped_agencies: list[str] | None = None,
    project_alias_map: dict[str, Any] | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for piece in _split_multi_value_text(raw_value):
        matched_project = _resolve_single_project_value(
            piece,
            scoped_agencies=scoped_agencies,
            project_alias_map=project_alias_map,
        )
        if matched_project and matched_project not in seen:
            resolved.append(matched_project)
            seen.add(matched_project)

    return resolved


def _resolve_expected_doc_titles(
    docs: list[str],
    metadata_path: str | Path | None = None,
    duplicates_map_path: str | Path | None = None,
) -> list[str]:
    """Resolve legacy `.json` doc refs to the actual ingested source filenames."""
    resolver = _load_expected_doc_title_map(
        str(metadata_path) if metadata_path else None,
        str(duplicates_map_path) if duplicates_map_path else None,
    )
    resolved: list[str] = []
    seen: set[str] = set()

    for doc in docs:
        text = str(doc or "").strip()
        if not text:
            continue

        resolved_doc = resolver.get(_normalize_filename_key(text))
        if not resolved_doc and text.lower().endswith(".json"):
            resolved_doc = Path(text).stem
        if not resolved_doc:
            resolved_doc = text

        if resolved_doc not in seen:
            resolved.append(resolved_doc)
            seen.add(resolved_doc)

    return resolved


def normalize_metadata_filter(
    raw: dict[str, Any] | None,
    question: str = "",
    agency_list: list[str] | None = None,
    agency_alias_map: dict[str, str] | None = None,
    project_alias_map: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """평가셋 metadata_filter의 영문 키를 ChromaDB 한국어 키로 정규화한다.

    Args:
        raw: 평가셋의 metadata_filter 딕셔너리 (영문 키).
        question: 평가 질문 문자열 ("다중" 처리 시 기관명 추출에 사용).
        agency_list: 전체 발주기관 목록 ("다중" 처리 시 매칭에 사용).

    Returns:
        ChromaDB where 절에 사용할 정규화된 필터 딕셔너리, 또는 None.
    """
    if not raw or not isinstance(raw, dict):
        return None

    resolved_agency_scope: list[str] = []
    seen_agencies: set[str] = set()

    def add_resolved_agencies(values: list[str]) -> None:
        for agency in values:
            if agency and agency not in seen_agencies:
                resolved_agency_scope.append(agency)
                seen_agencies.add(agency)

    for key, value in raw.items():
        target_key = EVAL_FILTER_KEY_MAP.get(key, key)
        if not isinstance(value, str):
            continue
        if value == "다중" and target_key == EVAL_FILTER_KEY_MAP["agency"] and agency_list:
            add_resolved_agencies(_extract_agencies_from_question(question, agency_list))
            continue
        if target_key in {EVAL_FILTER_KEY_MAP["agency"], EVAL_FILTER_KEY_MAP["domain"]}:
            add_resolved_agencies(
                _resolve_agency_values(
                    value,
                    agency_list=agency_list,
                    agency_alias_map=agency_alias_map,
                )
            )

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        # 영문 키 → 한국어 키 변환 (예: "agency" → "발주 기관")
        target_key = EVAL_FILTER_KEY_MAP.get(key, key)
        if target_key not in EVAL_FILTER_KEY_MAP.values() and target_key == key:
            logger.warning(
                "Unknown metadata_filter key %r in eval sample — passing through as-is",
                key,
            )
        # "다중" → 질문에서 기관명 추출 후 $in 필터로 변환
        # 예: {"agency": "다중"} → {"발주 기관": {"$in": ["한국가스공사", "고려대학교"]}}
        if target_key == EVAL_FILTER_KEY_MAP["domain"] and isinstance(value, str):
            matched_agencies = _resolve_agency_values(
                value,
                agency_list=agency_list,
                agency_alias_map=agency_alias_map,
            )
            if matched_agencies:
                logger.warning(
                    "Reinterpreting legacy eval filter domain=%r as agency filter=%s",
                    value,
                    matched_agencies,
                )
                if len(matched_agencies) == 1:
                    normalized[EVAL_FILTER_KEY_MAP["agency"]] = matched_agencies[0]
                else:
                    normalized[EVAL_FILTER_KEY_MAP["agency"]] = {"$in": matched_agencies}
                continue
            if value not in _KNOWN_DOMAIN_VALUES:
                logger.warning(
                    "Dropping unrecognized legacy domain filter value=%r because it is neither a known domain nor a matched agency",
                    value,
                )
                continue
        if value == "다중" and target_key == "발주 기관" and agency_list:
            agencies = _extract_agencies_from_question(question, agency_list)
            if len(agencies) < 2:
                logger.warning(
                    "Multi-agency metadata_filter matched fewer than 2 agencies: matched=%s question=%r",
                    agencies,
                    question,
                )
            if agencies:
                normalized[target_key] = {"$in": agencies}
            # 기관을 못 찾으면 필터 없이 전체 검색
            continue
        if target_key == EVAL_FILTER_KEY_MAP["agency"] and isinstance(value, str):
            if agency_list or agency_alias_map:
                matched_agencies = _resolve_agency_values(
                    value,
                    agency_list=agency_list,
                    agency_alias_map=agency_alias_map,
                )
                if matched_agencies:
                    if len(matched_agencies) == 1:
                        normalized[target_key] = matched_agencies[0]
                    else:
                        normalized[target_key] = {"$in": matched_agencies}
                    continue
                logger.warning(
                    "Dropping unresolved agency-like filter value=%r because it did not map to a stored agency",
                    value,
                )
                continue
        if target_key == EVAL_FILTER_KEY_MAP["project"] and isinstance(value, str):
            if project_alias_map:
                matched_projects = _resolve_project_values(
                    value,
                    scoped_agencies=resolved_agency_scope,
                    project_alias_map=project_alias_map,
                )
                if matched_projects:
                    if len(matched_projects) == 1:
                        normalized[target_key] = matched_projects[0]
                    else:
                        normalized[target_key] = {"$in": matched_projects}
                    continue
                logger.warning(
                    "Dropping unresolved project-like filter value=%r agencies=%s because it did not map to a stored project title",
                    value,
                    resolved_agency_scope,
                )
                continue
        # 숫자형 필드는 ChromaDB가 int를 기대하므로 변환
        if target_key in _NUMERIC_FILTER_KEYS and isinstance(value, str):
            pieces = [part.strip() for part in value.split(",") if part.strip()]
            if len(pieces) > 1:
                numeric_values: list[int] = []
                for piece in pieces:
                    try:
                        numeric_values.append(int(piece))
                    except (TypeError, ValueError):
                        numeric_values = []
                        break
                if numeric_values:
                    normalized[target_key] = {"$in": numeric_values}
                    continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        normalized[target_key] = value
    return normalized or None


# 평가셋 버전 디렉터리는 data/eval/ 아래에 eval_v1, eval_v2 등으로 존재.
# CLI / Streamlit은 가장 높은 버전 번호를 자동으로 사용한다.
EVAL_ROOT = Path("data/eval")
_EVAL_VERSION_PATTERN = re.compile(r"^eval_v(\d+)$")


def find_latest_eval_dir(root: Path | str = EVAL_ROOT) -> Path:
    """가장 최신 eval_v* 디렉터리 경로를 반환한다.

    Args:
        root: 평가셋 루트 디렉터리.

    Returns:
        가장 높은 버전 번호의 eval_v* 디렉터리 경로.
    """
    root_path = Path(root)
    versions: list[tuple[int, Path]] = []
    if root_path.exists():
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            # eval_v1, eval_v2 등에서 버전 번호 추출
            match = _EVAL_VERSION_PATTERN.match(child.name)
            if match:
                versions.append((int(match.group(1)), child))
    # 가장 높은 버전 반환, 없으면 루트 자체 반환 (레거시 호환)
    if versions:
        return max(versions, key=lambda item: item[0])[1]
    return root_path


def list_eval_csvs(root: Path | str = EVAL_ROOT) -> list[Path]:
    """최신 eval 디렉터리 내의 eval_batch_*.csv 파일 목록을 반환한다.

    Args:
        root: 평가셋 루트 디렉터리.

    Returns:
        정렬된 CSV 파일 경로 리스트.
    """
    return sorted(find_latest_eval_dir(root).glob("eval_batch_*.csv"))


PROCESSED_ROOT = Path("data/processed")


def find_latest_metadata_path(
    root: Path | str = PROCESSED_ROOT,
) -> Path:
    """가장 최신 cleaned_documents.parquet 경로를 반환한다.

    Args:
        root: 처리된 데이터 루트 디렉터리.

    Returns:
        가장 최신 cleaned_documents.parquet 파일 경로.
    """
    root_path = Path(root)
    if not root_path.exists():
        return root_path / "cleaned_documents.parquet"

    # 실험별 sub-dir에서 mtime이 가장 최근인 parquet 탐색
    candidates: list[tuple[float, Path]] = []
    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / "cleaned_documents.parquet"
        if candidate.exists():
            candidates.append((candidate.stat().st_mtime, candidate))

    # 실험별 파일이 있으면 최신 것 반환, 없으면 공용 경로 반환 (레거시 폴백)
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return root_path / "cleaned_documents.parquet"


# EvalSample의 top-level 필드가 아니지만 metadata에 보존해야 하는 CSV 컬럼들.
# 다운스트림 필터(예: --filter-type)에서 사용된다.
_METADATA_PASSTHROUGH_COLUMNS = (
    "type",
    "difficulty",
    "ground_truth_answer",
    "metadata_filter",
    "history",
)


def _coerce_json_field(value: Any) -> Any:
    """CSV 셀 값을 JSON으로 파싱한다. 이미 파싱된 값은 그대로 반환.

    Args:
        value: CSV 셀 값 (문자열, dict, list, None 등).

    Returns:
        파싱된 Python 객체 또는 원본 값.
    """
    if value is None:
        return None
    # 이미 파싱된 dict/list는 그대로 반환
    if isinstance(value, (dict, list)):
        return value
    # NaN 체크
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    # 빈 문자열이나 빈 JSON 구조는 None 반환
    if not text or text in ("[]", "{}"):
        return None
    # JSON 파싱 시도, 실패 시 원본 문자열 반환
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _normalize_row(
    row: dict[str, Any],
    agency_list: list[str] | None = None,
    metadata_path: str | Path | None = None,
    project_alias_path: str | Path | None = None,
    duplicates_map_path: str | Path | None = None,
) -> dict[str, Any]:
    """평가셋 CSV 행을 EvalSample 스키마로 변환한다.

    Args:
        row: 평가셋 CSV의 한 행 (dict).
        agency_list: 전체 발주기관 목록 ("다중" 필터 변환에 사용).

    Returns:
        EvalSample.model_validate()에 전달할 정규화된 딕셔너리.
    """
    # 이미 정규화된 형식이면 그대로 반환
    if "question_id" in row and "question" in row:
        return row

    normalized: dict[str, Any] = {}
    # CSV의 "id" 컬럼 → EvalSample의 "question_id"로 매핑
    if "id" in row:
        normalized["question_id"] = str(row["id"])
    normalized["question"] = row.get("question", "")

    # ground_truth_docs: JSON 문자열 → 파일명 리스트로 변환
    docs = _coerce_json_field(row.get("ground_truth_docs"))
    if isinstance(docs, str):
        docs = [docs]
    if isinstance(docs, list):
        normalized["expected_doc_titles"] = _resolve_expected_doc_titles(
            [str(d) for d in docs],
            metadata_path=metadata_path,
            duplicates_map_path=duplicates_map_path,
        )

    # 메타데이터 컬럼들을 파싱하여 metadata 딕셔너리로 수집
    metadata: dict[str, Any] = {}
    agency_alias_map = _load_agency_alias_map(
        str(metadata_path) if metadata_path else None,
        str(duplicates_map_path) if duplicates_map_path else None,
    )
    project_alias_map = _load_project_alias_map(
        str(metadata_path) if metadata_path else None,
        str(project_alias_path) if project_alias_path else None,
        str(duplicates_map_path) if duplicates_map_path else None,
    )
    for col in _METADATA_PASSTHROUGH_COLUMNS:
        if col not in row:
            continue
        raw = row[col]
        if col == "metadata_filter":
            # metadata_filter: JSON 파싱 → 영문 키 정규화 → "다중" 처리
            parsed = _coerce_json_field(raw)
            normalized_filter = normalize_metadata_filter(
                parsed if isinstance(parsed, dict) else None,
                question=normalized.get("question", ""),
                agency_list=agency_list,
                agency_alias_map=agency_alias_map,
                project_alias_map=project_alias_map,
            )
            if normalized_filter:
                metadata["metadata_filter"] = normalized_filter
        elif col == "history":
            # history: JSON 파싱 → 비어있지 않은 리스트만 저장
            parsed = _coerce_json_field(raw)
            if isinstance(parsed, list) and parsed:
                metadata["history"] = parsed
        else:
            # 나머지 컬럼: NaN/빈 값 제외 후 그대로 저장
            try:
                if pd.isna(raw):
                    continue
            except (TypeError, ValueError):
                pass
            if raw is None or raw == "":
                continue
            metadata[col] = raw
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def load_eval_samples(
    path: str | Path,
    agency_list: list[str] | None = None,
    metadata_path: str | Path | None = None,
    project_alias_path: str | Path | None = None,
    duplicates_map_path: str | Path | None = None,
) -> list[EvalSample]:
    """평가셋 파일을 로딩하여 EvalSample 리스트로 반환한다.

    Args:
        path: 평가셋 파일 경로 (CSV, JSON, JSONL, Parquet 지원).
        agency_list: 전체 발주기관 목록 ("다중" 필터 변환에 사용).

    Returns:
        EvalSample 객체 리스트.
    """
    source = Path(path)
    # 파일 형식에 따라 행(row) 리스트로 로딩
    if source.suffix == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if source.suffix == ".csv":
            frame = pd.read_csv(source, encoding="utf-8-sig")
        else:
            frame = pd.read_parquet(source)
        rows = frame.to_dict(orient="records")
    # 각 행을 정규화 후 EvalSample 객체로 변환
    return [
        EvalSample.model_validate(
            _normalize_row(
                row,
                agency_list=agency_list,
                metadata_path=metadata_path,
                project_alias_path=project_alias_path,
                duplicates_map_path=duplicates_map_path,
            )
        )
        for row in rows
    ]
