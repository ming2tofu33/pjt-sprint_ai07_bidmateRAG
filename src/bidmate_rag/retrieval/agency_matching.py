"""Shared agency-name normalization and matching helpers."""

from __future__ import annotations

import re

_AGENCY_LEGAL_PREFIX_PATTERN = re.compile(
    r"(?:[\(（]\s*[사재주]\s*[\)）]|㈜|사단법인|재단법인|주식회사)"
)
_AGENCY_NOISE_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]+")
_LEADING_YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}\s+")
_TRAILING_DESCRIPTOR_PATTERN = re.compile(r"\s*[\(（][^()（）]{1,20}[\)）]\s*$")
_TRAILING_PORTAL_SUFFIX_PATTERN = re.compile(r"\s+(?:입찰공고|전자조달)$")
_KNOWN_ACRONYM_ALIASES = {
    "KOICA": "코이카",
}


def normalize_agency_name(value: str) -> str:
    """기관명 비교를 위해 법인 표기/특수문자를 제거한 문자열을 만든다."""
    cleaned = _AGENCY_LEGAL_PREFIX_PATTERN.sub("", value or "")
    return _AGENCY_NOISE_PATTERN.sub("", cleaned).strip()


def _expand_known_aliases(value: str) -> list[str]:
    candidates = [value]
    for acronym, alias in _KNOWN_ACRONYM_ALIASES.items():
        replaced = re.sub(re.escape(acronym), alias, value, flags=re.IGNORECASE)
        if replaced != value:
            candidates.extend([replaced, alias])
    return candidates


def _derive_structural_aliases(value: str) -> list[str]:
    queue = [re.sub(r"\s+", " ", value or "").strip()]
    aliases: list[str] = []
    seen: set[str] = set()

    while queue:
        current = queue.pop(0)
        if len(current) < 3 or current in seen:
            continue

        aliases.append(current)
        seen.add(current)

        year_stripped = _LEADING_YEAR_PATTERN.sub("", current).strip()
        if year_stripped and year_stripped != current:
            queue.append(year_stripped)

        descriptor_stripped = _TRAILING_DESCRIPTOR_PATTERN.sub("", current).strip()
        if descriptor_stripped and descriptor_stripped != current:
            queue.append(descriptor_stripped)

        portal_stripped = _TRAILING_PORTAL_SUFFIX_PATTERN.sub("", current).strip()
        if portal_stripped and portal_stripped != current:
            queue.append(portal_stripped)

    return aliases


def build_agency_aliases(agency: str) -> list[str]:
    """원문 기관명에서 비교 가능한 별칭 후보를 만든다."""
    aliases: list[str] = []
    seen: set[str] = set()

    for structural_alias in _derive_structural_aliases(agency.strip()):
        for candidate in _expand_known_aliases(structural_alias):
            normalized = normalize_agency_name(candidate)
            for item in (candidate, normalized):
                if len(item) < 3 or item in seen:
                    continue
                aliases.append(item)
                seen.add(item)

    return aliases


def extract_agencies_from_text(text: str, agency_list: list[str]) -> list[str]:
    """질문/텍스트에 명시적으로 등장하는 기관명을 추출한다."""
    matched: list[str] = []
    normalized_text = normalize_agency_name(text)
    for agency in agency_list:
        for alias in build_agency_aliases(agency):
            normalized_alias = normalize_agency_name(alias)
            if normalized_alias and normalized_alias in normalized_text:
                if agency not in matched:
                    matched.append(agency)
                break
    return matched
