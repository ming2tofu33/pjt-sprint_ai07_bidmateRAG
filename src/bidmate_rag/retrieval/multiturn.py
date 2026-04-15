"""Helpers for history-aware multiturn retrieval."""

from __future__ import annotations

import re

from bidmate_rag.retrieval.filters import extract_matched_agencies

FOLLOW_UP_TOPIC_KEYWORDS = [
    "그 사업",
    "이 사업",
    "해당 사업",
    "방금 사업",
    "그 문서",
    "이 문서",
    "해당 문서",
    "방금 문서",
    "거기",
    "그거",
]
FOLLOW_UP_AGENCY_KEYWORDS = [
    "그 기관",
    "이 기관",
    "해당 기관",
]
_TRAILING_REQUEST_PATTERN = re.compile(
    r"\s*(알려줘|정리해줘|설명해줘|보여줘|찾아줘|비교해줘|말해줘|요약해줘|뭐야)$"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _iter_history_texts(chat_history: list[dict] | None) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for message in reversed(chat_history or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            texts.append((role, content.strip()))
            continue
        for legacy_role in ("user", "assistant"):
            legacy_content = message.get(legacy_role)
            if isinstance(legacy_content, str) and legacy_content.strip():
                texts.append((legacy_role, legacy_content.strip()))
    return texts


def _normalize_topic_candidate(text: str) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip().strip("\"'")
    normalized = re.sub(r"[?!.]+$", "", normalized).strip()
    while True:
        trimmed = _TRAILING_REQUEST_PATTERN.sub("", normalized).strip()
        if trimmed == normalized:
            break
        normalized = trimmed
    return normalized


def _extract_recent_topic_from_history(chat_history: list[dict] | None) -> str | None:
    prioritized: list[str] = []
    fallback: list[str] = []
    for role, text in _iter_history_texts(chat_history):
        candidate = _normalize_topic_candidate(text)
        if len(candidate) < 3:
            continue
        if role == "user":
            prioritized.append(candidate)
        else:
            fallback.append(candidate)
    for candidate in prioritized + fallback:
        return candidate
    return None


def extract_recent_agency_filter(
    chat_history: list[dict] | None,
    agency_list: list[str],
) -> dict[str, str] | None:
    """Return the latest single-agency filter mentioned in chat history."""

    for _, text in _iter_history_texts(chat_history):
        matched = extract_matched_agencies(text, agency_list)
        if len(matched) == 1:
            return {"발주 기관": matched[0]}
    return None


def rewrite_query_with_history(
    query: str,
    chat_history: list[dict] | None,
    agency_list: list[str],
) -> str:
    """Rewrite underspecified follow-up questions using recent chat history."""

    if not chat_history:
        return query
    if not any(
        keyword in query for keyword in FOLLOW_UP_TOPIC_KEYWORDS + FOLLOW_UP_AGENCY_KEYWORDS
    ):
        return query

    rewritten = query
    recent_topic = _extract_recent_topic_from_history(chat_history)
    recent_agency_filter = extract_recent_agency_filter(chat_history, agency_list)
    recent_agency = recent_agency_filter["발주 기관"] if recent_agency_filter else ""

    if recent_agency:
        for keyword in FOLLOW_UP_AGENCY_KEYWORDS:
            rewritten = rewritten.replace(keyword, recent_agency)

    if recent_topic:
        for keyword in FOLLOW_UP_TOPIC_KEYWORDS:
            rewritten = rewritten.replace(keyword, recent_topic)

    rewritten = _WHITESPACE_PATTERN.sub(" ", rewritten).strip()
    if rewritten == query and recent_topic:
        rewritten = f"{recent_topic} {query}".strip()
    return rewritten
