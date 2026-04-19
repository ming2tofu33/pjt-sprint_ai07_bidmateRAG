"""Conversation memory helpers for multiturn RAG."""

from __future__ import annotations

import re

from bidmate_rag.retrieval.filters import extract_matched_agencies

_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAILING_REQUEST_PATTERN = re.compile(
    r"\s*(알려줘|정리해줘|설명해줘|보여줘|찾아줘|비교해줘|말해줘|요약해줘|뭐야)$"
)
_BUDGET_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(원|천만원|백만원|만원|억원)"
)
_PROJECT_HINT_PATTERN = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-\·()]{4,}")
_INTEREST_KEYWORDS = (
    ("평가기준", "평가기준"),
    ("기술평가", "평가기준"),
    ("배점", "평가기준"),
    ("예산", "예산"),
    ("금액", "예산"),
    ("일정", "일정"),
    ("기간", "일정"),
    ("마감", "일정"),
    ("종료", "일정"),
    ("유지보수", "유지보수"),
    ("보안", "보안"),
    ("하도급", "하도급"),
    ("클라우드", "클라우드"),
)
REWRITE_SAFE_SLOT_KEYS = ("발주기관", "사업명", "관심속성")


def _normalize_text(text: str) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip().strip("\"'")
    normalized = re.sub(r"[?!.]+$", "", normalized).strip()
    while True:
        trimmed = _TRAILING_REQUEST_PATTERN.sub("", normalized).strip()
        if trimmed == normalized:
            break
        normalized = trimmed
    return normalized


def _coerce_messages(chat_history: list[dict] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in chat_history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
            continue
        if isinstance(item.get("user"), str) and item["user"].strip():
            messages.append({"role": "user", "content": item["user"].strip()})
        if isinstance(item.get("assistant"), str) and item["assistant"].strip():
            messages.append({"role": "assistant", "content": item["assistant"].strip()})
    return messages


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3].rstrip()}..."


def _extract_latest_budget(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        matches = _BUDGET_PATTERN.findall(message["content"])
        if matches:
            amount, unit = matches[-1]
            return f"{amount}{unit}"
    return None


def _extract_latest_keyword_snippet(
    messages: list[dict[str, str]],
    keywords: tuple[str, ...],
) -> str | None:
    for message in reversed(messages):
        content = message["content"]
        if any(keyword in content for keyword in keywords):
            return _truncate_text(_normalize_text(content), 120)
    return None


def _extract_recent_topic(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        candidate = _normalize_text(message["content"])
        if len(candidate) < 5:
            continue
        match = _PROJECT_HINT_PATTERN.search(candidate)
        if match:
            return match.group(0).strip()
        return candidate
    return None


def _extract_topic_from_rewritten_query(rewritten_query: str | None) -> str | None:
    normalized = _normalize_text(rewritten_query or "")
    if not normalized:
        return None
    if "의 " in normalized:
        candidate = normalized.split("의 ", 1)[0].strip()
        if len(candidate) >= 5:
            return candidate
    return None


def _extract_interest(rewritten_query: str | None, current_question: str | None) -> str | None:
    combined = " ".join(filter(None, [rewritten_query, current_question]))
    for keyword, label in _INTEREST_KEYWORDS:
        if keyword in combined:
            return label
    return None


def build_rewrite_safe_slot_memory(slot_memory: dict[str, str] | None) -> dict[str, str]:
    """재작성 프롬프트에 안전하게 전달할 슬롯만 남긴다."""
    if not slot_memory:
        return {}
    return {
        key: value.strip()
        for key, value in slot_memory.items()
        if key in REWRITE_SAFE_SLOT_KEYS and isinstance(value, str) and value.strip()
    }


class ConversationMemory:
    """Lightweight summary buffer + slot memory builder."""

    def __init__(
        self,
        *,
        max_recent_turns: int = 4,
        max_summary_chars: int = 400,
        agency_list: list[str] | None = None,
        slot_enabled: bool = True,
    ) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_summary_chars = max_summary_chars
        self.agency_list = agency_list or []
        self.slot_enabled = slot_enabled

    def build(
        self,
        chat_history: list[dict] | None,
        *,
        current_question: str | None = None,
        rewritten_query: str | None = None,
    ) -> dict[str, object]:
        messages = _coerce_messages(chat_history)
        if not messages:
            return {
                "recent_turns": [],
                "summary_buffer": "",
                "slot_memory": {},
            }

        recent_turns = messages[-self.max_recent_turns :]
        older_turns = messages[: -self.max_recent_turns] if len(messages) > self.max_recent_turns else []

        summary_parts = [
            f"{message['role']}: {_normalize_text(message['content'])}" for message in older_turns
        ]
        summary_buffer = _truncate_text(" | ".join(summary_parts), self.max_summary_chars)

        slot_memory: dict[str, str] = {}
        if self.slot_enabled:
            agencies = extract_matched_agencies(
                " ".join(message["content"] for message in messages),
                self.agency_list,
            )
            if agencies:
                slot_memory["발주기관"] = agencies[-1]

            recent_topic = _extract_topic_from_rewritten_query(rewritten_query) or _extract_recent_topic(messages)
            if recent_topic:
                slot_memory["사업명"] = recent_topic

            budget = _extract_latest_budget(messages)
            if budget:
                slot_memory["예산"] = budget

            schedule = _extract_latest_keyword_snippet(messages, ("일정", "기간", "마감", "종료"))
            if schedule:
                slot_memory["일정"] = schedule

            evaluation = _extract_latest_keyword_snippet(
                messages,
                ("평가기준", "기술평가", "배점", "평가 항목"),
            )
            if evaluation:
                slot_memory["평가기준"] = evaluation

            interest = _extract_interest(rewritten_query, current_question)
            if interest:
                slot_memory["관심속성"] = interest

        return {
            "recent_turns": recent_turns,
            "summary_buffer": summary_buffer,
            "slot_memory": slot_memory,
        }
