"""Helpers for history-aware multiturn retrieval."""

from __future__ import annotations

import logging
import re

from bidmate_rag.retrieval.filters import extract_matched_agencies
from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing

logger = logging.getLogger(__name__)
_PRICING = load_pricing()

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
_REWRITE_PROMPT_TEMPLATE = """당신은 공공입찰 RAG 검색용 쿼리 재작성 전문가입니다.
대화 이력과 현재 후속 질문, 그리고 구조화된 슬롯 메모리를 보고
검색에 가장 적합한 독립 질문으로 다시 써 주세요.

규칙:
- 반드시 재작성된 질문 한 줄만 출력하세요.
- 설명, 해설, 따옴표, 머리말을 붙이지 마세요.
- 발주기관, 사업명, 비교 대상, 관심 속성 등 이전 문맥의 핵심 조건을 최대한 보존하세요.
- 현재 질문이 이미 독립적이면 그대로 유지하세요.
- 슬롯 메모리에 있는 값은 문맥 복원용 힌트로만 사용하고, 없는 사실을 새로 만들지 마세요.

슬롯 메모리:
{slot_memory}

최근 대화 이력:
{history}

현재 질문: {question}

재작성 질문:"""


_SAFE_REWRITE_PROMPT_TEMPLATE = """당신은 공공입찰 RAG 검색용 후속 질문 재작성 도우미입니다.

반드시 재작성된 질문 한 문장만 출력하고, 그 외 설명은 출력하지 마세요.

규칙:
- 현재 질문의 뜻을 유지하세요.
- 기관명, 사업명, 문서명, 섹션명, 항목명처럼 생략된 대상만 보완하세요.
- 현재 질문에 없는 새로운 사실은 추가하지 마세요.
- 이전 assistant 답변에만 있던 자세한 사실은 가져오지 마세요.
- 현재 질문에 없는 괄호 설명, 예시, 수치 기준, 날짜, 인용은 덧붙이지 마세요.
- 현재 질문이 이미 충분히 분명하면 그대로 반환하세요.

메모리 슬롯:
{slot_memory}

최근 참고 대화:
{history}

현재 질문: {question}

재작성 질문:"""


def _iter_history_texts(
    chat_history: list[dict] | None,
    *,
    roles: tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    allowed_roles = set(roles) if roles else None
    for message in reversed(chat_history or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            if allowed_roles is None or role in allowed_roles:
                texts.append((role, content.strip()))
            continue
        for legacy_role in ("user", "assistant"):
            if allowed_roles is not None and legacy_role not in allowed_roles:
                continue
            legacy_content = message.get(legacy_role)
            if isinstance(legacy_content, str) and legacy_content.strip():
                texts.append((legacy_role, legacy_content.strip()))
    return texts


def _build_rewrite_history_lines(chat_history: list[dict] | None) -> list[str]:
    # Prefer recent user turns so the rewrite model resolves omitted targets
    # without copying factual details from prior assistant answers into the query.
    recent_user_turns = _iter_history_texts(chat_history, roles=("user",))[:4]
    reference_turns = recent_user_turns or _iter_history_texts(chat_history)[:4]
    return [f"{role}: {text}" for role, text in reversed(reference_turns)]


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


def _build_rewrite_trace(
    *,
    original_query: str,
    rewritten_query: str,
    rewrite_reason: str,
    model_name: str = "gpt-5-mini",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    rewrite_error: str | None = None,
) -> dict[str, object]:
    cost_usd = calc_llm_cost(
        model_name,
        prompt_tokens,
        completion_tokens,
        _PRICING,
    )
    trace = {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "rewrite_applied": rewritten_query != original_query,
        "rewrite_reason": rewrite_reason,
        "rewrite_prompt_tokens": prompt_tokens,
        "rewrite_completion_tokens": completion_tokens,
        "rewrite_total_tokens": total_tokens,
        "rewrite_cost_usd": cost_usd,
    }
    if rewrite_error:
        trace["rewrite_error"] = rewrite_error
    return trace


def _format_slot_memory(slot_memory: dict[str, str] | None) -> str:
    if not slot_memory:
        return "(없음)"
    lines = [f"- {key}: {value}" for key, value in slot_memory.items() if value]
    return "\n".join(lines) if lines else "(없음)"


def _llm_rewrite(
    query: str,
    chat_history: list[dict] | None,
    llm: object,
    *,
    slot_memory: dict[str, str] | None = None,
    max_completion_tokens: int = 16000,
    timeout_seconds: int = 30,
) -> tuple[str, dict[str, object]]:
    """LLM을 사용해 후속 질문을 독립적인 검색 쿼리로 재작성한다."""

    history_lines = _build_rewrite_history_lines(chat_history)

    prompt = _SAFE_REWRITE_PROMPT_TEMPLATE.format(
        history="\n".join(history_lines) or "(없음)",
        slot_memory=_format_slot_memory(slot_memory),
        question=query,
    )

    try:
        response = llm.rewrite(
            prompt,
            max_tokens=max_completion_tokens,
            timeout=timeout_seconds,
        )
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
        total_tokens = response.total_tokens
        rewritten = _WHITESPACE_PATTERN.sub(" ", response.text).strip()
        if rewritten and rewritten != query:
            logger.info("쿼리 재작성: '%s' -> '%s'", query, rewritten)
            return rewritten, _build_rewrite_trace(
                original_query=query,
                rewritten_query=rewritten,
                rewrite_reason="llm",
                model_name=getattr(llm, "model_name", "gpt-5-mini"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
    except Exception as exc:
        logger.warning("LLM 쿼리 재작성 실패, 규칙 기반으로 폴백합니다: %s", exc)
        return query, _build_rewrite_trace(
            original_query=query,
            rewritten_query=query,
            rewrite_reason="original",
            model_name=getattr(llm, "model_name", "gpt-5-mini"),
            rewrite_error=str(exc),
        )

    return query, _build_rewrite_trace(
        original_query=query,
        rewritten_query=query,
        rewrite_reason="original",
        model_name=getattr(llm, "model_name", "gpt-5-mini"),
    )


def _rule_based_rewrite(
    query: str,
    chat_history: list[dict] | None,
    agency_list: list[str],
) -> str:
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


def rewrite_query_with_history(
    query: str,
    chat_history: list[dict] | None,
    agency_list: list[str],
    llm: object | None = None,
    mode: str = "llm_with_rule_fallback",
    slot_memory: dict[str, str] | None = None,
    max_completion_tokens: int = 16000,
    timeout_seconds: int = 30,
) -> tuple[str, dict[str, object]]:
    """Rewrite underspecified follow-up questions using recent chat history."""

    if not chat_history:
        return query, _build_rewrite_trace(
            original_query=query,
            rewritten_query=query,
            rewrite_reason="original",
        )

    if mode == "rule_only" or llm is None:
        rewritten = _rule_based_rewrite(query, chat_history, agency_list)
        return rewritten, _build_rewrite_trace(
            original_query=query,
            rewritten_query=rewritten,
            rewrite_reason="rule_fallback" if rewritten != query else "original",
        )

    llm_rewritten, llm_trace = _llm_rewrite(
        query,
        chat_history,
        llm,
        slot_memory=slot_memory,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=timeout_seconds,
    )
    if mode == "llm_only" or llm_rewritten != query:
        return llm_rewritten, llm_trace

    rewritten = _rule_based_rewrite(query, chat_history, agency_list)
    if rewritten != query:
        llm_trace["rewritten_query"] = rewritten
        llm_trace["rewrite_applied"] = True
        llm_trace["rewrite_reason"] = "rule_fallback"
    return rewritten, llm_trace
