"""Helpers for history-aware multiturn retrieval."""

from __future__ import annotations

import logging
import re

from bidmate_rag.retrieval.filters import extract_matched_agencies
from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing

logger = logging.getLogger(__name__)
_PRICING = load_pricing()

# 재작성 결과 검증: 숫자/연도/금액 단위를 기준으로 새 사실 주입을 탐지한다.
_NUMBER_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_TOKEN_PATTERN = re.compile(r"20\d{2}\s*년")
_MONEY_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:억원|억|천만원|백만원|만원|원)")

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

아래 JSON 포맷 **한 줄만** 출력하세요. 설명이나 코드블록은 금지입니다.

출력 JSON 스키마:
{{"rewritten_query": "<재작성된 질문 한 문장>", "section_hint": "<RFP 문서에서 이 질문이 묻는 섹션명 혹은 null>"}}

규칙:
- rewritten_query는 현재 질문의 뜻을 유지하고, 생략된 기관명/사업명/문서명/섹션명/항목명만 보완하세요.
- 현재 질문에 없는 새로운 사실(괄호 설명, 예시, 수치, 날짜, 인용 등)은 추가하지 마세요.
- 이전 assistant 답변에만 있던 사실은 가져오지 마세요.
- 현재 질문이 이미 충분히 분명하면 원문 그대로 반환하세요.
- section_hint는 RFP 문서 헤더에 실제로 나올 법한 자연어 구절을 그대로 쓰세요. (예: "평가 기준", "보안 요구사항", "예산", "접근성 진단"). 질문이 특정 섹션을 묻지 않으면 null.

메모리 슬롯:
{slot_memory}

최근 참고 대화:
{history}

현재 질문: {question}

출력:"""


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


def _collect_numeric_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    tokens.update(_YEAR_TOKEN_PATTERN.findall(text))
    tokens.update(_MONEY_TOKEN_PATTERN.findall(text))
    tokens.update(_NUMBER_TOKEN_PATTERN.findall(text))
    return {token.strip() for token in tokens if token.strip()}


def _validate_rewritten_query(
    original_query: str,
    rewritten_query: str,
    chat_history: list[dict] | None,
) -> bool:
    """rewritten_query가 원문·최근 user turn에 없던 숫자/연도/금액을 주입했는지 검사.

    허용되는 출처는 다음 두 가지뿐이다.
    - original_query에 등장한 숫자 토큰
    - 최근 4개 user turn에 등장한 숫자 토큰 (history inheritance 허용)
    """

    rewritten_tokens = _collect_numeric_tokens(rewritten_query)
    if not rewritten_tokens:
        return True

    allowed_tokens = _collect_numeric_tokens(original_query)
    for _, text in _iter_history_texts(chat_history, roles=("user",))[:4]:
        allowed_tokens.update(_collect_numeric_tokens(text))

    leaked = rewritten_tokens - allowed_tokens
    return not leaked


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
    section_hint: str | None = None,
    rewrite_validation: str = "n/a",
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
        "section_hint": section_hint,
        "rewrite_validation": rewrite_validation,
    }
    if rewrite_error:
        trace["rewrite_error"] = rewrite_error
    return trace


def _format_slot_memory(slot_memory: dict[str, str] | None) -> str:
    if not slot_memory:
        return "(없음)"
    lines = [f"- {key}: {value}" for key, value in slot_memory.items() if value]
    return "\n".join(lines) if lines else "(없음)"


def _parse_rewrite_response(raw_text: str) -> tuple[str, str | None]:
    """Rewrite LLM 응답에서 rewritten_query와 section_hint를 추출한다.

    JSON 파싱에 실패하면 raw_text 전체를 rewritten_query로, section_hint는 None으로 간주.
    """
    cleaned = _WHITESPACE_PATTERN.sub(" ", raw_text).strip()
    if not cleaned:
        return "", None
    # 코드블록(```json ... ```) 래핑 제거
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        import json as _json

        payload = _json.loads(cleaned)
    except (ValueError, TypeError):
        return cleaned, None
    if not isinstance(payload, dict):
        return cleaned, None
    rewritten = str(payload.get("rewritten_query") or "").strip()
    hint_raw = payload.get("section_hint")
    if isinstance(hint_raw, str):
        hint = hint_raw.strip() or None
    else:
        hint = None
    if not rewritten:
        return cleaned, hint
    return rewritten, hint


def _llm_rewrite(
    query: str,
    chat_history: list[dict] | None,
    llm: object,
    *,
    slot_memory: dict[str, str] | None = None,
    max_completion_tokens: int = 16000,
    timeout_seconds: int = 30,
) -> tuple[str, dict[str, object]]:
    """LLM을 사용해 후속 질문을 독립적인 검색 쿼리로 재작성한다.

    응답은 `{"rewritten_query": ..., "section_hint": ...}` JSON을 기대하며,
    파싱 실패 시 raw 텍스트를 그대로 rewritten_query로 사용한다.
    """

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
        rewritten, section_hint = _parse_rewrite_response(response.text)
        if rewritten and rewritten != query:
            validation_passed = _validate_rewritten_query(query, rewritten, chat_history)
            if not validation_passed:
                logger.warning(
                    "쿼리 재작성 검증 실패 (새 숫자/연도 주입): '%s' -> '%s'",
                    query,
                    rewritten,
                )
                return query, _build_rewrite_trace(
                    original_query=query,
                    rewritten_query=query,
                    rewrite_reason="validation_failed",
                    model_name=getattr(llm, "model_name", "gpt-5-mini"),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    section_hint=None,
                    rewrite_validation="failed",
                )
            logger.info("쿼리 재작성: '%s' -> '%s' (section=%s)", query, rewritten, section_hint)
            return rewritten, _build_rewrite_trace(
                original_query=query,
                rewritten_query=rewritten,
                rewrite_reason="llm",
                model_name=getattr(llm, "model_name", "gpt-5-mini"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                section_hint=section_hint,
                rewrite_validation="passed",
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
        section_hint=section_hint if 'section_hint' in locals() else None,
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
    validation_failed = llm_trace.get("rewrite_validation") == "failed"
    if mode == "llm_only":
        return llm_rewritten, llm_trace
    if llm_rewritten != query and not validation_failed:
        return llm_rewritten, llm_trace

    rewritten = _rule_based_rewrite(query, chat_history, agency_list)
    if rewritten != query:
        llm_trace["rewritten_query"] = rewritten
        llm_trace["rewrite_applied"] = True
        llm_trace["rewrite_reason"] = "rule_fallback"
        if validation_failed:
            llm_trace["section_hint"] = None
    elif validation_failed:
        # validation 실패 후 rule 폴백도 적용되지 않으면 최종 상태는 original.
        llm_trace["rewritten_query"] = query
        llm_trace["rewrite_applied"] = False
        llm_trace["rewrite_reason"] = "original"
        llm_trace["section_hint"] = None
    return rewritten, llm_trace
