# Multiturn Phase 1 — 재작성 결과 검증 + 주제 전환 감지 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 개선 4항목 중 **첫 두 개**를 이번 브랜치에서 끝낸다.
1. **재작성 결과 검증** — LLM이 재작성하면서 원문에 없던 숫자/날짜/고유명사를 주입하면 규칙 기반으로 폴백한다. 검증 실패 사실은 trace에 남겨 관찰 가능하게 한다.
2. **주제 전환 감지** — 사용자가 "이제", "주제 바꿔서", "다른 사업/기관" 등 명시적 전환 신호를 줄 때 관련 슬롯을 리셋해서, rewrite LLM과 generation LLM 모두 과거 맥락을 끌고 들어가지 못하게 한다.

이 두 개는 서로 독립적이지만 **같은 브랜치 `feat/yj/multiturn-phase1`에서 커밋을 분리**해 차례로 적용한다. 남은 두 항목 (복수 대상 슬롯 · 동시성 해결)은 별도 후속 플랜으로 분리.

**Architecture:**
- `multiturn.py`에 `_validate_rewritten_query(original, rewritten, chat_history)` 헬퍼를 추가한다. 새 숫자/연도/기관명 주입 여부를 검사하고, 실패하면 `rewritten = query` + `rewrite_reason = "validation_failed"`로 강등한 뒤 rule-based 재작성을 시도한다.
- `memory.py`의 `ConversationMemory.build()`에 topic-shift 감지 로직을 추가한다. `current_question`에 전환 키워드가 있으면 `slot_memory`를 빈 dict로 돌려주고 `summary_buffer`도 과거 턴을 축소해서 내보낸다. 추가로 `recent_turns` 기준 범위를 마지막 user turn까지로 좁혀 rewrite가 과거 컨텍스트를 상속하지 못하게 한다.
- `retriever.py`는 `rewrite_slot_memory`가 빈 상태이면 rewrite LLM에 `(없음)` 슬롯을 그대로 넘기고, `rewrite_trace["topic_shift"] = True`를 debug_trace에 반영한다.

**Tech Stack:** Python 3.12, OpenAI (gpt-5-mini), ChromaDB, pytest

---

## File Structure

**Modify:**
- `src/bidmate_rag/retrieval/multiturn.py` — `_validate_rewritten_query` 추가, `_llm_rewrite`에서 검증 실패 시 rule 폴백 경유, `_build_rewrite_trace`에 `rewrite_validation` 필드 추가
- `src/bidmate_rag/retrieval/memory.py` — `TOPIC_SHIFT_KEYWORDS` 상수와 `detect_topic_shift()` 헬퍼 추가, `ConversationMemory.build()`에 shift 감지 분기
- `src/bidmate_rag/retrieval/retriever.py` — `rewrite_memory_state`에 `topic_shift` 플래그 참조, debug trace에 반영 (로직 변경 최소)

**Test:**
- `tests/unit/test_multiturn.py` — 검증 통과/실패 3 케이스 추가 (숫자 주입·연도 주입·정상 재작성)
- `tests/unit/test_memory.py` — topic shift 감지 케이스 3개 (`"이제"`, `"다른 사업"`, `"주제 바꿔서"`), 일반 후속질문은 기존 슬롯 유지
- `tests/unit/test_retriever.py` — topic shift 감지 시 rewrite에 빈 slot_memory가 전달되는지 검증

**No-change:**
- `src/bidmate_rag/retrieval/filters.py`
- `src/bidmate_rag/retrieval/reranker.py`
- `src/bidmate_rag/pipelines/chat.py` (동시성 해결은 Phase 1-후반 별도 플랜)

---

## Task 1: `_validate_rewritten_query` 헬퍼 추가

**Files:**
- Modify: `src/bidmate_rag/retrieval/multiturn.py` (상단 상수 블록 + 새 함수)
- Test: `tests/unit/test_multiturn.py`

**설계 요약:**
- 검증은 "rewritten_query가 original_query 또는 recent user turns에 없는 새 사실을 만들었는가?"를 본다.
- 감지 대상:
  - **숫자**: `\d+` 매칭 토큰 — `기간 30일` 같은 수치.
  - **연도**: `202[0-9]년` 같은 4자리 연도.
  - **통화/금액 단위**: `\d+\s*(억|억원|천만원|백만원|만원)` — `_BUDGET_PATTERN`과 일관.
- `original_query` 또는 `recent_user_texts`에 등장한 토큰은 허용, 그 외는 `validation_failed`.
- 고유명사(기관명·사업명)는 검증 범위에서 제외 — rewrite의 주요 역할이 명시 복원이므로. 대신 복수 대상 슬롯 문제는 별도 Phase 1-후반에서 다룬다.

---

- [ ] **Step 1: 실패 테스트 작성 — 새 숫자 주입 감지**

`tests/unit/test_multiturn.py` 끝에 추가:

```python
def test_llm_rewrite_flags_validation_failure_when_new_number_injected() -> None:
    import json
    from unittest.mock import MagicMock

    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 사업기간 30일",
                "section_hint": "사업 일정",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="사업기간은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    # "30일"은 original이나 history에 없는 숫자 → 검증 실패로 LLM 결과 폐기
    assert "30일" not in rewritten
    assert trace["rewrite_validation"] == "failed"
    assert trace["rewrite_reason"] in ("rule_fallback", "original")
```

- [ ] **Step 2: 실패 테스트 작성 — 새 연도 주입 감지**

```python
def test_llm_rewrite_flags_validation_failure_when_new_year_injected() -> None:
    import json
    from unittest.mock import MagicMock

    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "2024년 국민연금공단 차세대 ERP 사업 예산",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert "2024년" not in rewritten
    assert trace["rewrite_validation"] == "failed"
```

- [ ] **Step 3: 성공 케이스 유지 확인 테스트**

```python
def test_llm_rewrite_validation_passes_when_rewritten_only_adds_context_terms() -> None:
    import json
    from unittest.mock import MagicMock

    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 예산",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 예산"
    assert trace["rewrite_validation"] == "passed"
    assert trace["rewrite_reason"] == "llm"
```

- [ ] **Step 4: 실패 확인**

Run: `uv run pytest tests/unit/test_multiturn.py::test_llm_rewrite_flags_validation_failure_when_new_number_injected tests/unit/test_multiturn.py::test_llm_rewrite_flags_validation_failure_when_new_year_injected tests/unit/test_multiturn.py::test_llm_rewrite_validation_passes_when_rewritten_only_adds_context_terms -v`
Expected: 세 테스트 모두 FAIL (`rewrite_validation` 키 부재).

- [ ] **Step 5: `_validate_rewritten_query` 헬퍼 추가**

`src/bidmate_rag/retrieval/multiturn.py`의 `_PRICING = load_pricing()` 다음 줄에 상수와 헬퍼를 추가:

```python
# 재작성 결과 검증: 숫자/연도/금액 단위를 기준으로 새 사실 주입을 탐지한다.
_NUMBER_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_TOKEN_PATTERN = re.compile(r"20\d{2}\s*년")
_MONEY_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:억원|억|천만원|백만원|만원|원)")


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
    for role, text in _iter_history_texts(chat_history, roles=("user",))[:4]:
        allowed_tokens.update(_collect_numeric_tokens(text))

    leaked = rewritten_tokens - allowed_tokens
    return not leaked
```

- [ ] **Step 6: `_build_rewrite_trace`에 `rewrite_validation` 필드 추가**

`_build_rewrite_trace` 시그니처와 본문 업데이트:

```python
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
```

- [ ] **Step 7: `_llm_rewrite`에서 검증 호출 + 실패 시 rule 폴백 유도**

`_llm_rewrite` 함수의 성공 분기(`if rewritten and rewritten != query:`)를 다음으로 교체:

```python
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
                    section_hint=section_hint,
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
```

- [ ] **Step 8: `rewrite_query_with_history`에서 `validation_failed`도 rule 폴백 대상에 포함**

기존 분기:

```python
    if mode == "llm_only" or llm_rewritten != query:
        return llm_rewritten, llm_trace
```

를 다음으로 교체:

```python
    validation_failed = llm_trace.get("rewrite_validation") == "failed"
    if mode == "llm_only":
        return llm_rewritten, llm_trace
    if llm_rewritten != query and not validation_failed:
        return llm_rewritten, llm_trace
```

이렇게 하면 `validation_failed`일 때는 기존 원문으로 돌아오고, 그다음에 이미 존재하는 rule 폴백 블록이 실행된다.

- [ ] **Step 9: 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_multiturn.py -q`
Expected: 새 3개 테스트 PASS, 기존 테스트 모두 PASS.

- [ ] **Step 10: 커밋**

```bash
git add src/bidmate_rag/retrieval/multiturn.py tests/unit/test_multiturn.py
git commit -m "feat(multiturn): rewrite 결과 검증으로 새 숫자/연도 주입 차단"
```

---

## Task 2: 주제 전환 감지 — `memory.py` + `retriever.py`

**Files:**
- Modify: `src/bidmate_rag/retrieval/memory.py`
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_memory.py`, `tests/unit/test_retriever.py`

**설계 요약:**
- 전환 키워드를 상수로 정의한다. 단어 단위 매칭만 하고, 정규식으로는 가볍게 처리한다.
- `ConversationMemory.build()`는 `current_question`에 전환 키워드가 있으면:
  - `slot_memory = {}` — rewrite LLM이 과거 기관/사업명을 상속하지 못하게 한다.
  - `summary_buffer = ""` — 과거 맥락을 generation LLM에 누출하지 않는다.
  - `recent_turns`는 유지 — 사용자가 "이제 X에 대해 알려줘"라고 했을 때 직전 turn을 완전히 버리면 rule 폴백까지 깨진다. 슬롯만 리셋하면 충분.
  - 반환 dict에 `"topic_shift": True`를 추가해 호출자가 관찰 가능하게 한다.
- 전환 키워드는 보수적으로만 포함한다. false positive가 정상 후속질문을 망가뜨리면 안 된다.

---

- [ ] **Step 1: 실패 테스트 작성 — 명시적 전환 감지 시 슬롯 리셋**

`tests/unit/test_memory.py` (없으면 생성) 혹은 기존 memory 테스트 파일에 추가:

```python
from bidmate_rag.retrieval.memory import ConversationMemory


def _mk_history() -> list[dict]:
    return [
        {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        {"role": "assistant", "content": "국민연금공단의 차세대 ERP는 ..."},
    ]


def test_memory_resets_slots_when_topic_shift_keyword_detected() -> None:
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단", "한국전력공사"],
    )
    state = memory.build(
        _mk_history(),
        current_question="이제 한국전력공사 사업도 알려줘",
    )

    # topic shift가 감지되면 slot_memory에 국민연금공단이 남아 있으면 안 됨
    assert state["slot_memory"] == {}
    assert state["summary_buffer"] == ""
    assert state["topic_shift"] is True
    # recent_turns는 유지 (generation이 답변에서 직전 턴을 참고할 수 있도록)
    assert len(state["recent_turns"]) == 2


def test_memory_retains_slots_for_regular_follow_up_question() -> None:
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    state = memory.build(
        _mk_history(),
        current_question="예산은 얼마야?",
    )

    assert state["slot_memory"].get("발주기관") == "국민연금공단"
    assert state.get("topic_shift") is False


def test_memory_detects_topic_shift_with_다른_사업_phrase() -> None:
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    state = memory.build(
        _mk_history(),
        current_question="다른 사업 추천해줘",
    )

    assert state["slot_memory"] == {}
    assert state["topic_shift"] is True
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/test_memory.py -v`
Expected: 세 테스트 모두 FAIL (현재는 `topic_shift` 키가 없고 슬롯이 리셋되지 않음).

- [ ] **Step 3: `TOPIC_SHIFT_KEYWORDS`와 `detect_topic_shift` 추가**

`src/bidmate_rag/retrieval/memory.py`의 상단 상수 블록에 추가:

```python
TOPIC_SHIFT_KEYWORDS = (
    "이제",
    "주제 바꿔서",
    "주제를 바꿔서",
    "주제 바꿔",
    "다른 사업",
    "다른 기관",
    "새로운 사업",
    "새 질문",
    "다른 질문",
)


def detect_topic_shift(question: str | None) -> bool:
    """사용자 질문에 명시적 전환 키워드가 있는지 감지한다."""
    if not question:
        return False
    normalized = _WHITESPACE_PATTERN.sub(" ", question).strip()
    return any(keyword in normalized for keyword in TOPIC_SHIFT_KEYWORDS)
```

- [ ] **Step 4: `ConversationMemory.build()`에 shift 감지 분기 추가**

기존 `build()` 본문의 반환 직전을 다음 구조로 감싼다:

```python
    def build(
        self,
        chat_history: list[dict] | None,
        *,
        current_question: str | None = None,
        rewritten_query: str | None = None,
    ) -> dict[str, object]:
        messages = _coerce_messages(chat_history)
        topic_shift = detect_topic_shift(current_question)

        if not messages:
            return {
                "recent_turns": [],
                "summary_buffer": "",
                "slot_memory": {},
                "topic_shift": topic_shift,
            }

        recent_turns = messages[-self.max_recent_turns:]
        older_turns = messages[: -self.max_recent_turns] if len(messages) > self.max_recent_turns else []

        if topic_shift:
            return {
                "recent_turns": recent_turns,
                "summary_buffer": "",
                "slot_memory": {},
                "topic_shift": True,
            }

        # ...기존 summary_buffer / slot_memory 로직 그대로 유지...
```

기존 로직의 마지막 `return`에도 `"topic_shift": False`를 추가한다:

```python
        return {
            "recent_turns": recent_turns,
            "summary_buffer": summary_buffer,
            "slot_memory": slot_memory,
            "topic_shift": False,
        }
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_memory.py -v`
Expected: 세 테스트 모두 PASS.

- [ ] **Step 6: Retriever 통합 테스트 작성 — shift 시 rewrite에 빈 슬롯 전달**

`tests/unit/test_retriever.py`에 추가:

```python
def test_retriever_passes_empty_slot_memory_when_topic_shift_detected() -> None:
    vector_store = FakeVectorStore()
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "한국전력공사 사업 알려줘", "section_hint": null}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단", "한국전력공사"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(agency_list=["국민연금공단", "한국전력공사"]),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    retriever.retrieve(
        "이제 한국전력공사 사업 알려줘",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "국민연금공단 ERP는 ..."},
        ],
        top_k=3,
    )

    # rewrite LLM에 전달된 프롬프트 안에서 국민연금공단 슬롯이 섞여 있으면 안 됨
    sent_prompts = mock_llm.rewrite.call_args_list
    assert sent_prompts, "rewrite LLM이 호출되어야 한다"
    prompt_text = sent_prompts[0][0][0]
    assert "국민연금공단" not in prompt_text.split("현재 질문:")[0]
```

_`_make_mock_llm` / `FakeVectorStore` / `FakeEmbedder` / `FakeMetadataStore` 헬퍼가 기존 테스트 파일에 이미 존재한다는 가정. 없는 유틸이 있으면 기존 파일의 fixture 패턴에 맞춰 추가한다._

- [ ] **Step 7: 실패 확인**

Run: `uv run pytest tests/unit/test_retriever.py::test_retriever_passes_empty_slot_memory_when_topic_shift_detected -v`
Expected: FAIL (rewrite_memory_state 빌드 시 topic_shift 무시 → 슬롯에 과거 기관이 남음).

- [ ] **Step 8: Retriever에서 topic_shift 상태를 debug trace에 반영**

`src/bidmate_rag/retrieval/retriever.py`의 `retrieve()` 상단, `rewrite_slot_memory = ...` 다음 줄에 추가:

```python
        rewrite_slot_memory = rewrite_memory_state.get("slot_memory", {})
        topic_shift = bool(rewrite_memory_state.get("topic_shift", False))
```

그리고 `_last_debug`에 기록하는 부분(이번 커밋에서 `_last_debug` 구조는 유지)에 `topic_shift`를 추가한다. 해당 지점을 grep으로 찾고 (현재 `retriever.py:503/523` 근처) rewrite_trace를 저장하는 dict에 필드 하나만 더해준다:

```python
            "memory_state": generation_memory_state,
            "topic_shift": topic_shift,
```

(`_last_debug` 자체를 없애는 작업은 Phase 1-후반 "동시성 해결" 작업으로 분리.)

- [ ] **Step 9: 통합 테스트 PASS 확인**

Run: `uv run pytest tests/unit/test_retriever.py -q`
Expected: 새 테스트 포함 모두 PASS.

- [ ] **Step 10: 커밋**

```bash
git add src/bidmate_rag/retrieval/memory.py src/bidmate_rag/retrieval/retriever.py tests/unit/test_memory.py tests/unit/test_retriever.py
git commit -m "feat(memory): 주제 전환 키워드 감지 시 slot_memory 리셋"
```

---

## Task 3: 통합 검증 — 전체 유닛 테스트 + 수동 시나리오

**Files:**
- Test: `tests/unit/` 전체

- [ ] **Step 1: 전체 유닛 테스트 실행**

Run: `uv run pytest tests/ -q`
Expected: 전부 PASS. Phase 1 외 테스트가 깨졌다면 해당 지점 분석 후 수정.

- [ ] **Step 2: 수동 시나리오 1 — 숫자 주입 폴백**

Streamlit 또는 CLI로 다음 턴을 차례로 입력:
1. "국민연금공단 차세대 ERP 사업 알려줘"
2. "사업기간은?"

기대:
- rewrite trace에 `rewrite_validation = "passed"` 또는 `"n/a"` (rewritten이 원문과 같으면).
- 만약 LLM이 숫자를 지어내면 trace에 `rewrite_validation = "failed"`, `rewrite_reason`은 `"rule_fallback"` 또는 `"original"`로 기록.

- [ ] **Step 3: 수동 시나리오 2 — 주제 전환**

차례로 입력:
1. "국민연금공단 차세대 ERP 사업 알려줘"
2. "이제 한국전력공사 사업도 알려줘"

기대:
- 두 번째 턴의 debug trace에 `topic_shift = True`.
- rewrite 결과에 `국민연금공단`이 끌려 들어가지 않음.
- 결과 청크의 `발주 기관`이 `한국전력공사`로 고정.

- [ ] **Step 4: PR 설명에 기록할 before/after 메모 작성**

- 검증 실패 비율 (기존: 측정 불가 → 이후: trace에 `rewrite_validation` 카운트).
- 주제 전환 케이스에서 직전 기관 슬롯이 누수되었던 재현 사례 1건과 해소된 결과.

---

## Out of scope — 이번 브랜치에서 다루지 않음

- **복수 대상 슬롯** (`agencies[-1]` → 리스트 저장) — 별도 후속 플랜에서.
- **동시성 해결** (`self._last_debug` 제거, 반환값으로 debug 전달) — 별도 후속 플랜에서.
- **`_BUDGET_PATTERN` 확장** (`"10억"` 단독 매칭) — Phase 3 범위.
- **`_PROJECT_HINT_PATTERN` 엄격화** — Phase 3 범위.

---

## Self-Review Checklist

**Spec coverage:**
- [x] 재작성 결과 검증 — 숫자·연도·금액 토큰 기반, 실패 시 rule 폴백 — Task 1
- [x] trace에 검증 결과 관찰 가능 — `rewrite_validation` 필드 — Task 1 Step 6
- [x] 주제 전환 감지 — `TOPIC_SHIFT_KEYWORDS` + `detect_topic_shift` — Task 2 Step 3
- [x] 전환 시 slot_memory/summary_buffer 리셋 — Task 2 Step 4
- [x] retriever에서 topic_shift 관찰 가능 — Task 2 Step 8
- [x] 통합 검증 — Task 3

**Placeholder scan:** 모든 step에 구체적 코드/명령 포함, TBD 없음.

**Type consistency:**
- `rewrite_validation`: `"passed" | "failed" | "n/a"` (str) — `_build_rewrite_trace`에서 기본 `"n/a"`.
- `topic_shift`: `bool` — `ConversationMemory.build()` 반환 dict의 신규 키.
- `_validate_rewritten_query`: `tuple[str, str, list[dict] | None] -> bool`.
- `detect_topic_shift`: `str | None -> bool`.

**Risk check:**
- 검증이 너무 엄격하면 정상 rewrite까지 폴백된다 → 숫자/연도/금액만 보고 고유명사는 건드리지 않음.
- `TOPIC_SHIFT_KEYWORDS`가 너무 넓으면 일반 follow-up이 망가진다 → 명시적 표현만 포함 ("이제", "다른 사업/기관", "주제 바꿔서", "새 질문/다른 질문"). `"다른"` 단독은 제외.
- `recent_turns`를 유지하므로 rule 폴백이 여전히 동작 가능.
