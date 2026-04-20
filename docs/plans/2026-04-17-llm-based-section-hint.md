# LLM 기반 Section Hint + Soft Boost 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 하드코딩된 `SECTION_KEYWORDS` 사전의 **호출 경로를 차단**하고, rewrite LLM이 자유 텍스트로 뽑은 `section_hint`를 hard filter(`where_document`)가 아닌 boost 신호로만 사용해 섹션 오탐으로 인한 핵심 chunk 배제를 제거한다. `SECTION_KEYWORDS`와 `extract_section_hint` 자체는 동료 작업물이므로 **삭제하지 않고 유지**하되 더 이상 사용되지 않는 dead code 상태로 둔다 (후속 정리는 별도 PR에서 진행).

**Architecture:**
- `rewrite_llm`이 JSON(`{"rewritten_query", "section_hint"}`)을 반환하도록 프롬프트와 파싱을 변경한다. `RewriteResponse`에 `section_hint` 필드를 추가해 trace로 흘린다.
- Retriever는 `extract_section_hint(resolved_query)` 호출을 제거하고 rewrite trace의 `section_hint`를 사용한다. `_build_where_document`는 항상 `None`을 반환해 Chroma 본문 `$contains` hard filter를 비활성화한다.
- `rerank_with_boost`의 `section_weight` 기본값을 0.12 → 0.20으로 상향해 soft boost만으로도 순위 교정 효과를 낸다.

**Tech Stack:** Python 3.12, OpenAI (gpt-5-mini), ChromaDB, pytest

---

## File Structure

**Modify:**
- `src/bidmate_rag/providers/llm/base.py` — `RewriteResponse` 데이터클래스에 `section_hint: str | None = None` 필드 추가
- `src/bidmate_rag/providers/llm/openai_compat.py` — `OpenAICompatibleLLM.rewrite()`의 raw text를 유지하되 JSON 파싱은 호출자(multiturn.py)에서 처리 (변경 없음 확인)
- `src/bidmate_rag/retrieval/multiturn.py` — `_SAFE_REWRITE_PROMPT_TEMPLATE`을 JSON 출력으로 교체, `_llm_rewrite`에서 JSON 파싱 후 `RewriteResponse` 빌드, `_build_rewrite_trace`에 `section_hint` 포함, `rewrite_query_with_history` 시그니처에 section_hint 전파
- `src/bidmate_rag/retrieval/retriever.py` — `extract_section_hint` import 제거, `retrieve()`에서 rewrite trace의 `section_hint` 사용, `_build_where_document`가 항상 `None` 반환
- `src/bidmate_rag/retrieval/reranker.py` — `rerank_with_boost` 기본 `section_weight`를 0.12 → 0.20
- `src/bidmate_rag/retrieval/filters.py` — **변경 없음** (`SECTION_KEYWORDS`와 `extract_section_hint`는 유지하되, retriever가 호출하지 않게 하여 dead code 처리)

**Test:**
- `tests/unit/test_filters.py` — `extract_section_hint` 관련 assertion 삭제
- `tests/unit/test_multiturn.py` — JSON 파싱 + section_hint 추출 테스트 추가
- `tests/unit/test_retriever.py` — `test_retriever_retries_without_where_document_when_section_filter_over_prunes` 삭제 또는 업데이트 (where_document은 이제 항상 None), rewrite trace의 section_hint가 rerank_with_boost에 전달되는지 검증
- `tests/unit/test_reranker.py` — 기본 `section_weight` 변경에 따른 경계 케이스 확인

---

## Task 1: `RewriteResponse`에 `section_hint` 필드 추가

**Files:**
- Modify: `src/bidmate_rag/providers/llm/base.py:19-26`
- Test: `tests/unit/test_multiturn.py` (새 테스트는 Task 3에서)

- [ ] **Step 1: 현재 RewriteResponse 확인**

Run: `uv run python -c "from bidmate_rag.providers.llm.base import RewriteResponse; print(RewriteResponse.__dataclass_fields__.keys())"`
Expected: `dict_keys(['text', 'prompt_tokens', 'completion_tokens', 'total_tokens'])`

- [ ] **Step 2: section_hint 필드 추가**

`src/bidmate_rag/providers/llm/base.py:19-26`을 다음으로 교체:

```python
@dataclass
class RewriteResponse:
    """멀티턴 쿼리 재작성 응답 — provider-agnostic."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    section_hint: str | None = None
```

- [ ] **Step 3: 기존 테스트가 깨지지 않는지 확인**

Run: `uv run pytest tests/unit/ -q`
Expected: PASS (default 값이 None이라 기존 호출부 모두 호환)

- [ ] **Step 4: 커밋**

```bash
git add src/bidmate_rag/providers/llm/base.py
git commit -m "feat(llm): RewriteResponse에 section_hint 필드 추가"
```

---

## Task 2: rewrite 프롬프트를 JSON 출력으로 변경

**Files:**
- Modify: `src/bidmate_rag/retrieval/multiturn.py:57-77` (`_SAFE_REWRITE_PROMPT_TEMPLATE`)

- [ ] **Step 1: 프롬프트 템플릿을 JSON 요청으로 교체**

`src/bidmate_rag/retrieval/multiturn.py:57-77`을 다음으로 교체:

```python
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
```

- [ ] **Step 2: 테스트가 여전히 통과하는지 확인 (이 단계까지는 파싱 미구현이므로 일부 테스트가 실패할 수 있음)**

Run: `uv run pytest tests/unit/test_multiturn.py -q`
Expected: 기존 LLM rewrite 테스트가 FAIL 가능 — 다음 Task에서 파싱 추가 후 복구

- [ ] **Step 3: 중간 커밋 생략, Task 3까지 묶어서 커밋**

---

## Task 3: `_llm_rewrite`에서 JSON 파싱 + section_hint 추출

**Files:**
- Modify: `src/bidmate_rag/retrieval/multiturn.py:153-183` (`_build_rewrite_trace`)
- Modify: `src/bidmate_rag/retrieval/multiturn.py:192-247` (`_llm_rewrite`)
- Test: `tests/unit/test_multiturn.py`

- [ ] **Step 1: 실패 테스트 작성 — JSON 응답 파싱**

`tests/unit/test_multiturn.py` 상단에 import 확인 후 파일 끝에 추가:

```python
import json
from unittest.mock import MagicMock

from bidmate_rag.providers.llm.base import RewriteResponse
from bidmate_rag.retrieval.multiturn import rewrite_query_with_history


def test_llm_rewrite_extracts_section_hint_from_json_response() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 평가 기준",
                "section_hint": "평가 기준",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="평가 기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가 기준"
    assert trace["section_hint"] == "평가 기준"
    assert trace["rewrite_reason"] == "llm"


def test_llm_rewrite_falls_back_when_json_invalid() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text="국민연금공단 차세대 ERP 사업의 평가 기준",  # plain text, no JSON
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="평가 기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가 기준"
    assert trace["section_hint"] is None


def test_llm_rewrite_section_hint_null_when_missing() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {"rewritten_query": "국민연금공단 사업 알려줘", "section_hint": None},
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    _, trace = rewrite_query_with_history(
        query="사업 알려줘",
        chat_history=[{"role": "user", "content": "국민연금공단 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert trace["section_hint"] is None
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/test_multiturn.py::test_llm_rewrite_extracts_section_hint_from_json_response -v`
Expected: FAIL (trace에 `section_hint` 키 없음 or 파싱 실패)

- [ ] **Step 3: `_build_rewrite_trace` 확장**

`src/bidmate_rag/retrieval/multiturn.py:153-182`의 `_build_rewrite_trace` 시그니처와 본문을 교체:

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
    }
    if rewrite_error:
        trace["rewrite_error"] = rewrite_error
    return trace
```

- [ ] **Step 4: `_llm_rewrite`에서 JSON 파싱 로직 추가**

`src/bidmate_rag/retrieval/multiturn.py:192-247`의 `_llm_rewrite`를 다음으로 교체:

```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_multiturn.py -v`
Expected: 새로 추가한 3개 테스트 PASS, 기존 테스트 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/bidmate_rag/providers/llm/base.py src/bidmate_rag/retrieval/multiturn.py tests/unit/test_multiturn.py
git commit -m "feat(rewrite): LLM rewrite 응답을 JSON으로 받아 section_hint 추출"
```

---

## Task 4: Retriever가 rewrite trace의 section_hint를 사용 + `where_document` 비활성화

**Files:**
- Modify: `src/bidmate_rag/retrieval/retriever.py:15-22` (import 제거)
- Modify: `src/bidmate_rag/retrieval/retriever.py:213-225` (`_build_where_document`)
- Modify: `src/bidmate_rag/retrieval/retriever.py:390-400` (section_hint 획득부)
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: 실패 테스트 작성 — where_document 항상 None**

`tests/unit/test_retriever.py`에 추가:

```python
def test_retriever_never_sets_where_document_hard_filter() -> None:
    """Soft boost 전환 후 where_document은 항상 None이어야 한다."""
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "이 사업의 예산 규모를 알려줘",
        top_k=5,
        metadata_filter={"발주 기관": "한국원자력연구원"},
    )

    assert vector_store.last_kwargs["where_document"] is None


def test_retriever_forwards_rewrite_section_hint_to_rerank_boost() -> None:
    """rewrite trace의 section_hint가 rerank_with_boost까지 전달돼야 한다."""
    vector_store = FakeVectorStore(
        query_results=[
            _retrieved_chunk("overview", 0.91, agency="국민연금공단", section="사업개요"),
            _retrieved_chunk("budget", 0.80, agency="국민연금공단", section="예산"),
        ]
    )
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 차세대 ERP 예산", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    results = retriever.retrieve(
        "예산은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
    )

    # section_hint가 "예산"으로 들어오면 budget chunk가 boost로 1위가 됨
    assert results[0].chunk.chunk_id == "budget"
```

- [ ] **Step 2: 기존 테스트 중 hard-filter 가정 테스트 삭제**

`tests/unit/test_retriever.py:962-984`의 `test_retriever_retries_without_where_document_when_section_filter_over_prunes` 전체를 삭제. (더 이상 `where_document={"$contains": "예산"}`이 생성되지 않으므로 전제가 무너짐)

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/unit/test_retriever.py::test_retriever_never_sets_where_document_hard_filter tests/unit/test_retriever.py::test_retriever_forwards_rewrite_section_hint_to_rerank_boost -v`
Expected: 둘 다 FAIL (새 동작 미구현)

- [ ] **Step 4: import 정리**

`src/bidmate_rag/retrieval/retriever.py:15-22`의 import 블록에서 `extract_section_hint`를 제거:

```python
from bidmate_rag.retrieval.filters import (
    extract_matched_agencies,
    extract_metadata_filters,
    extract_project_clues,
    extract_range_filters,
    should_fan_out_multi_source_query,
)
```

- [ ] **Step 5: `_build_where_document`는 항상 None 반환**

`src/bidmate_rag/retrieval/retriever.py:213-225`의 `_build_where_document`를 다음으로 교체:

```python
def _build_where_document(
    self,
    query: str,
    where: dict | None,
    section_hint: str | None,
    *,
    force_scoped: bool = False,
) -> dict | None:
    """Chroma 본문 $contains hard filter는 오탐 시 핵심 chunk를 배제하므로 비활성화한다.

    section_hint는 rerank_with_boost의 soft boost 신호로만 사용한다.
    """
    _ = (query, where, section_hint, force_scoped)  # 시그니처 호환 유지
    return None
```

- [ ] **Step 6: `retrieve()`에서 section_hint를 trace에서 가져오도록 수정**

`src/bidmate_rag/retrieval/retriever.py:390-400`을 찾아 다음으로 교체:

```python
section_hint = rewrite_trace.get("section_hint") if isinstance(rewrite_trace, dict) else None
where_document = self._build_where_document(
    resolved_query,
    where,
    section_hint,
    force_scoped=force_scoped,
)
```

즉 `metadata_filter is not None` 조건과 `extract_section_hint(resolved_query)` 호출을 모두 제거하고, 오직 rewrite trace의 값만 사용한다.

- [ ] **Step 7: 전체 retriever 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_retriever.py -v`
Expected: 모든 테스트 PASS (삭제된 테스트 제외, 새 2개 포함)

- [ ] **Step 8: 커밋**

```bash
git add src/bidmate_rag/retrieval/retriever.py tests/unit/test_retriever.py
git commit -m "feat(retriever): rewrite trace의 section_hint 사용, where_document hard filter 제거"
```

---

## Task 5: `rerank_with_boost` section_weight 상향

**Files:**
- Modify: `src/bidmate_rag/retrieval/reranker.py:130-143` (`rerank_with_boost`)
- Test: `tests/unit/test_reranker.py`

- [ ] **Step 1: 실패 테스트 작성 — 새 기본 가중치로 section match가 더 큰 raw score 역전**

`tests/unit/test_reranker.py` 끝에 추가:

```python
def test_rerank_with_boost_default_section_weight_promotes_matching_section() -> None:
    """section_weight 기본값이 0.20 이상이어야 raw score 0.15 차이를 역전 가능."""
    chunks = [
        _make_chunk("overview", 0.85, section="사업개요"),
        _make_chunk("budget", 0.70, section="예산"),
    ]

    results = rerank_with_boost(chunks, query="예산 알려줘", section_hint="예산")

    # 0.70 + 0.20 = 0.90 > 0.85 이어야 budget이 올라옴
    assert results[0].chunk.chunk_id == "budget"
    assert results[0].rank == 1
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/test_reranker.py::test_rerank_with_boost_default_section_weight_promotes_matching_section -v`
Expected: FAIL (현재 기본 0.12로는 0.70+0.12=0.82 < 0.85)

- [ ] **Step 3: 기본 section_weight 상향**

`src/bidmate_rag/retrieval/reranker.py:133`의 한 줄을 변경:

```python
    section_weight = cfg.get("section", 0.20)
```

(기존 0.12 → 0.20. max_total 기본 0.15는 상향된 단일 가중치를 수용하도록 0.25로 조정)

`src/bidmate_rag/retrieval/reranker.py:135`도 다음으로 변경:

```python
    max_total = cfg.get("max_total", 0.25)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_reranker.py -v`
Expected: 새 테스트 PASS, 기존 테스트도 모두 PASS (기존 테스트 중 `test_rerank_with_boost_cap_limits_total_bonus`는 `boost_config`를 명시해 기본값에 의존하지 않으므로 영향 없음)

- [ ] **Step 5: 커밋**

```bash
git add src/bidmate_rag/retrieval/reranker.py tests/unit/test_reranker.py
git commit -m "feat(rerank): section boost 기본 가중치 0.12 → 0.20 상향 (CE 없는 환경 대응)"
```

---

## Task 6: `extract_section_hint` 호출 경로 차단 확인 + 테스트 정리

**Files:**
- Keep as-is: `src/bidmate_rag/retrieval/filters.py` (`SECTION_KEYWORDS`, `extract_section_hint` 유지)
- Modify: `tests/unit/test_filters.py:40-53` (`test_extract_range_and_section_filters` — section_hint 관련 부분만 정리)

- [ ] **Step 1: 코드 내 `extract_section_hint` 사용처 확인**

Grep 툴로 `src/`, `tests/`, `scripts/`에서 `extract_section_hint`와 `SECTION_KEYWORDS`를 검색.

Expected:
- `src/bidmate_rag/retrieval/filters.py` — 정의부만 남음
- `src/bidmate_rag/retrieval/retriever.py` — Task 4에서 import·호출 모두 제거됨 (재확인)
- `tests/unit/test_filters.py` — 기존 테스트가 여전히 `extract_section_hint`를 import·사용 중 → Step 2에서 정리
- 그 외 사용처는 없어야 함

만약 retriever 외 다른 경로에서 호출하고 있다면 그 경로도 같이 끊어야 함 (발견 시 별도 Step 추가).

- [ ] **Step 2: `test_extract_range_and_section_filters` 테스트 축소**

테스트는 `extract_section_hint`의 기존 동작을 그대로 검증하는 것이므로 **함수가 살아있는 한 유지하는 게 맞다**. 단, 기획안 범위에서 추가적인 변경은 없음을 확인하기 위해 이 테스트는 **그대로 둔다**.

즉 이 Task는 "정리 불필요"가 결론일 수도 있음. 검색 결과에 따라 판단:
- `extract_section_hint`가 retriever에서만 호출되고 있었다면 → 테스트·정의 모두 그대로 유지, 추가 변경 없음
- retriever 외 다른 곳에서도 호출하고 있었다면 → 그 호출부만 제거하고 테스트는 유지

- [ ] **Step 3: 전체 테스트 통과 재확인**

Run: `uv run pytest tests/unit/ -q`
Expected: 모든 테스트 PASS. Task 4·5에서 이미 통과했으므로 회귀 없음을 최종 확인하는 단계.

- [ ] **Step 4: 이 Task에서 변경사항이 없으면 커밋 생략**

실제로 코드를 건드릴 필요가 없으면 커밋하지 않음. 변경이 생겼다면:

```bash
git add <변경된 파일>
git commit -m "chore(filters): extract_section_hint dead-code 경로 정리"
```

---

## Task 7: 통합 검증 — 벤치마크 재실행

**Files:**
- Test: `tests/unit/` (전체)
- Eval: bench-c77b768b와 동일 조건

- [ ] **Step 1: 전체 유닛 테스트 실행**

Run: `uv run pytest tests/ -q`
Expected: 모든 테스트 PASS

- [ ] **Step 2: 이전 벤치마크 구성 확인**

Run: `uv run python -c "from pathlib import Path; print(sorted(Path('experiments').rglob('bench-c77b768b*')))"` — 또는 Glob 툴로 `experiments/**/bench-c77b768b*` 검색.

Expected: 벤치마크 config 파일 경로가 나타남 (없으면 사용자에게 재실행 명령을 문의).

- [ ] **Step 3: 벤치마크 스크립트 식별**

Run: `uv run python -c "import subprocess; subprocess.run(['ls', 'scripts/'])"` — 또는 Glob 툴로 `scripts/*eval*` 검색해 runner 확인.

Expected: `scripts/run_benchmark.py` 또는 유사한 스크립트 발견.

- [ ] **Step 4: 벤치마크 재실행 (동일 config)**

Run: 사용자가 bench-c77b768b 실행 시 사용한 명령과 동일하게 실행. (예: `uv run python scripts/run_benchmark.py --provider-config configs/providers/openai_gpt5mini.yaml --eval-set data/eval/type_c.yaml`)

Expected: Type C 13문항에 대해 retrieval·generation 결과 생성.

- [ ] **Step 5: Q043, Q048, Q050 결과 비교**

Before/after로 다음을 확인:
- `where_document`이 debug trace에서 모두 `None`으로 찍히는지
- Q050의 retrieved chunks에서 "평가" 섹션 오탐이 사라졌는지
- Q048의 지재권 관련 chunk가 rank 상위로 올라왔는지
- judge 4개 점수 합산 0.0인 문항이 줄었는지

Expected: Q050/Q043의 섹션 오탐 관련 chunk가 retrieval 결과에서 사라지고, judge 점수 개선.

- [ ] **Step 6: 결과가 악화된 경우 회귀 포인트 식별**

- reranker 쪽 section_weight 0.20이 기존 통과 케이스를 깼는지 확인 (주로 Type A/B 단순 lookup에서 섹션 점수가 과하게 반영되는지)
- 필요 시 `section_weight`를 0.15~0.18로 미세조정 후 벤치 재실행

- [ ] **Step 7: 벤치마크 결과를 커밋에 포함시키지는 않되, PR 설명에 before/after 요약 기록**

`experiments/` 하위 결과 파일은 gitignore 정책에 따름. 커밋 필요시 다음 메시지로:

```bash
git add experiments/<벤치결과_경로>
git commit -m "chore(eval): section_hint 리팩토링 후 Type C 벤치 결과 기록"
```

---

## Task 8: `section_hint` 부스트를 `chunk.section`뿐 아니라 본문/메타 텍스트까지 확장

**Files:**
- Modify: `src/bidmate_rag/retrieval/reranker.py`
- Test: `tests/unit/test_reranker.py`

- [ ] **Step 1: 실패 테스트 작성 — section 필드가 비어 있어도 본문에 section_hint가 있으면 boost**

`tests/unit/test_reranker.py`에 아래 테스트를 추가:

```python
def test_rerank_with_boost_uses_chunk_text_when_section_field_is_empty() -> None:
    chunks = [
        _make_chunk("overview", 0.88, section="사업개요", text="사업 개요와 일반 설명"),
        _make_chunk(
            "security",
            0.78,
            section="",
            text="SER-002 보안 요구사항 USB 반입 반출 통제 규정",
        ),
    ]

    results = rerank_with_boost(
        chunks,
        query="USB 반입 반출 규정 알려줘",
        section_hint="보안 요구사항",
    )

    assert results[0].chunk.chunk_id == "security"
    assert results[0].rank == 1
```

- [ ] **Step 2: 실패 확인**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_reranker.py -q`
Expected: FAIL — 기존 구현은 `section_hint in result.chunk.section`만 보므로 `section=""`인 청크를 올리지 못함.

- [ ] **Step 3: `rerank_with_boost`의 section match 대상을 확장**

`src/bidmate_rag/retrieval/reranker.py`에 helper를 추가:

```python
SECTION_HINT_MIN_MATCH_LEN = 2


def _section_hint_matches_result(section_hint: str | None, result) -> bool:
    """section_hint가 섹션명뿐 아니라 본문/메타 텍스트에도 드러나는지 확인한다."""
    hint_norm = _normalize_text(section_hint)
    if len(hint_norm) < SECTION_HINT_MIN_MATCH_LEN:
        return False

    candidates = (
        result.chunk.section,
        result.chunk.text,
        getattr(result.chunk, "text_with_meta", ""),
    )
    return any(hint_norm in _normalize_text(value) for value in candidates)
```

그리고 `rerank_with_boost()`의 기존 조건:

```python
if section_hint and section_hint in result.chunk.section:
    bonus += section_weight
```

를 아래로 교체:

```python
if _section_hint_matches_result(section_hint, result):
    bonus += section_weight
```

- [ ] **Step 4: task 전용 테스트 재실행**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_reranker.py -q`
Expected: PASS — 기존 `예산`/`보안` 섹션 부스트 테스트와 새 본문 기반 부스트 테스트가 모두 통과.

- [ ] **Step 5: 커밋**

```bash
git add src/bidmate_rag/retrieval/reranker.py tests/unit/test_reranker.py
git commit -m "feat(rerank): section hint boost를 본문/메타 텍스트까지 확장"
```

---

## Task 9: rewrite가 원 질문 핵심어를 잃지 않도록 `원 질문 + 재작성 질문` 듀얼 조회

**Files:**
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: 실패 테스트 작성 — rewrite 적용 시 원 질문도 보조 질의로 조회**

`tests/unit/test_retriever.py`에 아래 테스트를 추가:

```python
def test_retriever_queries_original_text_as_secondary_variant_when_rewrite_applied() -> None:
    vector_store = SequenceFakeVectorStore(
        responses=[
            [_retrieved_chunk("rewritten-hit", 0.91, agency="국민연금공단")],
            [_retrieved_chunk("original-hit", 0.89, agency="국민연금공단")],
        ]
    )
    embedder = FakeEmbedder()
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 보안 규정", "section_hint": "보안 요구사항"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    results = retriever.retrieve(
        "USB 반입 반출해도 되나요?",
        chat_history=[{"role": "user", "content": "국민연금공단 보안 요구사항 알려줘"}],
        top_k=2,
    )

    assert embedder.queries == ["국민연금공단 보안 규정", "USB 반입 반출해도 되나요?"]
    assert [result.chunk.chunk_id for result in results] == ["rewritten-hit", "original-hit"]
```

- [ ] **Step 2: 실패 확인**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_retriever.py::test_retriever_queries_original_text_as_secondary_variant_when_rewrite_applied -q`
Expected: FAIL — 현재 구현은 rewritten query만 조회하므로 embedder.queries 길이가 1이거나 `original-hit`이 빠짐.

- [ ] **Step 3: 원 질문/재작성 질문 결과 union helper 추가**

`src/bidmate_rag/retrieval/retriever.py`에 아래 helper를 추가:

```python
def _merge_query_variant_results(self, primary_results: list, secondary_results: list) -> list:
    """원 질문/재작성 질문 결과를 합쳐 unique chunk 기준으로 정렬한다."""
    if not secondary_results:
        return primary_results

    merged_by_chunk_id: dict[str, object] = {}
    for item in [*primary_results, *secondary_results]:
        chunk_id = item.chunk.chunk_id
        existing = merged_by_chunk_id.get(chunk_id)
        if existing is None or float(item.score) > float(existing.score):
            merged_by_chunk_id[chunk_id] = item

    return sorted(
        merged_by_chunk_id.values(),
        key=lambda result: float(result.score),
        reverse=True,
    )
```

- [ ] **Step 4: rewrite 적용 시 보조 질의 실행**

`retrieve()`에서 rewritten query 1차 조회 직후 아래 분기를 추가:

```python
if rewrite_trace.get("rewrite_applied") and resolved_query != query:
    original_query_embedding = self.embedder.embed_query(query)
    original_query_results = self._query_vector_store(
        query_embedding=original_query_embedding,
        query=query,
        top_k=final_top_k,
        dense_pool_k=dense_pool_k,
        sparse_pool_k=sparse_pool_k,
        where=where,
        where_document=where_document,
        force_scoped=force_scoped,
    )
    results = self._merge_query_variant_results(results, original_query_results)
```

- [ ] **Step 5: task 전용 테스트 재실행**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_retriever.py::test_retriever_queries_original_text_as_secondary_variant_when_rewrite_applied -q`
Expected: PASS — embedder가 rewritten/original 두 쿼리를 모두 기록하고, 결과 리스트에 두 query variant hit이 모두 포함.

- [ ] **Step 6: retriever 관련 회귀 테스트 실행**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_retriever.py -q`
Expected: PASS — 기존 shortlist/fallback/round-robin 테스트와 새 dual-query 테스트가 함께 통과.

- [ ] **Step 7: 커밋**

```bash
git add src/bidmate_rag/retrieval/retriever.py tests/unit/test_retriever.py
git commit -m "feat(retriever): rewrite 적용 시 원 질문을 보조 질의로 함께 조회"
```

---

## Task 10: 회귀 케이스(`Q016`, `Q073`, `Q077`) 분석 및 듀얼 조회 부작용 완화

**Files:**
- Analyze: `artifacts/logs/runs/bench-2c730a68.jsonl`
- Analyze: `artifacts/logs/runs/bench-b4bb4aed.jsonl`
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: 회귀 케이스 before/after 비교**

다음 항목을 `bench-2c730a68`와 `bench-b4bb4aed`에서 각각 비교:
- rewritten_query
- retrieved_chunk_ids / top chunk 순서
- answer preview
- judge score 4종

Run:

```bash
.\.venv\Scripts\python.exe - <<'PY'
import json
from pathlib import Path
for run_id in ("bench-2c730a68", "bench-b4bb4aed"):
    path = Path(f"artifacts/logs/runs/{run_id}.jsonl")
    rows = {row["question_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    print("RUN", run_id)
    for qid in ("Q016", "Q073", "Q077"):
        row = rows[qid]
        print(qid, row["judge_scores"])
        print("chunks", row["retrieved_chunk_ids"][:5])
        print("answer", row["answer"][:500].replace("\n", "\\n"))
        print()
PY
```

Expected: 듀얼 조회 이후 어떤 케이스에서 original query가 오히려 노이즈를 끌어왔는지 확인 가능.

- [ ] **Step 2: 실패 테스트 작성 — 듀얼 조회는 underspecified follow-up에만 적용**

`tests/unit/test_retriever.py`에 아래 테스트를 추가:

```python
def test_retriever_does_not_query_original_variant_for_standalone_questions() -> None:
    vector_store = SequenceFakeVectorStore(
        responses=[[_retrieved_chunk("primary-hit", 0.91, agency="국민연금공단")]]
    )
    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve("경상북도 봉화군 사업 기간은 며칠인가요?", top_k=1)

    assert embedder.queries == ["경상북도 봉화군 사업 기간은 며칠인가요?"]
```

Expected: PASS. standalone question에서는 보조 조회가 없어야 함.

- [ ] **Step 3: 듀얼 조회 적용 조건을 follow-up/underspecified 케이스로 제한**

`src/bidmate_rag/retrieval/retriever.py`에서 현재 조건:

```python
if rewrite_trace.get("rewrite_applied") and resolved_query != query:
```

를 아래처럼 좁힌다:

```python
follow_up_indicators = ("그 ", "그것", "해당", "위에서", "방금", "그러면", "그럼")
should_query_original_variant = (
    rewrite_trace.get("rewrite_applied")
    and resolved_query != query
    and any(indicator in query for indicator in follow_up_indicators)
)

if should_query_original_variant:
```

Expected: 명시적 standalone question에는 듀얼 조회를 적용하지 않고, 진짜 후속질문에서만 원 질문 복원 검색이 동작.

- [ ] **Step 4: task 전용 테스트 재실행**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest \
  tests\unit\test_retriever.py::test_retriever_queries_original_text_as_secondary_variant_when_rewrite_applied \
  tests\unit\test_retriever.py::test_retriever_does_not_query_original_variant_for_standalone_questions -q
```

Expected: PASS — 후속질문은 듀얼 조회 유지, standalone은 단일 조회 유지.

- [ ] **Step 5: retriever 전체 회귀 테스트 재실행**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_retriever.py -q`
Expected: PASS.

- [ ] **Step 6: eval_batch_35 재실행 후 회귀 3건 재확인**

Run:

```bash
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m bidmate_rag.cli.eval --evaluation-path data/eval/eval_v1/eval_batch_35.csv --provider-config configs/providers/openai_gpt5mini.yaml --progress
```

Expected:
- `Q016`, `Q073`, `Q077` 중 최소 2건 이상이 baseline 수준으로 회복
- `Q043`, `Q050`, `Q054`, `Q076`의 개선은 유지

- [ ] **Step 7: 커밋**

```bash
git add src/bidmate_rag/retrieval/retriever.py tests/unit/test_retriever.py
git commit -m "fix(retriever): 듀얼 조회를 후속질문 케이스로 제한해 회귀 완화"
```

---

## Task 11: 하드코딩 allowlist를 제거하고 질문에서 직접 뽑은 구체 명사구만 `where_document`에 사용

**Files:**
- Analyze: `artifacts/logs/runs/bench-2c730a68.jsonl`
- Analyze: `artifacts/logs/runs/bench-b4bb4aed.jsonl`
- Modify: `src/bidmate_rag/retrieval/filters.py`
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: 원칙 재정의**

이번 단계의 목표는 `WHERE_DOCUMENT_STRONG_HINTS` 같은 완성형 하드코딩 목록을 없애고, 질문 안에 실제로 들어 있는 구체 표현만 동적으로 뽑아 hard filter에 쓰는 것이다.

원칙:
- `section_hint`는 계속 `soft boost` 전용으로 사용한다.
- `where_document`는 질문 안에 나온 표현을 그대로 쓰는 경우에만 허용한다.
- broad section label(`평가`, `기준`, `요구사항`)은 hard filter에 쓰지 않는다.
- 비교형 / fan-out / scoped query에서는 계속 `where_document=None`을 유지한다.

Expected: `Q016/Q073/Q077`처럼 직접 조회형 질문은 살리고, `Q043/Q050`처럼 section_hint 오탐으로 잘못 좁히는 문제는 다시 만들지 않는다.

- [ ] **Step 2: 실패 테스트 작성 - 질문 안의 구체 명사구를 anchor로 추출**

`tests/unit/test_retriever.py`에 아래 테스트를 추가:

```python
def test_retriever_uses_dynamic_where_document_anchor_from_query_phrase() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        '경상북도 봉화군의 "재난통합관리시스템 고도화 사업"의 사업기간은 며칠로 명시되어 있습니까?',
        top_k=3,
    )

    assert vector_store.last_kwargs["where_document"] == {"$contains": "사업기간"}
```

또한 broad hint가 hard filter로 번지지 않는지 확인하는 테스트를 유지/보강:

```python
def test_retriever_does_not_use_section_hint_as_where_document_anchor() -> None:
    ...
    assert vector_store.last_kwargs["where_document"] is None
```

Expected: 현재 구현에서는 첫 테스트가 FAIL하거나, broad section_hint가 섞여 hard filter가 과적용될 수 있다.

- [ ] **Step 3: 동적 phrase 추출 helper 추가**

`src/bidmate_rag/retrieval/filters.py`에 `extract_where_document_anchor()`를 추가:

```python
WHERE_DOCUMENT_BLOCKLIST = {"평가", "기준", "요구사항", "사업개요", "예산"}

DIRECT_PHRASE_PATTERN = re.compile(
    r"([가-힣A-Za-z0-9·()/-]{2,20}"
    r"(?:기간|기한|일수|인력|인수인계|지체상금률|지체상금|버전|명칭))"
)


def extract_where_document_anchor(query: str) -> str | None:
    candidates = []
    for match in DIRECT_PHRASE_PATTERN.finditer(query):
        phrase = match.group(1).strip()
        if phrase in WHERE_DOCUMENT_BLOCKLIST:
            continue
        candidates.append(phrase)

    if not candidates:
        return None

    # 더 긴 구체 표현 우선
    candidates.sort(key=len, reverse=True)
    return candidates[0]
```

핵심:
- 고정 완성어 목록이 아니라 `질문에 실제로 등장한 표현`을 반환한다.
- `section_hint`는 여기서 받지 않는다.
- 정규식의 역할은 경계 인식이지, 특정 질문 답을 외워 넣는 것이 아니다.

- [ ] **Step 4: `retrieve()`에서 dynamic anchor만 hard filter에 연결**

`src/bidmate_rag/retrieval/retriever.py`에서 `where_document` 계산 규칙을 아래처럼 바꾼다:

```python
dynamic_anchor = extract_where_document_anchor(resolved_query)

if where_document is None and not force_scoped and where and dynamic_anchor:
    where_document = {"$contains": dynamic_anchor}
```

원칙:
- `section_hint`는 `rerank_with_boost(..., section_hint=...)`에만 전달한다.
- `where_document`는 오직 `resolved_query`의 실제 표현에서만 만든다.
- shortlist / metadata constraint가 없는 broad query에는 적용하지 않는다.

- [ ] **Step 5: focused test 실행**

Run:

```bash
.\.venv\Scripts\python.exe -m pytest \
  tests\unit\test_retriever.py::test_retriever_uses_dynamic_where_document_anchor_from_query_phrase \
  tests\unit\test_retriever.py::test_retriever_does_not_use_section_hint_as_where_document_anchor \
  tests\unit\test_retriever.py::test_retriever_does_not_apply_where_document_filter_for_scoped_comparison -q
```

Expected:
- 질문에 직접 나온 구체 명사구는 `where_document` anchor로 사용
- `section_hint`만으로는 hard filter가 생기지 않음
- scoped comparison query는 계속 `where_document=None`

- [ ] **Step 6: retriever 전체 테스트 실행**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\test_retriever.py -q`
Expected: PASS.

- [ ] **Step 7: eval_batch_35 재실행**

Run:

```bash
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m bidmate_rag.cli.eval --evaluation-path data/eval/eval_v1/eval_batch_35.csv --provider-config configs/providers/openai_gpt5mini.yaml --progress
```

Expected:
- `Q016`, `Q073`, `Q077`은 회복되거나 최소한 후보 랭킹이 개선
- `Q043`, `Q050`은 section_hint hard filter 부작용 없이 기존 개선을 유지
- 전체적으로는 `원 질문 + 재작성 질문 병렬 조회`와 `soft boost`가 주력이고, `where_document`는 직접 조회형 질문에서만 보조 수단으로 남는다

- [ ] **Step 8: commit**

```bash
git add docs/plans/2026-04-17-llm-based-section-hint.md src/bidmate_rag/retrieval/filters.py src/bidmate_rag/retrieval/retriever.py tests/unit/test_retriever.py
git commit -m "fix(retriever): use dynamic query phrase for where_document anchor"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] SECTION_KEYWORDS·extract_section_hint **호출 경로 차단** (정의는 유지, dead code화) — Task 4·6
- [x] LLM rewrite 출력에 section_hint 추가 — Task 1·2·3
- [x] where_document hard filter 제거 — Task 4
- [x] rerank_with_boost section_weight 상향 — Task 5
- [x] 기존 테스트 업데이트 — Task 4·5
- [x] 통합 검증 — Task 7
- [x] section_hint boost 대상을 본문/메타 텍스트까지 확장 — Task 8
- [x] rewrite 적용 시 원 질문 보조 조회 추가 — Task 9
- [x] 회귀 케이스 분석 및 듀얼 조회 적용 조건 축소 — Task 10

**Placeholder scan:** 모든 step이 구체적 코드/명령 포함, TBD 없음.

**Type consistency:**
- `section_hint`는 `str | None` 타입으로 `RewriteResponse` → `rewrite_trace["section_hint"]` → `rerank_with_boost(section_hint=...)` 전구간 동일.
- `_parse_rewrite_response`의 반환은 `tuple[str, str | None]`로 Task 3 내 일관.
- `_build_rewrite_trace`의 `section_hint` 파라미터는 default `None`으로 기존 rule-based 경로 호환.
