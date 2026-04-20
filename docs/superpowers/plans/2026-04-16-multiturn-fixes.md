# Multiturn Prompt Flow — Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 멀티턴 RAG 파이프라인의 4대 결함(LLM rewrite 토큰 기근, 프로바이더 추상화 우회, 메모리 이중 빌드, rewrite된 쿼리 미활용)을 `fix/am/multiturn-prompt-recover` 브랜치에서 일괄 수정한다.

**Architecture:** `BaseLLMProvider`에 `rewrite()` 추상 메서드를 도입해 OpenAI/HF 양쪽에서 공통 인터페이스로 호출한다. 메모리 상태 소유권을 `RAGRetriever`로 단일화하고 `_last_debug["memory_state"]`로 chat 파이프라인에 전달한다. `resolved_query`를 retriever 내부 모든 경로에서 일관 사용한다. 토큰/타임아웃은 `RewriteConfig`로 노출한다.

**Tech Stack:** Python 3.12 · Pydantic v2 · pytest · transformers (HF) · openai (OpenAI SDK) · uv

**Base branch:** `fix/am/multiturn-prompt-recover` (origin/feat/yj의 후속, 커밋 `24c8d58` 위에 쌓음)

---

## File Structure

수정될 파일:
- `src/bidmate_rag/retrieval/retriever.py` — `resolved_query` 일관화, memory_state 소유
- `src/bidmate_rag/pipelines/chat.py` — memory_state를 debug에서 소비
- `src/bidmate_rag/providers/llm/base.py` — `RewriteResponse` 데이터클래스 + `rewrite()` 인터페이스
- `src/bidmate_rag/providers/llm/openai_compat.py` — `rewrite()` 구현
- `src/bidmate_rag/providers/llm/hf_local.py` — `rewrite()` 구현 (local generator 재사용)
- `src/bidmate_rag/retrieval/multiturn.py` — `llm.rewrite()` 호출로 전환, config 파라미터 수용
- `src/bidmate_rag/config/settings.py` — `RewriteConfig.max_completion_tokens`, `timeout_seconds` 필드 추가
- `configs/retrieval.yaml` — 새 필드 명시
- `tests/unit/test_multiturn.py` — mock 리팩터 (client 대신 rewrite 메서드)
- `tests/integration/test_chat_pipeline.py` — memory state 단일 소유 검증 추가

신규 테스트:
- `tests/unit/test_provider_rewrite.py` — OpenAI/HF provider rewrite 단위 테스트

---

## Task 1: D1 — Retriever에서 `resolved_query` 일관 사용

**배경:** [retriever.py:309-310, 340-341](../../src/bidmate_rag/retrieval/retriever.py#L309-L341) 에서 `_augment_where_with_project_docs`/`_augment_where_with_history_docs` 호출 시 원문 `query`를 사용 중. 후속 질의("그 사업 예산은?")에서 LLM rewrite로 복원된 맥락이 문서 부스팅 경로에 반영되지 않아 멀티턴 이점이 절반만 구현된다.

**Files:**
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_retriever_multiturn.py` (신규)

- [ ] **Step 1: 실패 테스트 작성** — `tests/unit/test_retriever_multiturn.py`

```python
"""Retriever가 rewrite된 쿼리를 문서 부스팅/히스토리 경로에 전파하는지 검증."""

from unittest.mock import MagicMock

from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.schema import Chunk, RetrievedChunk


class _RecordingMetadataStore:
    def __init__(self, agency_list: list[str], relevant_docs: list[str]) -> None:
        self.agency_list = agency_list
        self._relevant_docs = relevant_docs
        self.find_relevant_docs_calls: list[str] = []

    def find_relevant_docs(self, query: str, top_n: int = 3) -> list[str]:
        self.find_relevant_docs_calls.append(query)
        return self._relevant_docs


def _make_chunk(chunk_id: str = "c-1") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}-doc",
        text="t",
        text_with_meta="t",
        char_count=1,
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={"파일명": "a.hwp"},
    )
    return RetrievedChunk(rank=1, score=0.9, chunk=chunk)


def test_retriever_uses_resolved_query_for_project_clue_augmentation() -> None:
    metadata_store = _RecordingMetadataStore(
        agency_list=["국민연금공단"],
        relevant_docs=["차세대_ERP.hwp", "차세대_ERP_2.hwp"],
    )
    vector_store = MagicMock()
    vector_store.query.return_value = [_make_chunk()]
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8

    rewrite_llm = MagicMock()
    rewrite_llm.rewrite.return_value = MagicMock(
        text="국민연금공단 차세대 ERP 구축 사업의 예산은 얼마인가요?",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    rewrite_llm.model_name = "gpt-5-mini"

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=metadata_store,
        rewrite_llm=rewrite_llm,
    )

    retriever.retrieve(
        "그 사업 예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 구축 사업 알려줘"}
        ],
    )

    # resolved_query("차세대 ERP 구축...")가 find_relevant_docs에 전달되어야 한다.
    # 원문 "그 사업 예산은?"만 전달되면 project_clues 추출 실패로 부스팅 누락.
    assert any(
        "차세대 ERP" in call for call in metadata_store.find_relevant_docs_calls
    ), (
        "resolved_query가 문서 부스팅에 전달되지 않음. "
        f"실제 호출: {metadata_store.find_relevant_docs_calls}"
    )
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/unit/test_retriever_multiturn.py -v
```

Expected: FAIL — "resolved_query가 문서 부스팅에 전달되지 않음. 실제 호출: ['그 사업 예산은?']" (또는 rewrite 경로가 아직 `llm.rewrite()` 미구현으로 AttributeError).

여기서 AttributeError가 먼저 나면 Task 3-5 완료 후 다시 돌리면 됨. 하지만 이 작업 기준으로는 `rewrite_llm=None`으로 간소화한 버전이 필요하면 임시로 다음 테스트를 먼저 쓴다:

```python
def test_retriever_uses_resolved_query_for_project_clue_augmentation_without_llm() -> None:
    """LLM 없이 규칙 기반 rewrite가 발생하는 케이스로 resolved_query 일관성 검증."""
    metadata_store = _RecordingMetadataStore(
        agency_list=["국민연금공단"],
        relevant_docs=["차세대_ERP.hwp", "차세대_ERP_2.hwp"],
    )
    vector_store = MagicMock()
    vector_store.query.return_value = [_make_chunk()]
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=metadata_store,
        rewrite_llm=None,  # 룰 기반만
        rewrite_mode="rule_only",
    )

    retriever.retrieve(
        "그 사업 예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 구축 사업 알려줘"}
        ],
    )

    # 룰 rewrite가 "그 사업" → "국민연금공단 차세대 ERP 구축 사업"으로 치환.
    # find_relevant_docs는 resolved_query로 호출되어야 project_clues 추출 가능.
    assert any(
        "차세대 ERP" in call for call in metadata_store.find_relevant_docs_calls
    ), (
        "resolved_query가 문서 부스팅에 전달되지 않음. "
        f"실제 호출: {metadata_store.find_relevant_docs_calls}"
    )
```

- [ ] **Step 3: 최소 구현** — `src/bidmate_rag/retrieval/retriever.py` 309-310, 340-341 라인 수정

변경 전:
```python
where = self._augment_where_with_project_docs(query, where)
where = self._augment_where_with_history_docs(query, where, chat_history)
```

변경 후 (4곳 모두):
```python
where = self._augment_where_with_project_docs(resolved_query, where)
where = self._augment_where_with_history_docs(resolved_query, where, chat_history)
```

정확히는:
- [retriever.py:309](../../src/bidmate_rag/retrieval/retriever.py#L309) — `query` → `resolved_query`
- [retriever.py:310](../../src/bidmate_rag/retrieval/retriever.py#L310) — `query` → `resolved_query`
- [retriever.py:340](../../src/bidmate_rag/retrieval/retriever.py#L340) — `query` → `resolved_query`
- [retriever.py:341](../../src/bidmate_rag/retrieval/retriever.py#L341) — `query` → `resolved_query`

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/unit/test_retriever_multiturn.py -v
```

Expected: PASS

- [ ] **Step 5: 전체 회귀 확인**

```bash
uv run pytest tests/unit/ -q
```

Expected: 240+ passed (기존 테스트 무회귀).

- [ ] **Step 6: 커밋**

```bash
git add tests/unit/test_retriever_multiturn.py src/bidmate_rag/retrieval/retriever.py
git commit -m "$(cat <<'EOF'
fix(multiturn): 문서 부스팅 경로에서 resolved_query 일관 사용

rewrite된 쿼리("그 사업" → "차세대 ERP 구축 사업")가 project_clues/history
augment에 전달되지 않던 문제를 수정. 4곳에서 query → resolved_query.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: A3 — Memory state 소유권을 Retriever로 단일화

**배경:** [retriever.py:270-278](../../src/bidmate_rag/retrieval/retriever.py#L270-L278)에서 `rewritten_query=None`으로 memory.build 호출, [chat.py:73-81](../../src/bidmate_rag/pipelines/chat.py#L73-L81)에서 실제 `rewritten_query`로 다시 호출. 두 상태가 서로 다를 수 있고, `enable_multiturn=False, memory=<instance>` 설정 시 chat만 memory를 injecting하는 불일치도 존재.

**Files:**
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Modify: `src/bidmate_rag/pipelines/chat.py`
- Test: `tests/integration/test_chat_pipeline.py` (기존 + 추가)

- [ ] **Step 1: 실패 테스트 작성** — `tests/integration/test_chat_pipeline.py` 말미에 추가

```python
def test_chat_pipeline_reuses_memory_state_from_retriever_debug() -> None:
    """Chat은 retriever가 이미 빌드한 memory_state를 재사용해야 한다."""

    class CountingMemory(ConversationMemory):
        def __init__(self) -> None:
            super().__init__(
                max_recent_turns=4, max_summary_chars=120, agency_list=["국민연금공단"]
            )
            self.build_calls = 0

        def build(self, chat_history, *, current_question=None, rewritten_query=None):
            self.build_calls += 1
            return super().build(
                chat_history,
                current_question=current_question,
                rewritten_query=rewritten_query,
            )

    class RetrieverWithMemoryState:
        _last_debug = {
            "original_query": "평가기준은?",
            "rewritten_query": "국민연금공단 차세대 ERP 사업의 평가기준은?",
            "rewrite_applied": True,
            "rewrite_reason": "rule_fallback",
            "rewrite_prompt_tokens": 0,
            "rewrite_completion_tokens": 0,
            "rewrite_total_tokens": 0,
            "rewrite_cost_usd": 0.0,
            "retrieved_chunks_before_rerank": [],
            "retrieved_chunks_after_rerank": [],
            "memory_state": {
                "recent_turns": [
                    {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}
                ],
                "summary_buffer": "테스트 요약",
                "slot_memory": {"발주기관": "국민연금공단", "사업명": "차세대 ERP 사업"},
            },
        }

        def retrieve(self, query, chat_history=None, top_k=5, metadata_filter=None):
            chunk = Chunk(
                chunk_id="c-1",
                doc_id="d-1",
                text="t",
                text_with_meta="t",
                char_count=1,
                section="요구사항",
                content_type="text",
                chunk_index=0,
                metadata={"파일명": "a.hwp"},
            )
            return [RetrievedChunk(rank=1, score=0.9, chunk=chunk)]

    memory = CountingMemory()
    pipeline = RAGChatPipeline(
        retriever=RetrieverWithMemoryState(), llm=FakeLLM(), memory=memory
    )

    result = pipeline.answer(
        "평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    # Chat은 memory를 다시 빌드하지 않고 retriever의 _last_debug에서 재사용한다.
    assert memory.build_calls == 0, (
        f"Chat이 memory.build를 {memory.build_calls}회 호출. "
        "retriever의 _last_debug['memory_state']를 재사용해야 한다."
    )
    # Retriever가 제공한 슬롯이 그대로 debug에 노출되어야 한다.
    assert result.debug["memory_slots"] == {
        "발주기관": "국민연금공단",
        "사업명": "차세대 ERP 사업",
    }
    assert result.debug["memory_summary"] == "테스트 요약"
```

- [ ] **Step 2: 실패 확인**

```bash
uv run pytest tests/integration/test_chat_pipeline.py::test_chat_pipeline_reuses_memory_state_from_retriever_debug -v
```

Expected: FAIL — `memory.build_calls == 1` (현재는 chat이 항상 memory.build 호출).

- [ ] **Step 3: Retriever 쪽 수정** — [retriever.py:260-429](../../src/bidmate_rag/retrieval/retriever.py#L260-L429)

현재 `retrieve()` 메서드에서 rewrite 전 memory.build 호출 후 `rewrite_memory_state`만 사용. rewrite 후에도 다시 build해서 `generation_memory_state`를 만들고, `_last_debug["memory_state"]`에 저장해야 함.

`retrieve` 메서드 내부에서, 기존 `rewrite_memory_state = self.memory.build(...)` 블록 바로 뒤에 rewrite 결과를 받은 후 다음 로직 추가 (구체적 삽입 위치: `resolved_query, rewrite_trace = ...` 바로 다음):

```python
        # Generation용 memory state는 rewritten_query를 반영해 한 번 더 빌드.
        # rewrite LLM이 본 slot_memory와 generation LLM이 보는 slot_memory를
        # 소스 단일화 — chat 파이프라인이 이 state만 재사용한다.
        generation_memory_state = (
            self.memory.build(
                chat_history or [],
                current_question=query,
                rewritten_query=resolved_query,
            )
            if self.enable_multiturn and self.memory is not None
            else None
        )
```

그리고 `_last_debug` 할당부 ([retriever.py:418-426](../../src/bidmate_rag/retrieval/retriever.py#L418-L426))에 추가:

```python
        if self.debug_trace_enabled:
            self._last_debug = {
                **rewrite_trace,
                "rewrite_slot_memory": rewrite_slot_memory,
                "where": where,
                "where_document": where_document,
                "retrieved_chunks_before_rerank": self._serialize_results(before_rerank[:final_top_k]),
                "retrieved_chunks_after_rerank": self._serialize_results(final_results),
                "memory_state": generation_memory_state,  # ← 신규
            }
```

- [ ] **Step 4: Chat 쪽 수정** — [chat.py:66-102](../../src/bidmate_rag/pipelines/chat.py#L66-L102)

`retriever.retrieve()` 호출 뒤 memory.build 호출부를 debug 소비로 대체. 변경 전:

```python
        retrieval_debug = getattr(self.retriever, "_last_debug", {}) or {}
        memory_state = (
            self.memory.build(
                chat_history or [],
                current_question=question,
                rewritten_query=retrieval_debug.get("rewritten_query", question),
            )
            if self.memory is not None
            else {"recent_turns": [], "summary_buffer": "", "slot_memory": {}}
        )
```

변경 후:

```python
        retrieval_debug = getattr(self.retriever, "_last_debug", {}) or {}
        # Retriever가 generation용 memory state를 소유한다 — 있으면 재사용.
        # 없는 경우(레거시 retriever, memory 비활성화)만 폴백으로 빌드.
        memory_state = retrieval_debug.get("memory_state")
        if memory_state is None:
            memory_state = (
                self.memory.build(
                    chat_history or [],
                    current_question=question,
                    rewritten_query=retrieval_debug.get("rewritten_query", question),
                )
                if self.memory is not None
                else {"recent_turns": [], "summary_buffer": "", "slot_memory": {}}
            )
```

- [ ] **Step 5: 테스트 통과 + 회귀 확인**

```bash
uv run pytest tests/integration/test_chat_pipeline.py -v tests/unit/ -q
```

Expected: 새 테스트 PASS + 기존 `test_chat_pipeline_includes_memory_debug` PASS (폴백 경로 동작).

- [ ] **Step 6: 커밋**

```bash
git add src/bidmate_rag/retrieval/retriever.py src/bidmate_rag/pipelines/chat.py tests/integration/test_chat_pipeline.py
git commit -m "$(cat <<'EOF'
fix(multiturn): memory state 소유권을 Retriever로 단일화

rewrite 전후 두 번 빌드되던 memory를 retriever 내부에서 모두 처리하고
_last_debug['memory_state']로 chat에 전달. Chat은 해당 값을 재사용하고
없는 경우에만 폴백 빌드. enable_multiturn 플래그 일관성도 해소.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: A2a — `BaseLLMProvider.rewrite()` 인터페이스 + OpenAI 구현

**배경:** [multiturn.py:173](../../src/bidmate_rag/retrieval/multiturn.py#L173)에서 `llm.client.chat.completions.create` 직접 호출. `HFLocalLLM`에는 `.client` 속성이 없어 AttributeError 크래시. BaseLLMProvider 추상화를 우회하는 설계 결함.

**Files:**
- Modify: `src/bidmate_rag/providers/llm/base.py`
- Modify: `src/bidmate_rag/providers/llm/openai_compat.py`
- Test: `tests/unit/test_provider_rewrite.py` (신규)

- [ ] **Step 1: 실패 테스트 작성** — `tests/unit/test_provider_rewrite.py`

```python
"""LLM 프로바이더의 rewrite 인터페이스 단위 테스트."""

from unittest.mock import MagicMock

import pytest

from bidmate_rag.providers.llm.base import BaseLLMProvider, RewriteResponse
from bidmate_rag.providers.llm.openai_compat import OpenAICompatibleLLM


def _make_openai_client(response_text: str, prompt_tokens: int = 50,
                        completion_tokens: int = 10) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = response_text
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    client.chat.completions.create.return_value = response
    return client


def test_base_provider_rewrite_raises_by_default() -> None:
    """기본 구현은 NotImplementedError — 서브클래스가 override해야 한다."""

    class BareProvider(BaseLLMProvider):
        provider_name = "bare"
        model_name = "bare-model"

        def generate(self, *args, **kwargs):
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        BareProvider().rewrite("test prompt")


def test_openai_provider_rewrite_returns_structured_response() -> None:
    client = _make_openai_client(
        response_text="국민연금공단 차세대 ERP 사업의 예산은 얼마인가요?",
        prompt_tokens=250,
        completion_tokens=40,
    )
    provider = OpenAICompatibleLLM(
        provider_name="openai", model_name="gpt-5-mini", client=client
    )

    response = provider.rewrite("재작성 프롬프트", max_tokens=16000, timeout=30)

    assert isinstance(response, RewriteResponse)
    assert response.text == "국민연금공단 차세대 ERP 사업의 예산은 얼마인가요?"
    assert response.prompt_tokens == 250
    assert response.completion_tokens == 40
    assert response.total_tokens == 290

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-5-mini"
    assert create_kwargs["max_completion_tokens"] == 16000
    assert create_kwargs["timeout"] == 30
    assert create_kwargs["messages"] == [
        {"role": "user", "content": "재작성 프롬프트"}
    ]


def test_openai_provider_rewrite_handles_empty_content() -> None:
    client = _make_openai_client(response_text="")
    provider = OpenAICompatibleLLM(
        provider_name="openai", model_name="gpt-5-mini", client=client
    )

    response = provider.rewrite("test", max_tokens=1000)
    assert response.text == ""
```

- [ ] **Step 2: 실패 확인**

```bash
uv run pytest tests/unit/test_provider_rewrite.py -v
```

Expected: FAIL — `ImportError: cannot import name 'RewriteResponse'` 또는 `AttributeError: ... has no attribute 'rewrite'`.

- [ ] **Step 3: Base 인터페이스 정의** — [base.py](../../src/bidmate_rag/providers/llm/base.py) 수정

`StreamDelta` 아래에 `RewriteResponse` 추가, `BaseLLMProvider`에 `rewrite` 기본 구현 추가.

변경 전 ([base.py:12-17](../../src/bidmate_rag/providers/llm/base.py#L12-L17)):
```python
@dataclass
class StreamDelta:
    """스트리밍 중 증분 토큰. `generate_stream`이 토큰 단위로 방출한다."""

    text: str
```

변경 후:
```python
@dataclass
class StreamDelta:
    """스트리밍 중 증분 토큰. `generate_stream`이 토큰 단위로 방출한다."""

    text: str


@dataclass
class RewriteResponse:
    """멀티턴 쿼리 재작성 응답 — provider-agnostic."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

`BaseLLMProvider` 클래스 본문 끝에 다음 메서드 추가 ([base.py:64](../../src/bidmate_rag/providers/llm/base.py#L64) 이후):

```python
    def rewrite(
        self,
        prompt: str,
        *,
        max_tokens: int = 16000,
        timeout: int | None = 30,
    ) -> RewriteResponse:
        """짧은 지시형 프롬프트에 대한 원샷 텍스트 생성.

        멀티턴 query rewrite처럼 retrieval context 없이 LLM에게 단문 생성을
        요청하는 경로에서 사용한다. 기본 구현은 미지원으로 예외를 던진다 —
        구현하지 않은 프로바이더를 rewrite_llm으로 주입하면 즉시 실패해
        룰 기반 폴백으로 빠지도록 하기 위함.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rewrite()"
        )
```

- [ ] **Step 4: OpenAI 구현** — [openai_compat.py](../../src/bidmate_rag/providers/llm/openai_compat.py) 수정

import 블록에 `RewriteResponse` 추가:

```python
from bidmate_rag.providers.llm.base import BaseLLMProvider, RewriteResponse, StreamDelta
```

클래스 본문 `generate_stream` 메서드 뒤에 추가:

```python
    def rewrite(
        self,
        prompt: str,
        *,
        max_tokens: int = 16000,
        timeout: int | None = 30,
    ) -> RewriteResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            timeout=timeout,
        )
        usage = getattr(response, "usage", None)
        return RewriteResponse(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
uv run pytest tests/unit/test_provider_rewrite.py -v
```

Expected: 3 passed.

- [ ] **Step 6: 커밋**

```bash
git add src/bidmate_rag/providers/llm/base.py src/bidmate_rag/providers/llm/openai_compat.py tests/unit/test_provider_rewrite.py
git commit -m "$(cat <<'EOF'
feat(llm): BaseLLMProvider.rewrite() 인터페이스 + OpenAI 구현

rewrite가 provider별 client API를 직접 호출하던 결합을 해소.
RewriteResponse 데이터클래스로 (text, prompt/completion/total_tokens) 반환.
기본 구현은 NotImplementedError — 지원 안 하는 provider는 룰 폴백으로.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: A2b — `HFLocalLLM.rewrite()` 구현

**Files:**
- Modify: `src/bidmate_rag/providers/llm/hf_local.py`
- Test: `tests/unit/test_provider_rewrite.py` (기존 파일에 추가)

- [ ] **Step 1: 실패 테스트 추가** — `tests/unit/test_provider_rewrite.py` 말미에

```python
def test_hf_local_provider_rewrite_uses_local_generator() -> None:
    from bidmate_rag.providers.llm.hf_local import HFLocalLLM

    class _FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

    class _FakeGenerator:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()
            self.last_call: dict | None = None

        def __call__(self, prompt: str, **kwargs) -> list[dict]:
            self.last_call = {"prompt": prompt, **kwargs}
            return [{"generated_text": "재작성된 쿼리"}]

    generator = _FakeGenerator()
    provider = HFLocalLLM(
        model_name="hf-test", provider_name="huggingface", generator=generator
    )

    response = provider.rewrite("재작성 프롬프트", max_tokens=256)

    assert response.text == "재작성된 쿼리"
    assert response.prompt_tokens == len("재작성 프롬프트")
    assert response.completion_tokens == len("재작성된 쿼리")
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert generator.last_call["max_new_tokens"] == 256
    assert generator.last_call["do_sample"] is False
    assert generator.last_call["return_full_text"] is False
```

- [ ] **Step 2: 실패 확인**

```bash
uv run pytest tests/unit/test_provider_rewrite.py::test_hf_local_provider_rewrite_uses_local_generator -v
```

Expected: FAIL — `NotImplementedError: HFLocalLLM does not implement rewrite()`.

- [ ] **Step 3: 구현** — [hf_local.py](../../src/bidmate_rag/providers/llm/hf_local.py) 수정

import 블록에 `RewriteResponse` 추가:

```python
from bidmate_rag.providers.llm.base import BaseLLMProvider, RewriteResponse
```

클래스 끝에 메서드 추가 ([hf_local.py:123](../../src/bidmate_rag/providers/llm/hf_local.py#L123) 이후):

```python
    def rewrite(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        timeout: int | None = None,
    ) -> RewriteResponse:
        """로컬 HF pipeline으로 짧은 텍스트 생성.

        timeout은 무시된다 — transformers pipeline은 동기 실행이라 강제 종료 불가.
        기본 max_tokens가 OpenAI 대비 낮은 이유: 로컬 모델은 reasoning 토큰이
        없으므로 256 이면 한 줄 재작성에 충분.
        """
        generator = self._get_generator()
        tokenizer = generator.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt))
        response = generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = response[0]["generated_text"].strip() if response else ""
        completion_tokens = len(tokenizer.encode(text)) if text else 0
        return RewriteResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/unit/test_provider_rewrite.py -v
```

Expected: 4 passed (기존 3 + 신규 1).

- [ ] **Step 5: 커밋**

```bash
git add src/bidmate_rag/providers/llm/hf_local.py tests/unit/test_provider_rewrite.py
git commit -m "$(cat <<'EOF'
feat(llm): HFLocalLLM.rewrite() 구현 — 로컬 generator 재사용

transformers pipeline으로 짧은 지시형 텍스트 생성. HF 로컬 모델 사용자가
멀티턴 rewrite 경로에서 AttributeError로 크래시하던 문제 해소.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: A2c — `multiturn._llm_rewrite`를 `llm.rewrite()` 호출로 전환

**배경:** [multiturn.py:153-211](../../src/bidmate_rag/retrieval/multiturn.py#L153-L211)에서 `llm.client.chat.completions.create`를 직접 호출하는 것을 `llm.rewrite(prompt, max_tokens=..., timeout=...)`로 교체. 테스트 mock도 `client.chat.completions`가 아닌 `rewrite` 메서드를 mocking.

**Files:**
- Modify: `src/bidmate_rag/retrieval/multiturn.py`
- Modify: `tests/unit/test_multiturn.py`

- [ ] **Step 1: 기존 mock 헬퍼 업데이트** — `tests/unit/test_multiturn.py`

변경 전 ([test_multiturn.py:9-17](../../tests/unit/test_multiturn.py#L9-L17)):

```python
def _make_mock_llm(rewritten_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = rewritten_text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_llm.client.chat.completions.create.return_value = mock_response
    mock_llm.model_name = "gpt-5-mini"
    return mock_llm
```

변경 후:

```python
def _make_mock_llm(
    rewritten_text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> MagicMock:
    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=rewritten_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    mock_llm.model_name = "gpt-5-mini"
    return mock_llm
```

그리고 기존 테스트의 assertion을 업데이트:

[test_multiturn.py:45](../../tests/unit/test_multiturn.py#L45) —
변경 전: `mock_llm.client.chat.completions.create.assert_called_once()`
변경 후: `mock_llm.rewrite.assert_called_once()`

[test_multiturn.py:68](../../tests/unit/test_multiturn.py#L68) —
변경 전: `prompt = mock_llm.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]`
변경 후: `prompt = mock_llm.rewrite.call_args.args[0]`

[test_multiturn.py:86](../../tests/unit/test_multiturn.py#L86) —
변경 전: `mock_llm.client.chat.completions.create.assert_not_called()`
변경 후: `mock_llm.rewrite.assert_not_called()`

[test_multiturn.py:90-93](../../tests/unit/test_multiturn.py#L90-L93) — 에러 케이스:
변경 전:
```python
mock_llm = MagicMock()
mock_llm.model_name = "gpt-5-mini"
mock_llm.client.chat.completions.create.side_effect = Exception("timeout")
```
변경 후:
```python
mock_llm = MagicMock()
mock_llm.model_name = "gpt-5-mini"
mock_llm.rewrite.side_effect = Exception("timeout")
```

- [ ] **Step 2: multiturn 구현 교체** — [multiturn.py:172-195](../../src/bidmate_rag/retrieval/multiturn.py#L172-L195)

변경 전:

```python
    try:
        response = llm.client.chat.completions.create(
            model=llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
            timeout=5,
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        rewritten = (response.choices[0].message.content or "").strip()
        rewritten = _WHITESPACE_PATTERN.sub(" ", rewritten).strip()
```

변경 후:

```python
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
```

그리고 `_llm_rewrite` 함수 시그니처를 확장:

변경 전 ([multiturn.py:153-159](../../src/bidmate_rag/retrieval/multiturn.py#L153-L159)):

```python
def _llm_rewrite(
    query: str,
    chat_history: list[dict] | None,
    llm: object,
    *,
    slot_memory: dict[str, str] | None = None,
) -> tuple[str, dict[str, object]]:
```

변경 후:

```python
def _llm_rewrite(
    query: str,
    chat_history: list[dict] | None,
    llm: object,
    *,
    slot_memory: dict[str, str] | None = None,
    max_completion_tokens: int = 16000,
    timeout_seconds: int = 30,
) -> tuple[str, dict[str, object]]:
```

그리고 `rewrite_query_with_history` 함수도 동일 파라미터 수용:

변경 전 ([multiturn.py:243-250](../../src/bidmate_rag/retrieval/multiturn.py#L243-L250)):

```python
def rewrite_query_with_history(
    query: str,
    chat_history: list[dict] | None,
    agency_list: list[str],
    llm: object | None = None,
    mode: str = "llm_with_rule_fallback",
    slot_memory: dict[str, str] | None = None,
) -> tuple[str, dict[str, object]]:
```

변경 후:

```python
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
```

`_llm_rewrite` 호출부도 새 파라미터 전달 ([multiturn.py:268-273](../../src/bidmate_rag/retrieval/multiturn.py#L268-L273)):

변경 전:
```python
    llm_rewritten, llm_trace = _llm_rewrite(
        query,
        chat_history,
        llm,
        slot_memory=slot_memory,
    )
```

변경 후:
```python
    llm_rewritten, llm_trace = _llm_rewrite(
        query,
        chat_history,
        llm,
        slot_memory=slot_memory,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=timeout_seconds,
    )
```

- [ ] **Step 3: 실패/통과 확인**

```bash
uv run pytest tests/unit/test_multiturn.py -v
```

Expected: 6 passed (모든 기존 테스트가 새 인터페이스로 동작).

- [ ] **Step 4: 회귀 확인**

```bash
uv run pytest tests/unit/ -q
```

Expected: 240+ passed.

- [ ] **Step 5: 커밋**

```bash
git add src/bidmate_rag/retrieval/multiturn.py tests/unit/test_multiturn.py
git commit -m "$(cat <<'EOF'
refactor(multiturn): rewrite를 BaseLLMProvider.rewrite() 추상화로 전환

multiturn._llm_rewrite가 llm.client 직접 호출 대신 llm.rewrite(prompt,
max_tokens, timeout)을 사용. HF 로컬/OpenAI 양쪽에서 동일 경로. rewrite
토큰 예산과 타임아웃은 함수 파라미터로 외부 주입.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: A1 — `RewriteConfig` 토큰/타임아웃 필드 + 배선

**배경:** [multiturn.py:176](../../src/bidmate_rag/retrieval/multiturn.py#L176)의 `max_completion_tokens=200`, `timeout=5` 하드코딩. CLAUDE.md는 gpt-5 reasoning 토큰 포함 "16000+" 설정 권고. Config로 올려서 실제 운영 환경에 맞게 조정 가능하게.

**Files:**
- Modify: `src/bidmate_rag/config/settings.py`
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Modify: `configs/retrieval.yaml`
- Test: `tests/unit/test_multiturn.py` (추가)

- [ ] **Step 1: 실패 테스트 추가** — `tests/unit/test_multiturn.py` 말미에

```python
def test_rewrite_query_with_history_passes_config_to_provider() -> None:
    mock_llm = _make_mock_llm("재작성 결과")

    rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
        max_completion_tokens=8000,
        timeout_seconds=60,
    )

    call_kwargs = mock_llm.rewrite.call_args.kwargs
    assert call_kwargs["max_tokens"] == 8000
    assert call_kwargs["timeout"] == 60
```

- [ ] **Step 2: 실패 확인**

```bash
uv run pytest tests/unit/test_multiturn.py::test_rewrite_query_with_history_passes_config_to_provider -v
```

Task 5에서 이미 `max_completion_tokens`, `timeout_seconds` 파라미터를 추가했으므로 **이 테스트는 바로 통과할 수도 있음.** 통과하면 Step 3-4 건너뛰고 Step 5(Config 노출)로 직행.

만약 실패한다면 `rewrite_query_with_history`에서 `_llm_rewrite` 호출 시 새 파라미터 미전달 — Task 5 Step 2 검토.

- [ ] **Step 3: `RewriteConfig`에 필드 추가** — [settings.py:38-41](../../src/bidmate_rag/config/settings.py#L38-L41)

변경 전:
```python
class RewriteConfig(BaseModel):
    """멀티턴 쿼리 재작성 설정."""

    mode: str = "llm_with_rule_fallback"
```

변경 후:
```python
class RewriteConfig(BaseModel):
    """멀티턴 쿼리 재작성 설정."""

    mode: str = "llm_with_rule_fallback"
    max_completion_tokens: int = 16000
    timeout_seconds: int = 30
```

- [ ] **Step 4: `RAGRetriever`에 파라미터 추가** — [retriever.py:38-66](../../src/bidmate_rag/retrieval/retriever.py#L38-L66)

`__init__` 시그니처 확장 (기존 `rewrite_mode: str = "llm_with_rule_fallback"` 뒤에 추가):

변경 전:
```python
    def __init__(
        self,
        vector_store,
        embedder,
        metadata_store=None,
        sparse_store=None,
        reranker_model=None,
        enable_multiturn: bool = True,
        boost_config: dict | None = None,
        hybrid_config: dict | None = None,
        rewrite_llm=None,
        rewrite_mode: str = "llm_with_rule_fallback",
        memory=None,
        debug_trace_enabled: bool = True,
    ) -> None:
```

변경 후:
```python
    def __init__(
        self,
        vector_store,
        embedder,
        metadata_store=None,
        sparse_store=None,
        reranker_model=None,
        enable_multiturn: bool = True,
        boost_config: dict | None = None,
        hybrid_config: dict | None = None,
        rewrite_llm=None,
        rewrite_mode: str = "llm_with_rule_fallback",
        rewrite_max_completion_tokens: int = 16000,
        rewrite_timeout_seconds: int = 30,
        memory=None,
        debug_trace_enabled: bool = True,
    ) -> None:
```

그리고 `__init__` 본문에 저장:

기존 `self.rewrite_mode = rewrite_mode` 바로 아래 ([retriever.py:63](../../src/bidmate_rag/retrieval/retriever.py#L63)) 추가:
```python
        self.rewrite_max_completion_tokens = rewrite_max_completion_tokens
        self.rewrite_timeout_seconds = rewrite_timeout_seconds
```

`retrieve()` 내부의 `rewrite_query_with_history` 호출에 새 파라미터 전달 ([retriever.py:280-289](../../src/bidmate_rag/retrieval/retriever.py#L280-L289)):

변경 전:
```python
        resolved_query, rewrite_trace = (
            rewrite_query_with_history(
                query,
                chat_history,
                agency_list,
                llm=self.rewrite_llm,
                mode=self.rewrite_mode,
                slot_memory=rewrite_slot_memory,
            )
            if self.enable_multiturn
            else ...
        )
```

변경 후:
```python
        resolved_query, rewrite_trace = (
            rewrite_query_with_history(
                query,
                chat_history,
                agency_list,
                llm=self.rewrite_llm,
                mode=self.rewrite_mode,
                slot_memory=rewrite_slot_memory,
                max_completion_tokens=self.rewrite_max_completion_tokens,
                timeout_seconds=self.rewrite_timeout_seconds,
            )
            if self.enable_multiturn
            else ...
        )
```

- [ ] **Step 5: `configs/retrieval.yaml` 업데이트**

변경 전 ([retrieval.yaml:22-23](../../configs/retrieval.yaml#L22-L23)):
```yaml
rewrite:
  mode: llm_with_rule_fallback
```

변경 후:
```yaml
rewrite:
  mode: llm_with_rule_fallback
  # gpt-5 계열은 reasoning 토큰이 completion에 포함되므로 넉넉하게 (CLAUDE.md 참조).
  # 로컬 HF 모델은 HFLocalLLM.rewrite()가 내부적으로 256 제한을 사용.
  max_completion_tokens: 16000
  timeout_seconds: 30
```

- [ ] **Step 6: Runtime 배선 확인**

`src/bidmate_rag/pipelines/runtime.py`에서 `RewriteConfig` → `RAGRetriever` 생성자 전달 경로를 확인하고, 새 필드가 빠지지 않았는지 체크. 필요 시 다음 패턴 적용:

```bash
uv run python -c "
from bidmate_rag.config.settings import RetrievalConfig
import yaml
with open('configs/retrieval.yaml') as f:
    cfg = RetrievalConfig(**yaml.safe_load(f))
print('rewrite.max_completion_tokens:', cfg.rewrite.max_completion_tokens)
print('rewrite.timeout_seconds:', cfg.rewrite.timeout_seconds)
"
```

Expected output:
```
rewrite.max_completion_tokens: 16000
rewrite.timeout_seconds: 30
```

Runtime 빌더에서 `rewrite_max_completion_tokens=retrieval_config.rewrite.max_completion_tokens`, `rewrite_timeout_seconds=retrieval_config.rewrite.timeout_seconds` 로 전달하도록 수정. 실제 수정 위치는 `runtime.py` 내 `RAGRetriever(...)` 생성부 (grep으로 찾기).

```bash
grep -n "RAGRetriever(" src/bidmate_rag/pipelines/runtime.py
```

찾은 위치에서 기존 `rewrite_mode=...` 파라미터 옆에 두 줄 추가.

- [ ] **Step 7: 전체 회귀 테스트**

```bash
uv run pytest tests/ -q
```

Expected: 전체 통과. 기존 240+ + 신규 6개 (Task 1 2개 + Task 2 1개 + Task 3 3개 + Task 4 1개 + Task 6 1개 ≈ 8개 신규).

- [ ] **Step 8: 커밋**

```bash
git add src/bidmate_rag/config/settings.py src/bidmate_rag/retrieval/retriever.py src/bidmate_rag/pipelines/runtime.py configs/retrieval.yaml tests/unit/test_multiturn.py
git commit -m "$(cat <<'EOF'
fix(multiturn): rewrite 토큰/타임아웃을 RewriteConfig로 노출

max_completion_tokens=16000, timeout_seconds=30을 기본값으로. gpt-5 계열의
reasoning 토큰을 수용하지 못하던 기존 200 하드코딩을 수정.
runtime에서 configs/retrieval.yaml의 값을 RAGRetriever에 전달.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 최종 검증 + 문서 업데이트 + 푸시

- [ ] **Step 1: 전체 테스트 스위트**

```bash
uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: 240+ passed, 0 failed.

- [ ] **Step 2: 수동 smoke 테스트 (선택, API 키 있을 때만)**

```bash
uv run python -c "
import os
if not os.getenv('OPENAI_API_KEY'):
    print('API 키 없음 — skip')
    raise SystemExit
from bidmate_rag.providers.llm.openai_compat import OpenAICompatibleLLM
llm = OpenAICompatibleLLM(provider_name='openai', model_name='gpt-5-mini')
resp = llm.rewrite('Rewrite: \"그 사업\" → 독립 질문으로. 맥락: 국민연금공단 차세대 ERP. 답만 출력.', max_tokens=16000, timeout=30)
print('text:', resp.text[:120])
print('tokens:', resp.prompt_tokens, '+', resp.completion_tokens, '=', resp.total_tokens)
"
```

Expected: non-empty `text`, non-zero `completion_tokens`. 이전 코드였다면 `text=""`로 조용히 실패했을 것.

- [ ] **Step 3: docs/architecture.md 업데이트 (해당 섹션만)**

`docs/architecture.md`에서 rewrite 경로 설명이 있으면, `llm.client` 직접 호출 대신 `BaseLLMProvider.rewrite()` 추상화를 타는 방식으로 수정. 없으면 skip.

```bash
grep -n "llm.client\|rewrite" docs/architecture.md | head
```

- [ ] **Step 4: 변경사항 요약 커밋 (선택 — docs 수정이 있을 때만)**

```bash
git add docs/architecture.md
git commit -m "$(cat <<'EOF'
docs: rewrite 경로 BaseLLMProvider.rewrite() 추상화 반영

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: 최종 커밋 확인 및 푸시 대기**

```bash
git log --oneline -8
```

Expected: `24c8d58` 위에 이번 작업의 6개 커밋(Task 1-6) + 선택적 docs 커밋이 쌓여 있음.

```bash
git status --short
```

Expected: 작업 관련 변경사항은 모두 커밋됨. (이전 브랜치에서 넘어온 미관련 docs 삭제/미추적 파일만 남아 있음.)

- [ ] **Step 6: 사용자 승인 후 푸시**

**사용자에게 확인 후** 실행:

```bash
git push -u origin fix/am/multiturn-prompt-recover
```

PR 대상은 `feat/yj`가 아니라 **`develop`으로 내는 것을 권장** — 이번 브랜치가 `origin/feat/yj` 위에 쌓였으므로 PR에 팀원의 `a2e9e33` + 이번 수정이 함께 들어가 멀티턴 기능이 실제로 동작하는 상태로 develop에 반영된다.

```bash
gh pr create --base develop --title "fix(multiturn): rewrite 토큰/추상화/메모리 일관성 복구" --body "$(cat <<'EOF'
## Summary
- 팀원의 `feat(multiturn)` 위에 운영 블로커 4건 수정
- LLM rewrite 토큰 기근 (`max_completion_tokens=200` → config 노출, 기본 16000)
- Provider 추상화 우회 해소 (`BaseLLMProvider.rewrite()` 도입, OpenAI/HF 각각 구현)
- Memory state 소유권 Retriever로 단일화 (pre/post-rewrite 일관성)
- Retriever 내부 `resolved_query` 일관 사용 (문서 부스팅 경로에 rewrite 반영)

## Test plan
- [x] `uv run pytest tests/ -q` 전체 그린
- [x] 신규 테스트: `test_retriever_multiturn.py`, `test_provider_rewrite.py`, `test_chat_pipeline_reuses_memory_state_from_retriever_debug`
- [ ] 수동 smoke: gpt-5-mini rewrite 비어있지 않은 응답 확인
- [ ] HF local rewrite AttributeError 재현 안 됨 확인

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Critical A1 (토큰 기근) → Task 6
- ✅ Critical A2 (provider 추상화) → Task 3, 4, 5
- ✅ Critical A3 (메모리 이중 빌드) → Task 2
- ✅ High B (enable_multiturn 불일치) → Task 2 (chat.py가 memory_state 없으면 skip)
- ✅ High C (resolved_query 혼용) → Task 1

**Placeholder scan:** 전 섹션에 구체 코드/경로/명령어가 있음. TBD/TODO 없음.

**Type consistency:**
- `RewriteResponse(text, prompt_tokens, completion_tokens, total_tokens)` — Task 3/4/5 전부 동일
- `rewrite(prompt, *, max_tokens, timeout)` — Task 3(base/OpenAI default 16000), Task 4(HF default 256) — 기본값은 다르지만 인터페이스 동일 ✓
- `rewrite_query_with_history(..., max_completion_tokens, timeout_seconds)` — Task 5에서 도입, Task 6에서 retriever가 전달 ✓
- `RAGRetriever(rewrite_max_completion_tokens, rewrite_timeout_seconds)` — Task 6에서 도입, runtime에서 전달 ✓
- `_last_debug["memory_state"]` — Task 2에서 retriever가 세팅, chat이 소비 ✓

모든 signatures와 속성명이 task 간 일관.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-multiturn-fixes.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Task 1~7을 각각 fresh subagent에게 디스패치, 태스크 사이에 리뷰 체크포인트. Critical 수정 4건이 서로 독립이라 병렬성도 확보 가능.

**2. Inline Execution** — 현재 세션에서 순차 실행, Task 3 이후 커밋 지점마다 체크포인트. 컨텍스트 연속성 유지.

**어느 방식으로 진행할까요?**
