# 멀티턴 메모리형 하이브리드 검색 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 추가 YAML 파일 없이, `LLM Query Rewriting -> Hybrid Retrieval -> 선택적 Reranking -> Summary Buffer Memory + Slot Memory -> LLM 응답 생성` 구조를 CLI에서 먼저 검증 가능하게 만든다.

**Architecture:** 기존 `retriever -> chat pipeline -> LLM provider` 흐름은 유지하되, 검색 전에는 쿼리 재작성, 검색 후에는 메모리 계층을 추가한다. 설정은 기존 `configs/retrieval.yaml`만 확장하고, CLI에서 재작성 결과·메모리 상태·비용 기록을 직접 출력해서 멀티턴 동작을 눈으로 확인할 수 있게 한다.

**Tech Stack:** Python 3.12, OpenAI API, ChromaDB, BM25 sparse retrieval, pytest

---

## 핵심 제약

- 새 멀티턴 전용 YAML 파일은 만들지 않는다.
- `configs/retrieval.yaml`만 사용하고 필요한 항목만 추가한다.
- 1차 구현 확인 경로는 CLI다.
- Streamlit / 웹 연결은 CLI 검증 후 진행한다.
- 멀티턴 평가는 나중에 붙이되, 지금은 trace와 cost를 남겨서 이후 확장 가능하게 만든다.

## 현재 기준에서 꼭 보완할 점

1. 지금 구조는 재작성 결과를 CLI에서 직접 못 본다.
2. Summary Buffer Memory와 Slot Memory가 아직 없다.
3. 멀티턴 추가 비용이 질의 단위로 분리되어 보이지 않는다.
4. 현재 `history`를 넣어도 “정말 멀티턴이 동작했는지”를 확실히 확인하기 어렵다.

이번 계획은 이 네 가지를 우선 해결한다.

## 파일 구조

### 수정할 파일

- `configs/retrieval.yaml`
- `scripts/run_rag.py`
- `src/bidmate_rag/retrieval/multiturn.py`
- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/pipelines/runtime.py`
- `src/bidmate_rag/pipelines/chat.py`
- `src/bidmate_rag/providers/llm/openai_compat.py`

### 새로 만들 파일

- `src/bidmate_rag/retrieval/memory.py`
- `tests/unit/test_memory.py`
- `tests/unit/test_run_rag.py`

### 보강할 테스트 파일

- `tests/unit/test_multiturn.py`
- `tests/unit/test_retriever.py`
- `tests/unit/test_runtime_pipeline.py`

---

### Task 1: 기존 retrieval 설정 파일만으로 멀티턴/메모리 설정 확장

**Files:**
- Modify: `configs/retrieval.yaml`
- Test: `tests/unit/test_runtime_pipeline.py`

- [ ] **Step 1: 현재 retrieval 설정 구조를 확인하는 테스트를 먼저 추가**

```python
def test_runtime_accepts_multiturn_memory_settings(tmp_path):
    config_text = """
reranker_model: null
boost:
  section: 0.12
  table: 0.08
  max_total: 0.15
hybrid:
  enabled: true
  dense_pool_multiplier: 3
  sparse_pool_multiplier: 3
  rrf_k: 60
enable_multiturn: true
rewrite:
  mode: llm_with_rule_fallback
memory:
  enabled: true
  summary_buffer:
    max_recent_turns: 4
    max_summary_tokens: 256
  slot_memory:
    enabled: true
debug_trace:
  enabled: true
"""
    path = tmp_path / "retrieval.yaml"
    path.write_text(config_text, encoding="utf-8")
    runtime = load_runtime_config(retrieval_config_path=path)

    assert runtime.retrieval.enable_multiturn is True
    assert runtime.retrieval.rewrite.mode == "llm_with_rule_fallback"
    assert runtime.retrieval.memory.enabled is True
    assert runtime.retrieval.memory.summary_buffer.max_recent_turns == 4
    assert runtime.retrieval.memory.slot_memory.enabled is True
    assert runtime.retrieval.debug_trace.enabled is True
```

- [ ] **Step 2: 기존 `configs/retrieval.yaml`에 필요한 항목만 추가**

추가할 항목 예시:

```yaml
reranker_model: null

boost:
  section: 0.12
  table: 0.08
  max_total: 0.15

hybrid:
  enabled: true
  dense_pool_multiplier: 3
  sparse_pool_multiplier: 3
  rrf_k: 60

enable_multiturn: false

rewrite:
  mode: llm_with_rule_fallback

memory:
  enabled: true
  summary_buffer:
    max_recent_turns: 4
    max_summary_tokens: 256
  slot_memory:
    enabled: true

debug_trace:
  enabled: true
```

- [ ] **Step 3: runtime 설정 모델이 새 필드를 읽도록 최소 수정**

```python
class RewriteSettings(BaseModel):
    mode: Literal["rule_only", "llm_only", "llm_with_rule_fallback"] = "llm_with_rule_fallback"


class SummaryBufferSettings(BaseModel):
    max_recent_turns: int = 4
    max_summary_tokens: int = 256


class SlotMemorySettings(BaseModel):
    enabled: bool = True


class MemorySettings(BaseModel):
    enabled: bool = True
    summary_buffer: SummaryBufferSettings = Field(default_factory=SummaryBufferSettings)
    slot_memory: SlotMemorySettings = Field(default_factory=SlotMemorySettings)


class DebugTraceSettings(BaseModel):
    enabled: bool = True
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/unit/test_runtime_pipeline.py -q`
Expected: PASS

---

### Task 2: LLM Query Rewriting에 재작성 trace를 붙인다

**Files:**
- Modify: `src/bidmate_rag/retrieval/multiturn.py`
- Test: `tests/unit/test_multiturn.py`

- [ ] **Step 1: 재작성 결과와 디버그 정보를 검증하는 테스트 추가**

```python
def test_llm_rewrite_returns_query_and_trace():
    llm = make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준")

    rewritten, trace = rewrite_query_with_history(
        query="평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=llm,
        mode="llm_with_rule_fallback",
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가기준"
    assert trace["original_query"] == "평가기준은?"
    assert trace["rewritten_query"] == rewritten
    assert trace["rewrite_applied"] is True
```

- [ ] **Step 2: 재작성 함수가 아래를 반환하도록 수정**

```python
{
    "original_query": query,
    "rewritten_query": rewritten_query,
    "rewrite_applied": rewritten_query != query,
    "rewrite_reason": "llm" | "rule_fallback" | "original",
    "rewrite_prompt_tokens": ...,
    "rewrite_completion_tokens": ...,
    "rewrite_total_tokens": ...,
    "rewrite_cost_usd": ...,
}
```

- [ ] **Step 3: 실패 시 폴백 규칙 유지**

```python
if mode == "rule_only":
    ...
elif mode == "llm_only":
    ...
else:
    # llm_with_rule_fallback
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/unit/test_multiturn.py -q`
Expected: PASS

---

### Task 3: Hybrid Retrieval과 Rerank 결과에 trace를 붙인다

**Files:**
- Modify: `src/bidmate_rag/retrieval/retriever.py`
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: 검색 전/후 결과를 보존하는 테스트 추가**

```python
def test_retriever_returns_rerank_and_trace():
    result = retriever.retrieve(
        query="평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        top_k=3,
    )

    assert "retrieved_chunks_before_rerank" in result.debug
    assert "retrieved_chunks_after_rerank" in result.debug
```

- [ ] **Step 2: retriever에 trace payload 추가**

```python
debug = {
    "original_query": query,
    "rewritten_query": resolved_query,
    "retrieved_chunks_before_rerank": before_rerank,
    "retrieved_chunks_after_rerank": after_rerank,
}
```

- [ ] **Step 3: reranker OFF일 때도 동일 필드 유지**

```python
after_rerank = before_rerank if self.reranker is None else reranked
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/unit/test_retriever.py -q`
Expected: PASS

---

### Task 4: Summary Buffer Memory + Slot Memory 계층을 추가

**Files:**
- Create: `src/bidmate_rag/retrieval/memory.py`
- Modify: `src/bidmate_rag/pipelines/chat.py`
- Test: `tests/unit/test_memory.py`

- [ ] **Step 1: 메모리 동작 테스트 작성**

```python
def test_memory_keeps_recent_turns_and_slots():
    memory = ConversationMemory(max_recent_turns=4, max_summary_tokens=128)

    state = memory.build(
        chat_history=[
            {"role": "user", "content": "교육부 클라우드 사업 알려줘"},
            {"role": "assistant", "content": "예산은 3억입니다."},
            {"role": "user", "content": "평가기준도 정리해줘"},
            {"role": "assistant", "content": "기술평가 비중이 큽니다."},
            {"role": "user", "content": "예산 다시 말해줘"},
        ]
    )

    assert len(state["recent_turns"]) <= 4
    assert "예산" in state["slot_memory"]
    assert "3억" in str(state["slot_memory"])
```

- [ ] **Step 2: 메모리 모듈 구현**

```python
class ConversationMemory:
    def __init__(self, max_recent_turns: int, max_summary_tokens: int) -> None:
        ...

    def build(self, chat_history: list[dict]) -> dict:
        return {
            "recent_turns": recent_turns,
            "summary_buffer": summary_buffer,
            "slot_memory": slot_memory,
        }
```

- [ ] **Step 3: 슬롯 메모리 최소 규칙 구현**

```python
slot_memory = {
    "발주기관": ...,
    "사업명": ...,
    "예산": ...,
    "일정": ...,
    "평가기준": ...,
}
```

- [ ] **Step 4: chat pipeline에서 메모리 생성 호출**

```python
memory_state = self.memory.build(chat_history or [])
```

- [ ] **Step 5: 테스트 실행**

Run: `uv run pytest tests/unit/test_memory.py -q`
Expected: PASS

---

### Task 5: 최종 생성 프롬프트에 메모리를 포함한다

**Files:**
- Modify: `src/bidmate_rag/pipelines/chat.py`
- Modify: `src/bidmate_rag/providers/llm/openai_compat.py`
- Test: `tests/unit/test_runtime_pipeline.py`

- [ ] **Step 1: 생성 입력에 메모리가 전달되는 테스트 추가**

```python
def test_chat_pipeline_passes_memory_to_llm():
    result = pipeline.answer(
        question="평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )
    assert result.debug["memory_slots"]
```

- [ ] **Step 2: 생성 입력 스키마에 메모리 추가**

```python
generation_payload = {
    "question": question,
    "rewritten_query": rewritten_query,
    "contexts": contexts,
    "memory_summary": memory_state["summary_buffer"],
    "memory_slots": memory_state["slot_memory"],
}
```

- [ ] **Step 3: 최종 답변 결과에도 메모리 trace 남기기**

```python
result.debug.update(
    {
        "memory_summary": memory_state["summary_buffer"],
        "memory_slots": memory_state["slot_memory"],
    }
)
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/unit/test_runtime_pipeline.py -q`
Expected: PASS

---

### Task 6: CLI에서 재작성/메모리/비용을 직접 보이게 한다

**Files:**
- Modify: `scripts/run_rag.py`
- Test: `tests/unit/test_run_rag.py`

- [ ] **Step 1: CLI 출력 테스트 작성**

```python
def test_run_rag_prints_rewrite_memory_and_cost(capsys):
    ...
    assert "원본 질문:" in captured.out
    assert "재작성 질문:" in captured.out
    assert "메모리 요약:" in captured.out
    assert "메모리 슬롯:" in captured.out
    assert "총 비용(USD):" in captured.out
```

- [ ] **Step 2: `history-file` BOM 처리도 같이 보강**

```python
payload = Path(history_file).read_text(encoding="utf-8-sig").lstrip("\ufeff")
```

- [ ] **Step 3: 최종 출력 형식 추가**

```python
print(f"원본 질문: {debug['original_query']}")
print(f"재작성 질문: {debug['rewritten_query']}")
print(f"메모리 요약: {debug['memory_summary']}")
print(f"메모리 슬롯: {debug['memory_slots']}")
print(f"재작성 비용(USD): {debug['rewrite_cost_usd']}")
print(f"생성 비용(USD): {result.cost_usd}")
print(f"총 비용(USD): {debug['total_cost_usd']}")
print(result.answer)
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/unit/test_run_rag.py -q`
Expected: PASS

---

### Task 7: CLI 기준 스모크 테스트와 문서 동기화

**Files:**
- Modify: `docs/superpowers/specs/2026-04-15-hybrid-search-design.md`
- Modify: `docs/superpowers/plans/2026-04-15-hybrid-search.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: CLI 검증 명령을 문서에 남긴다**

```powershell
uv run python scripts/run_rag.py --provider-config configs/providers/openai_gpt5mini.yaml --question "평가기준은?" --history-file .\history.json
```

- [ ] **Step 2: 문서에 현재 구조와 실제 구현 상태를 맞춘다**

핵심 반영:
- 추가 YAML 파일 없음
- Summary Buffer Memory + Slot Memory 포함
- CLI에서 재작성/메모리/비용 확인 가능

- [ ] **Step 3: 전체 관련 테스트 실행**

Run: `uv run pytest tests/unit/test_multiturn.py tests/unit/test_retriever.py tests/unit/test_memory.py tests/unit/test_runtime_pipeline.py tests/unit/test_run_rag.py -q`
Expected: PASS

---

## 완료 조건

아래가 CLI 한 번 실행으로 보이면 이번 1차 구현은 완료다.

- 원본 질문
- 재작성 질문
- 재작성 전/후 검색 결과
- 메모리 요약
- 메모리 슬롯
- 재작성 비용
- 생성 비용
- 총 비용
- 최종 답변

즉 이번 계획의 성공 기준은 `멀티턴이 실제로 어떻게 동작했는지 사용자가 CLI에서 바로 확인할 수 있는 상태`다.
