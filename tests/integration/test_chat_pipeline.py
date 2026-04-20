from bidmate_rag.pipelines.chat import RAGChatPipeline
from bidmate_rag.retrieval.memory import ConversationMemory
from bidmate_rag.schema import Chunk, GenerationResult, RetrievedChunk


class FakeRetriever:
    _last_debug = {
        "original_query": "요구사항 알려줘",
        "rewritten_query": "요구사항 알려줘",
        "rewrite_applied": False,
        "rewrite_reason": "original",
        "rewrite_prompt_tokens": 0,
        "rewrite_completion_tokens": 0,
        "rewrite_total_tokens": 0,
        "rewrite_cost_usd": 0.0,
        "retrieved_chunks_before_rerank": [],
        "retrieved_chunks_after_rerank": [],
    }

    def retrieve(
        self,
        query: str,
        chat_history=None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        chunk = Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="핵심 요구사항",
            text_with_meta="[발주기관: 기관 | 사업명: 사업]\n핵심 요구사항",
            char_count=7,
            section="요구사항",
            content_type="text",
            chunk_index=0,
            metadata={"사업명": "사업", "발주 기관": "기관", "파일명": "sample.hwp"},
        )
        return [RetrievedChunk(rank=1, score=0.95, chunk=chunk)]


class FakeLLM:
    provider_name = "fake-llm"
    model_name = "fake-model"

    def generate(self, question, context_chunks, history, generation_config, system_prompt):
        return GenerationResult(
            question_id="q-1",
            question=question,
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer="문서 기준 답변",
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in context_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in context_chunks],
            retrieved_chunks=context_chunks,
            latency_ms=10,
            token_usage={"total": 5},
        )


def test_chat_pipeline_returns_generation_result() -> None:
    pipeline = RAGChatPipeline(retriever=FakeRetriever(), llm=FakeLLM())

    result = pipeline.answer("요구사항 알려줘")

    assert result.answer == "문서 기준 답변"
    assert result.retrieved_chunk_ids == ["chunk-1"]
    assert result.llm_model == "fake-model"


def test_chat_pipeline_includes_memory_debug() -> None:
    pipeline = RAGChatPipeline(
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        memory=ConversationMemory(
            max_recent_turns=4,
            max_summary_chars=120,
            agency_list=["국민연금공단"],
        ),
    )

    result = pipeline.answer(
        "평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert "memory_summary" in result.debug
    assert "memory_slots" in result.debug
    assert result.debug["generation_cost_usd"] == 0.0


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


def test_chat_pipeline_uses_minimal_rewrite_state_when_debug_trace_disabled() -> None:
    class CapturingLLM(FakeLLM):
        def __init__(self) -> None:
            self.last_generation_config = None

        def generate(self, question, context_chunks, history, generation_config, system_prompt):
            self.last_generation_config = dict(generation_config)
            return super().generate(
                question,
                context_chunks,
                history,
                generation_config,
                system_prompt,
            )

    class RetrieverWithMinimalRuntimeState:
        _last_debug = {
            "rewritten_query": "국민연금공단 차세대 ERP 사업의 평가기준은?",
            "rewrite_prompt_tokens": 12,
            "rewrite_completion_tokens": 7,
            "rewrite_total_tokens": 19,
            "rewrite_cost_usd": 0.000123,
            "memory_state": {
                "recent_turns": [],
                "summary_buffer": "요약-노디버그",
                "slot_memory": {"발주기관": "국민연금공단"},
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

    llm = CapturingLLM()
    pipeline = RAGChatPipeline(
        retriever=RetrieverWithMinimalRuntimeState(),
        llm=llm,
        memory=ConversationMemory(
            max_recent_turns=4,
            max_summary_chars=120,
            agency_list=["국민연금공단"],
        ),
        debug_trace_enabled=False,
    )

    result = pipeline.answer(
        "평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 ERP 사업 알려줘"}],
    )

    assert llm.last_generation_config["rewritten_query"] == "국민연금공단 차세대 ERP 사업의 평가기준은?"
    assert llm.last_generation_config["memory_summary"] == "요약-노디버그"
    assert llm.last_generation_config["memory_slots"] == {"발주기관": "국민연금공단"}
    assert result.token_usage["rewrite_prompt"] == 12
    assert result.token_usage["rewrite_completion"] == 7
    assert result.token_usage["rewrite_total"] == 19
    assert result.debug == {}


def test_chat_pipeline_reuses_memory_state_when_debug_trace_disabled() -> None:
    """debug_trace_enabled=False 경로에서도 retriever의 memory_state를 재사용해야 한다."""

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

    class RetrieverWithOnlyMemoryState:
        """debug 꺼진 retriever 흉내 — _last_debug에 memory_state만 담긴 경우."""

        _last_debug = {
            "memory_state": {
                "recent_turns": [],
                "summary_buffer": "요약-노디버그",
                "slot_memory": {"발주기관": "국민연금공단"},
            }
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
        retriever=RetrieverWithOnlyMemoryState(), llm=FakeLLM(), memory=memory
    )

    result = pipeline.answer(
        "평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 ERP 사업 알려줘"}],
    )

    assert memory.build_calls == 0, (
        f"debug 꺼진 경로에서도 chat이 memory.build를 {memory.build_calls}회 호출."
    )
    assert result.debug["memory_summary"] == "요약-노디버그"
    assert result.debug["memory_slots"] == {"발주기관": "국민연금공단"}
