"""RAG 채팅 파이프라인.

검색(retrieval) → LLM 생성(generation)을 연결하여
사용자 질문에 문서 근거 기반 답변을 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.generation.calculation_engine import build_calculation_generation_result
from bidmate_rag.schema import GenerationResult


@dataclass(slots=True)
class RAGChatPipeline:
    """RAG 질의응답 파이프라인.

    retriever로 관련 청크를 검색하고 llm으로 답변을 생성한다.

    Attributes:
        retriever: 벡터 검색 + 메타데이터 필터링 리트리버.
        llm: LLM 프로바이더 (generate 메서드 필요).
        system_prompt: 시스템 프롬프트.
        default_generation_config: 기본 생성 설정.
    """

    retriever: object
    llm: object
    memory: object | None = None
    system_prompt: str = SYSTEM_PROMPT
    default_generation_config: dict = field(default_factory=dict)
    debug_trace_enabled: bool = True
    calculation_engine: object | None = None

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        top_k: int = 5,
        generation_config: dict | None = None,
        metadata_filter: dict | None = None,
        question_id: str | None = None,
        scenario: str | None = None,
        run_id: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> GenerationResult:
        """사용자 질문에 대해 RAG 기반 답변을 생성한다.

        Args:
            question: 사용자 질문.
            chat_history: 이전 대화 히스토리 (multi-turn 평가용).
            top_k: 검색할 청크 수.
            generation_config: LLM 생성 설정 오버라이드.
            metadata_filter: ChromaDB ``where`` 절로 직접 적용할 필터.
                평가셋의 ``metadata_filter`` 또는 Streamlit UI의 수동 필터.
            question_id: 평가용 질문 ID.
            scenario: 실험 시나리오 (A/B).
            run_id: 실행 ID.
            embedding_provider: 임베딩 프로바이더명.
            embedding_model: 임베딩 모델명.

        Returns:
            GenerationResult (답변, 검색 청크, 토큰 사용량 등).
        """
        started_at = perf_counter()
        retrieved = self.retriever.retrieve(
            question,
            chat_history=chat_history,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
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
        config = {**self.default_generation_config, **(generation_config or {})}
        if question_id is not None:
            config["question_id"] = question_id
        if scenario is not None:
            config["scenario"] = scenario
        if run_id is not None:
            config["run_id"] = run_id
        if embedding_provider is not None:
            config["embedding_provider"] = embedding_provider
        if embedding_model is not None:
            config["embedding_model"] = embedding_model
        config["rewritten_query"] = retrieval_debug.get("rewritten_query", question)
        config["memory_summary"] = memory_state.get("summary_buffer", "")
        config["memory_slots"] = memory_state.get("slot_memory", {})

        if self.calculation_engine is not None:
            calculation_answer = self.calculation_engine.try_answer(
                question=question,
                retrieved_chunks=retrieved,
                metadata_filter=metadata_filter,
            )
            if calculation_answer is not None:
                result = build_calculation_generation_result(
                    question=question,
                    calculation_answer=calculation_answer,
                    context_chunks=retrieved,
                    llm_provider=getattr(self.llm, "provider_name", ""),
                    llm_model=getattr(self.llm, "model_name", ""),
                    generation_config=config,
                    latency_ms=(perf_counter() - started_at) * 1000,
                )
                if self.debug_trace_enabled:
                    rewrite_cost = float(retrieval_debug.get("rewrite_cost_usd", 0.0) or 0.0)
                    result.debug.update(retrieval_debug)
                    result.debug.update(
                        {
                            "memory_recent_turns": memory_state.get("recent_turns", []),
                            "memory_summary": memory_state.get("summary_buffer", ""),
                            "memory_slots": memory_state.get("slot_memory", {}),
                            "generation_cost_usd": 0.0,
                            "total_cost_usd": round(rewrite_cost, 6),
                        }
                    )
                return result

        result = self.llm.generate(
            question=question,
            context_chunks=retrieved,
            history=chat_history or [],
            generation_config=config,
            system_prompt=self.system_prompt,
        )
        rewrite_tokens = {
            "rewrite_prompt": int(retrieval_debug.get("rewrite_prompt_tokens", 0) or 0),
            "rewrite_completion": int(retrieval_debug.get("rewrite_completion_tokens", 0) or 0),
            "rewrite_total": int(retrieval_debug.get("rewrite_total_tokens", 0) or 0),
        }
        if any(rewrite_tokens.values()):
            result.token_usage.update(rewrite_tokens)
        if self.debug_trace_enabled:
            generation_cost = float(result.cost_usd or 0.0)
            rewrite_cost = float(retrieval_debug.get("rewrite_cost_usd", 0.0) or 0.0)
            result.debug.update(retrieval_debug)
            result.debug.update(
                {
                    "memory_recent_turns": memory_state.get("recent_turns", []),
                    "memory_summary": memory_state.get("summary_buffer", ""),
                    "memory_slots": memory_state.get("slot_memory", {}),
                    "generation_cost_usd": generation_cost,
                    "total_cost_usd": round(generation_cost + rewrite_cost, 6),
                }
            )
        return result
