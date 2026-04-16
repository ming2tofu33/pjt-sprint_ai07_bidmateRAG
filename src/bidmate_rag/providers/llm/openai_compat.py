"""OpenAI-compatible client adapter for OpenAI and local endpoints."""

from __future__ import annotations

import os
import time as _time
from collections.abc import Iterator
from uuid import uuid4

from openai import OpenAI

from bidmate_rag.config.prompts import build_rag_user_prompt
from bidmate_rag.generation.context_builder import build_numbered_context_block
from bidmate_rag.providers.llm.base import BaseLLMProvider, StreamDelta
from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing


class OpenAICompatibleLLM(BaseLLMProvider):
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_base = api_base
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )
        self.pricing = load_pricing()

    def _build_messages(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> tuple[list[dict], str, list[int]]:
        """공통 메시지 빌드 — generate/generate_stream이 공유.

        Returns:
            (messages, context_block, used_indices):
              - used_indices: max_context_chars 예산 내에 실제로 포함된
                원본 청크의 인덱스. `_build_result`가 이걸로 `retrieved_chunks`를
                LLM이 본 것만으로 필터링한다.
        """
        context, used_indices = build_numbered_context_block(
            context_chunks, max_chars=generation_config.get("max_context_chars", 8000)
        )
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # history는 두 가지 형식을 모두 지원:
        #  1. legacy: [{"user": "...", "assistant": "..."}]
        #  2. OpenAI 표준: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        for item in (history or [])[-4:]:
            if not isinstance(item, dict):
                continue
            if "role" in item and "content" in item:
                if item["role"] in ("user", "assistant"):
                    messages.append({"role": item["role"], "content": item["content"]})
            elif "user" in item or "assistant" in item:
                if "user" in item:
                    messages.append({"role": "user", "content": item["user"]})
                if "assistant" in item:
                    messages.append({"role": "assistant", "content": item["assistant"]})
        messages.append(
            {"role": "user", "content": build_rag_user_prompt(question, context)}
        )
        return messages, context, used_indices

    def _build_result(
        self,
        *,
        question: str,
        context_chunks: list[RetrievedChunk],
        context: str,
        generation_config: dict,
        answer: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        total_tokens: int,
        used_indices: list[int] | None = None,
    ) -> GenerationResult:
        computed_cost = calc_llm_cost(
            self.model_name,
            prompt_tokens,
            completion_tokens,
            self.pricing,
            cached_tokens=cached_tokens,
        )
        # LLM이 실제로 본 청크만 남긴다 — 본문의 [n]과 Citation 카드가 정확히 매칭되도록.
        # used_indices=None이면 legacy 경로 (전체 유지).
        visible_chunks = (
            [context_chunks[i] for i in used_indices]
            if used_indices is not None
            else context_chunks
        )
        return GenerationResult(
            question_id=generation_config.get("question_id", f"q-{uuid4().hex[:8]}"),
            question=question,
            scenario=generation_config.get("scenario", "ad-hoc"),
            run_id=generation_config.get("run_id", f"run-{uuid4().hex[:8]}"),
            embedding_provider=generation_config.get("embedding_provider", ""),
            embedding_model=generation_config.get("embedding_model", ""),
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer=answer or "(응답 없음)",
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in visible_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in visible_chunks],
            retrieved_chunks=visible_chunks,
            latency_ms=round(latency_ms, 1),
            token_usage={
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "cached": cached_tokens,
                "total": total_tokens,
            },
            cost_usd=float(generation_config.get("cost_usd", computed_cost)),
            context=context,
        )

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        messages, context, used_indices = self._build_messages(
            question, context_chunks, history, generation_config, system_prompt
        )
        _start = _time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=generation_config.get("max_completion_tokens", 16000),
        )
        _elapsed_ms = (_time.time() - _start) * 1000
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return self._build_result(
            question=question,
            context_chunks=context_chunks,
            context=context,
            generation_config=generation_config,
            answer=response.choices[0].message.content or "",
            latency_ms=_elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            used_indices=used_indices,
        )

    def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> Iterator[StreamDelta | GenerationResult]:
        """OpenAI SDK 네이티브 스트리밍. 토큰 delta를 yield하고 마지막에
        완성된 `GenerationResult`를 yield한다. usage는 `stream_options`의
        `include_usage=True`로 마지막 청크에 포함되며, 이를 기반으로 cost 계산."""
        messages, context, used_indices = self._build_messages(
            question, context_chunks, history, generation_config, system_prompt
        )
        _start = _time.time()
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=generation_config.get("max_completion_tokens", 16000),
            stream=True,
            stream_options={"include_usage": True},
        )
        full_text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        total_tokens = 0
        for chunk in stream:
            # 토큰 delta
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    full_text_parts.append(content)
                    yield StreamDelta(text=content)
            # 마지막 청크에 usage 포함 (include_usage=True 시)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        _elapsed_ms = (_time.time() - _start) * 1000
        yield self._build_result(
            question=question,
            context_chunks=context_chunks,
            context=context,
            generation_config=generation_config,
            answer="".join(full_text_parts),
            latency_ms=_elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            used_indices=used_indices,
        )
