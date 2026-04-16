"""Common interfaces for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from bidmate_rag.schema import GenerationResult, RetrievedChunk


@dataclass
class StreamDelta:
    """스트리밍 중 증분 토큰. `generate_stream`이 토큰 단위로 방출한다."""

    text: str


class BaseLLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> Iterator[StreamDelta | GenerationResult]:
        """토큰 스트리밍 인터페이스.

        구현체는 `StreamDelta`를 연속으로 yield한 뒤, 마지막에 **단 한 번**
        완성된 `GenerationResult`를 yield해야 한다. 프로토콜:

            yield StreamDelta("첫")
            yield StreamDelta("번째")
            yield StreamDelta(" 답변")
            yield GenerationResult(answer="첫번째 답변", ...)  # 1회만

        기본 구현은 동기 `generate()`로 폴백한다 — 답변 전체를 한 번에
        delta로 방출하므로 프론트 입장에서는 "즉시 완성"처럼 보인다.
        실제 토큰 스트리밍을 원하는 프로바이더는 이 메서드를 오버라이드.
        """
        result = self.generate(
            question=question,
            context_chunks=context_chunks,
            history=history,
            generation_config=generation_config,
            system_prompt=system_prompt,
        )
        yield StreamDelta(text=result.answer)
        yield result
