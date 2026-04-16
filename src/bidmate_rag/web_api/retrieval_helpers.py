"""Retrieval helpers for the web_api adapter.

web_api also uses the shared retriever path so web and Streamlit stay aligned.
Document mentions are handled here by converting them into explicit
`metadata_filter` constraints before calling `RAGRetriever.retrieve()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.generation.context_builder import build_numbered_context_block
from bidmate_rag.providers.llm.base import StreamDelta
from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.web_api.pipeline_cache import get_pipeline


class _RetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        chat_history: list | None = None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]: ...


def split_and_merge_chunks(
    retriever: _RetrieverProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """문서별로 top_k//N + 2개씩 검색한 뒤 점수로 정렬·절단한다.

    `retriever` 파라미터는 duck-typed (Protocol) — 테스트에서 FakeRetriever 주입 가능.
    """
    if not mentioned_doc_ids:
        raise ValueError("mentioned_doc_ids must be non-empty")
    per_doc_k = max(top_k // len(mentioned_doc_ids), 3) + 2
    all_chunks: list[RetrievedChunk] = []
    for doc_id in mentioned_doc_ids:
        chunks = retriever.retrieve(
            query,
            chat_history=None,
            top_k=per_doc_k,
            metadata_filter=_doc_where(doc_id),
        )
        all_chunks.extend(chunks)
    all_chunks.sort(key=lambda c: -c.score)
    return all_chunks[:top_k]


def _doc_where(doc_id: str) -> dict:
    """문서 id → ChromaDB where 절.

    Chunker의 `_chunk_doc_id`가 pandas NaN을 literal "nan"으로 저장해서 18개
    문서가 `doc_id='nan'`으로 인덱싱됨. 실제 식별자는 `파일명` 필드에 있으므로
    `doc_id` OR `파일명` 둘 다 매칭시켜야 한다.
    """
    return {"$or": [{"doc_id": doc_id}, {"파일명": doc_id}]}


def vector_search(
    retriever: _RetrieverProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """Use the shared retriever path with explicit document scoping when needed."""

    if not mentioned_doc_ids:
        return retriever.retrieve(query, chat_history=None, top_k=top_k)

    if len(mentioned_doc_ids) == 1:
        return retriever.retrieve(
            query,
            chat_history=None,
            top_k=top_k,
            metadata_filter=_doc_where(mentioned_doc_ids[0]),
        )

    return split_and_merge_chunks(
        retriever,
        query=query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )


def web_query(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> GenerationResult:
    """Web API의 통합 RAG 경로.

    모든 멘션 개수(0/1/N+)를 동일한 retriever 경로로 처리한다.
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = vector_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )

    return llm.generate(
        question=question,
        context_chunks=chunks,
        history=[],
        generation_config={
            "max_context_chars": max_context_chars,
            "scenario": runtime.provider.scenario or runtime.provider.provider,
            "run_id": f"web-{provider_config}",
            "embedding_provider": embedder.provider_name,
            "embedding_model": embedder.model_name,
        },
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )


def web_query_stream(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> Iterator[tuple[str, object]]:
    """Streaming 버전의 `web_query`.

    이벤트 스트림을 (event_type, payload) 튜플로 방출:
      1. ("retrieval", list[RetrievedChunk]) — 검색 완료 직후 1회.
         **컨텍스트 예산으로 절단될 청크는 이미 제외된 상태**로 방출되므로,
         프론트가 즉시 표시하는 Citation 카드 개수와 LLM이 답변에 쓰는 `[n]`
         번호가 1:1로 일치한다.
      2. ("token", str)                      — LLM delta마다
      3. ("done", GenerationResult)          — 스트림 종료 시 1회
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = vector_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )

    # 컨텍스트 예산 기반으로 절단될 청크를 미리 제거 — retrieval 이벤트에서
    # 프론트가 받는 카드 수와 LLM이 실제로 보는 청크 수를 처음부터 맞춘다.
    _, used_indices = build_numbered_context_block(chunks, max_chars=max_context_chars)
    visible_chunks = [chunks[i] for i in used_indices]
    yield ("retrieval", visible_chunks)

    gen_config = {
        "max_context_chars": max_context_chars,
        "scenario": runtime.provider.scenario or runtime.provider.provider,
        "run_id": f"web-{provider_config}",
        "embedding_provider": embedder.provider_name,
        "embedding_model": embedder.model_name,
    }
    for item in llm.generate_stream(
        question=question,
        context_chunks=visible_chunks,
        history=[],
        generation_config=gen_config,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    ):
        if isinstance(item, StreamDelta):
            yield ("token", item.text)
        elif isinstance(item, GenerationResult):
            yield ("done", item)


# Backward-compat alias: kept for tests + existing imports
def per_doc_split_query(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> GenerationResult:
    """멘션 2개 이상일 때 진입하던 기존 경로 — 이제 `web_query`로 위임."""
    return web_query(
        question=question,
        augmented_query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        provider_config=provider_config,
        chunking_config=chunking_config,
        system_prompt=system_prompt,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )
