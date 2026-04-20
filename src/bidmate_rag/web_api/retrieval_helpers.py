"""Retrieval helpers for the web_api adapter.

web_api also uses the shared retriever path so web and Streamlit stay aligned.
Document mentions are handled here by converting them into explicit
`metadata_filter` constraints before calling `RAGRetriever.retrieve()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter
from typing import Protocol

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.generation.calculation_engine import build_calculation_generation_result
from bidmate_rag.generation.context_builder import build_numbered_context_block
from bidmate_rag.providers.llm.base import StreamDelta
from bidmate_rag.retrieval.filters import is_comparison_query
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
    chat_history: list[dict] | None = None,
) -> list[RetrievedChunk]:
    """문서별로 분할 검색한 뒤 비교형 질문은 문서 커버리지를 우선 보장한다.

    `retriever` 파라미터는 duck-typed (Protocol) — 테스트에서 FakeRetriever 주입 가능.
    """
    if not mentioned_doc_ids:
        raise ValueError("mentioned_doc_ids must be non-empty")
    comparison_mode = is_comparison_query(query)
    per_doc_k = _resolve_per_doc_k(
        query=query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )
    all_chunks: list[RetrievedChunk] = []
    for doc_id in mentioned_doc_ids:
        chunks = retriever.retrieve(
            query,
            chat_history=chat_history,
            top_k=per_doc_k,
            metadata_filter=_doc_where(doc_id),
        )
        all_chunks.extend(chunks)
    return _merge_chunks(
        all_chunks,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
        comparison_mode=comparison_mode,
    )


def _doc_where(doc_id: str) -> dict:
    """문서 id → ChromaDB where 절.

    Chunker의 `_chunk_doc_id`가 pandas NaN을 literal "nan"으로 저장해서 18개
    문서가 `doc_id='nan'`으로 인덱싱됨. 실제 식별자는 `파일명` 필드에 있으므로
    `doc_id` OR `파일명` 둘 다 매칭시켜야 한다.
    """
    return {"$or": [{"doc_id": doc_id}, {"파일명": doc_id}]}


def _resolve_per_doc_k(*, query: str, mentioned_doc_ids: list[str], top_k: int) -> int:
    """문서별 검색량을 계산한다.

    비교형 질문은 문서별 근거가 빠지기 쉬우므로 기본값보다 조금 더 넉넉하게 가져온다.
    """
    base = max(top_k // len(mentioned_doc_ids), 3) + 2
    if is_comparison_query(query):
        return max(base, 6)
    return base


def _chunk_doc_keys(chunk: RetrievedChunk) -> set[str]:
    metadata = chunk.chunk.metadata or {}
    keys = {
        str(chunk.chunk.doc_id),
        str(metadata.get("파일명", "")),
        str(metadata.get("ingest_file", "")),
    }
    return {key for key in keys if key}


def _merge_chunks(
    chunks: list[RetrievedChunk],
    *,
    mentioned_doc_ids: list[str],
    top_k: int,
    comparison_mode: bool,
) -> list[RetrievedChunk]:
    """최종 청크를 병합한다.

    비교형 질문은 top_k가 허용하는 범위에서 문서별 대표 청크를 먼저 확보한다.
    """
    if top_k <= 0:
        return []

    ranked = sorted(chunks, key=lambda c: -c.score)
    if not comparison_mode or len(mentioned_doc_ids) <= 1:
        return ranked[:top_k]

    selected: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()

    for doc_id in mentioned_doc_ids:
        for chunk in ranked:
            if chunk.chunk.chunk_id in seen_chunk_ids:
                continue
            if doc_id not in _chunk_doc_keys(chunk):
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk.chunk_id)
            break
        if len(selected) >= top_k:
            return selected[:top_k]

    for chunk in ranked:
        if chunk.chunk.chunk_id in seen_chunk_ids:
            continue
        selected.append(chunk)
        seen_chunk_ids.add(chunk.chunk.chunk_id)
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def vector_search(
    retriever: _RetrieverProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
    chat_history: list[dict] | None = None,
) -> list[RetrievedChunk]:
    """Use the shared retriever path with explicit document scoping when needed."""

    if not mentioned_doc_ids:
        return retriever.retrieve(query, chat_history=chat_history, top_k=top_k)

    if len(mentioned_doc_ids) == 1:
        return retriever.retrieve(
            query,
            chat_history=chat_history,
            top_k=top_k,
            metadata_filter=_doc_where(mentioned_doc_ids[0]),
        )

    return split_and_merge_chunks(
        retriever,
        query=query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
        chat_history=chat_history,
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
    chat_history: list[dict] | None = None,
) -> GenerationResult:
    """Web API의 통합 RAG 경로.

    모든 멘션 개수(0/1/N+)를 동일한 retriever 경로로 처리한다.
    """
    started_at = perf_counter()
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = vector_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
        chat_history=chat_history,
    )

    calculation_engine = getattr(pipeline, "calculation_engine", None)
    if calculation_engine is not None:
        calculation_answer = calculation_engine.try_answer(
            question=question,
            retrieved_chunks=chunks,
            metadata_filter=None,
        )
        if calculation_answer is not None:
            return build_calculation_generation_result(
                question=question,
                calculation_answer=calculation_answer,
                context_chunks=chunks,
                llm_provider=llm.provider_name,
                llm_model=llm.model_name,
                generation_config={
                    "scenario": runtime.provider.scenario or runtime.provider.provider,
                    "run_id": f"web-{provider_config}",
                    "embedding_provider": embedder.provider_name,
                    "embedding_model": embedder.model_name,
                },
                latency_ms=(perf_counter() - started_at) * 1000,
            )

    return llm.generate(
        question=question,
        context_chunks=chunks,
        history=chat_history or [],
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
    chat_history: list[dict] | None = None,
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
    started_at = perf_counter()
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = vector_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
        chat_history=chat_history,
    )

    # 컨텍스트 예산 기반으로 절단될 청크를 미리 제거 — retrieval 이벤트에서
    # 프론트가 받는 카드 수와 LLM이 실제로 보는 청크 수를 처음부터 맞춘다.
    _, used_indices = build_numbered_context_block(chunks, max_chars=max_context_chars)
    visible_chunks = [chunks[i] for i in used_indices]
    yield ("retrieval", visible_chunks)

    calculation_engine = getattr(pipeline, "calculation_engine", None)
    if calculation_engine is not None:
        calculation_answer = calculation_engine.try_answer(
            question=question,
            retrieved_chunks=visible_chunks,
            metadata_filter=None,
        )
        if calculation_answer is not None:
            result = build_calculation_generation_result(
                question=question,
                calculation_answer=calculation_answer,
                context_chunks=visible_chunks,
                llm_provider=llm.provider_name,
                llm_model=llm.model_name,
                generation_config={
                    "scenario": runtime.provider.scenario or runtime.provider.provider,
                    "run_id": f"web-{provider_config}",
                    "embedding_provider": embedder.provider_name,
                    "embedding_model": embedder.model_name,
                },
                latency_ms=(perf_counter() - started_at) * 1000,
            )
            yield ("token", result.answer)
            yield ("done", result)
            return

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
        history=chat_history or [],
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
