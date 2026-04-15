"""Retrieval helpers for the web_api adapter.

web_api는 `@` 멘션과 `/` 슬래시 커맨드로 사용자 의도를 명시적으로 수집한다.
이 때문에 `RAGRetriever.retrieve`의 자동 추출·섹션 힌트 기반
`where_document={"$contains": ...}` 필터는 오히려 해가 된다 (문서에 그
섹션 키워드가 literal로 없으면 retrieval 결과 0). 따라서 이 모듈은
`vector_store.query`를 직접 호출하고 필요하면 여러 문서에 대해 loop를 돈다.
"""

from __future__ import annotations

from typing import Protocol

from bidmate_rag.config.prompts import SYSTEM_PROMPT
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


class _VectorStoreProtocol(Protocol):
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[RetrievedChunk]: ...


class _EmbedderProtocol(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


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
            metadata_filter={"doc_id": doc_id},
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
    vector_store: _VectorStoreProtocol,
    embedder: _EmbedderProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """`vector_store.query`를 직접 호출해 section_hint 필터를 우회한다.

    - 멘션 0개: 필터 없이 top_k 검색
    - 멘션 1개: `{"$or": [{"doc_id": id}, {"파일명": id}]}` 단일 검색
    - 멘션 2개+: 문서별 loop + 점수 병합 (per-doc split)
    """
    query_embedding = embedder.embed_query(query)

    if not mentioned_doc_ids:
        return vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=None,
            where_document=None,
        )

    if len(mentioned_doc_ids) == 1:
        return vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=_doc_where(mentioned_doc_ids[0]),
            where_document=None,
        )

    # per-doc split
    per_doc_k = max(top_k // len(mentioned_doc_ids), 3) + 2
    all_chunks: list[RetrievedChunk] = []
    for doc_id in mentioned_doc_ids:
        chunks = vector_store.query(
            query_embedding=query_embedding,
            top_k=per_doc_k,
            where=_doc_where(doc_id),
            where_document=None,
        )
        all_chunks.extend(chunks)
    all_chunks.sort(key=lambda c: -c.score)
    return all_chunks[:top_k]


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

    `run_live_query`와 `RAGRetriever.retrieve`를 우회하고 직접:
      1. `embedder.embed_query`로 임베딩 생성
      2. `vector_store.query`로 검색 (section_hint 필터 없음)
      3. `llm.generate`로 응답 생성

    모든 멘션 개수(0/1/N+)를 단일 경로로 처리한다.
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    vector_store = pipeline.retriever.vector_store

    chunks = vector_search(
        vector_store,
        embedder,
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
