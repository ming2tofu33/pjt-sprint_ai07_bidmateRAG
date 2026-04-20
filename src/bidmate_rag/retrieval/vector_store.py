"""Chroma vector store integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from bidmate_rag.schema import Chunk, RetrievedChunk


def _primitive_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """메타데이터 값을 Chroma 호환 원시 타입으로 변환

    Args:
        metadata: 원본 메타데이터 딕셔너리.

    Returns:
        원시 타입만 포함하는 메타데이터 딕셔너리.
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _normalize_field_where(field: str, value: Any) -> dict[str, Any]:
    """Normalize a single field clause into a Chroma-compatible ``where``."""
    if not isinstance(value, dict) or not value:
        return {field: value}

    operator_items = [
        (operator, operand)
        for operator, operand in value.items()
        if isinstance(operator, str) and operator.startswith("$")
    ]
    if len(operator_items) != len(value):
        return {field: value}
    if len(operator_items) <= 1:
        return {field: value}
    return {"$and": [{field: {operator: operand}} for operator, operand in operator_items]}


def _normalize_where_clause(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize flat/mixed ``where`` filters into Chroma-compatible form."""
    if not where:
        return None

    if len(where) == 1:
        key, value = next(iter(where.items()))
        if key in {"$and", "$or"} and isinstance(value, list):
            return {
                key: [
                    normalized
                    for clause in value
                    if (normalized := _normalize_where_clause(clause)) is not None
                ]
            }
        return _normalize_field_where(key, value)

    clauses: list[dict[str, Any]] = []
    for key, value in where.items():
        if key in {"$and", "$or"} and isinstance(value, list):
            nested = _normalize_where_clause({key: value})
            if nested:
                clauses.append(nested)
            continue
        clauses.append(_normalize_field_where(key, value))

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorStore:
    """Chroma 기반 벡터 스토어."""

    def __init__(self, persist_dir: str | Path, collection_name: str) -> None:
        """ChromaVectorStore를 초기화

        Args:
            persist_dir: Chroma DB 저장 경로.
            collection_name: 컬렉션 이름.
        """
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
    def count(self) -> int:
    #컬렉션에 저장된 청크 수를 반환한다
        return self.collection.count()
    
    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 5000
    ) -> None:
        """청크와 임베딩을 컬렉션에 업서트

        Args:
            chunks: 저장할 Chunk 리스트.
            embeddings: 각 청크에 대응하는 임베딩 벡터 리스트.
            batch_size: 한 번에 처리할 배치 크기.

        주의: 같은 chunk_id가 있으면 갱신하지만, **컬렉션에 있던 이전 청크는
        지우지 않습니다**. 청킹 설정을 바꿔 청크 수가 줄어들면 stale 청크가
        남아 retrieval을 오염시킵니다. 청킹/재빌드 시에는
        :meth:`replace_documents`를 사용하세요.
        """
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=batch_embs,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    _primitive_metadata(
                        chunk.metadata
                        | {
                            "doc_id": chunk.doc_id,
                            "section": chunk.section,
                            "content_type": chunk.content_type,
                            "chunk_index": chunk.chunk_index,
                        }
                    )
                    for chunk in batch_chunks
                ],
            )

    def replace_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 5000,
    ) -> None:
        """청크가 속한 문서들의 기존 청크를 모두 삭제 후 새로 upsert.

        ``upsert``와 달리, 같은 ``doc_id``의 stale 청크 (이전 빌드에서 만들어진
        뒤 새 빌드에서는 사라진 청크)를 안전하게 제거합니다. 청킹 설정 변경
        후 재빌드 시 stale 청크가 retrieval을 오염시키는 것을 방지합니다.

        주의: 같은 collection에 다른 doc_id의 청크가 있으면 그것들은 보존됩니다.
        부분 update가 가능하므로 한 문서만 다시 빌드해도 안전.
        """
        if not chunks:
            return
        doc_ids = sorted({chunk.doc_id for chunk in chunks})
        if doc_ids:
            # ChromaDB $in operator로 한 번에 삭제
            self.collection.delete(where={"doc_id": {"$in": doc_ids}})
        self.upsert(chunks, embeddings, batch_size=batch_size)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[RetrievedChunk]:
        """임베딩 벡터로 유사 청크를 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터.
            top_k: 반환할 최대 결과 수.
            where: 메타데이터 필터 조건.
            where_document: 문서 내용 필터 조건.

        Returns:
            RetrievedChunk 리스트.
        """
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = _normalize_where_clause(where)
        if where_document:
            kwargs["where_document"] = where_document
        results = self.collection.query(**kwargs)
        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][index]
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", metadata.get("공고 번호", ""))),
                text=results["documents"][0][index],
                text_with_meta=results["documents"][0][index],
                char_count=len(results["documents"][0][index]),
                section=str(metadata.get("section", "")),
                content_type=str(metadata.get("content_type", "text")),
                chunk_index=int(metadata.get("chunk_index", index)),
                metadata=metadata,
            )
            retrieved.append(
                RetrievedChunk(
                    rank=index + 1,
                    score=round(1 - results["distances"][0][index], 4),
                    chunk=chunk,
                )
            )
        return retrieved
