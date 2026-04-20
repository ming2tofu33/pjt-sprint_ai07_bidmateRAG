"""벡터 인덱스 빌드 파이프라인.

chunks.parquet를 읽어 임베딩을 생성하고 ChromaDB에 저장한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.schema import Chunk


def _row_to_chunk(row: dict) -> Chunk:
    """parquet 행(dict)을 Chunk 객체로 변환한다.

    Args:
        row: chunks.parquet의 한 행.

    Returns:
        Chunk 인스턴스.
    """
    metadata_keys = {
        key
        for key in row
        if key
        not in {
            "chunk_id",
            "doc_id",
            "text",
            "text_with_meta",
            "char_count",
            "section",
            "content_type",
            "chunk_index",
        }
    }
    metadata = {key: row[key] for key in metadata_keys}
    return Chunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row.get("doc_id") or metadata.get("공고 번호") or metadata.get("파일명")),
        text=str(row["text"]),
        text_with_meta=str(row["text_with_meta"]),
        char_count=int(row["char_count"]),
        section=str(row.get("section", "")),
        content_type=str(row.get("content_type", "text")),
        chunk_index=int(row.get("chunk_index", 0)),
        metadata=metadata,
    )


def build_index_from_parquet(
    chunks_path: str | Path,
    embedder,
    vector_store,
    min_chars: int = 50,
) -> dict[str, int | str]:
    """parquet 파일에서 청크를 읽어 벡터 인덱스를 생성한다.

    Args:
        chunks_path: chunks.parquet 경로.
        embedder: 임베딩 프로바이더 (embed_documents 메서드 필요).
        vector_store: 벡터 저장소 (upsert 메서드 필요).
        min_chars: 최소 글자수 미만의 청크는 제외.

    Returns:
        입력/인덱싱 청크 수, 임베딩 모델 정보 딕셔너리.
    """
    frame = pd.read_parquet(chunks_path, dtype_backend="numpy_nullable")
    filtered = frame[frame["char_count"] >= min_chars].copy()
    chunks = [_row_to_chunk(row) for row in filtered.to_dict(orient="records")]

    # 배치 임베딩 (토큰 한도 초과 시 배치 크기를 절반으로 줄여 재시도)
    batch_size = 100
    all_embeddings: list[list[float]] = []
    i = 0
    while i < len(chunks):
        batch = chunks[i : i + batch_size]
        print(f"임베딩 중: [{i + 1}~{i + len(batch)}/{len(chunks)}] (배치={batch_size})")
        try:
            batch_embeddings = embedder.embed_documents([c.text_with_meta for c in batch])
            all_embeddings.extend(batch_embeddings)
            i += batch_size
        except Exception as exc:
            err_msg = str(exc).lower()
            is_retryable = (
                "300000" in str(exc)
                or "out of memory" in err_msg
                or "cuda" in err_msg
                and "memory" in err_msg
            )
            if is_retryable and batch_size > 5:
                batch_size = batch_size // 2
                print(f"  → 메모리/토큰 한도 초과, 배치 {batch_size}로 축소 후 재시도")
            else:
                raise

    # replace_documents: 같은 doc_id의 이전 청크를 모두 삭제 후 upsert
    # → 청킹 설정 변경으로 청크 수가 줄어도 stale 청크가 남지 않음
    vector_store.replace_documents(chunks, all_embeddings)
    return {
        "input_chunks": int(len(frame)),
        "indexed_chunks": len(chunks),
        "embedding_provider": getattr(embedder, "provider_name", ""),
        "embedding_model": getattr(embedder, "model_name", ""),
        "embedding_total_tokens": int(getattr(embedder, "cumulative_tokens", 0) or 0),
    }
