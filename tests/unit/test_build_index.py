from pathlib import Path

import pandas as pd

from bidmate_rag.pipelines.build_index import build_index_from_parquet


class FakeEmbedder:
    provider_name = "fake-embedder"
    model_name = "fake-model"

    def __init__(self):
        self.last_texts = None

    def embed_documents(self, texts):
        self.last_texts = texts
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.upserts = []

    def upsert(self, chunks, embeddings):
        self.upserts.append((chunks, embeddings))

    def replace_documents(self, chunks, embeddings, batch_size=5000):
        """build_index가 replace_documents를 호출하므로 fake에서도 필요."""
        self.upserts.append((chunks, embeddings))


def test_build_index_filters_short_chunks_and_uses_text_with_meta(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [
            {
                "chunk_id": "short-1",
                "doc_id": "doc-1",
                "text": "짧음",
                "text_with_meta": "[meta]\n짧음",
                "char_count": 10,
                "section": "개요",
                "content_type": "text",
                "chunk_index": 0,
                "파일명": "a.hwp",
            },
            {
                "chunk_id": "long-1",
                "doc_id": "doc-1",
                "text": "충분히 긴 청크",
                "text_with_meta": "[meta]\n충분히 긴 청크",
                "char_count": 100,
                "section": "요구사항",
                "content_type": "text",
                "chunk_index": 1,
                "파일명": "a.hwp",
            },
        ]
    ).to_parquet(chunk_path, index=False)

    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    stats = build_index_from_parquet(
        chunk_path, embedder=embedder, vector_store=vector_store, min_chars=50
    )

    assert stats["indexed_chunks"] == 1
    assert embedder.last_texts == ["[meta]\n충분히 긴 청크"]
    assert vector_store.upserts[0][0][0].chunk_id == "long-1"
