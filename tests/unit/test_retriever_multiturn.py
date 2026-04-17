"""Retriever가 rewrite된 쿼리를 문서 부스팅/히스토리 경로에 전파하는지 검증."""

from unittest.mock import MagicMock

from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.schema import Chunk, RetrievedChunk


class _RecordingMetadataStore:
    def __init__(self, agency_list: list[str], relevant_docs: list[str]) -> None:
        self.agency_list = agency_list
        self._relevant_docs = relevant_docs
        self.find_relevant_docs_calls: list[str] = []

    def find_relevant_docs(self, query: str, top_n: int = 3) -> list[str]:
        self.find_relevant_docs_calls.append(query)
        return self._relevant_docs


def _make_chunk(chunk_id: str = "c-1") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}-doc",
        text="t",
        text_with_meta="t",
        char_count=1,
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={"파일명": "a.hwp"},
    )
    return RetrievedChunk(rank=1, score=0.9, chunk=chunk)


def test_retriever_uses_resolved_query_for_project_clue_augmentation_without_llm() -> None:
    """LLM 없이 규칙 기반 rewrite가 발생하는 케이스로 resolved_query 일관성 검증."""
    metadata_store = _RecordingMetadataStore(
        agency_list=["국민연금공단"],
        relevant_docs=["차세대_ERP.hwp", "차세대_ERP_2.hwp"],
    )
    vector_store = MagicMock()
    vector_store.query.return_value = [_make_chunk()]
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=metadata_store,
        rewrite_llm=None,  # 룰 기반만
        rewrite_mode="rule_only",
    )

    retriever.retrieve(
        "그 사업 예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 구축 사업 알려줘"}
        ],
    )

    # 룰 rewrite가 "그 사업" → "국민연금공단 차세대 ERP 구축 사업"으로 치환.
    # find_relevant_docs는 resolved_query로 호출되어야 project_clues 추출 가능.
    assert any(
        "차세대 ERP" in call for call in metadata_store.find_relevant_docs_calls
    ), (
        "resolved_query가 문서 부스팅에 전달되지 않음. "
        f"실제 호출: {metadata_store.find_relevant_docs_calls}"
    )
