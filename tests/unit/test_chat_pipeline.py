from bidmate_rag.pipelines.chat import RAGChatPipeline
from bidmate_rag.schema import Chunk, RetrievedChunk


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self._last_debug = {}

    def retrieve(self, *_args, **_kwargs):
        return self._chunks


class _FailLLM:
    provider_name = "fake"
    model_name = "fake-model"

    def generate(self, *args, **kwargs):
        raise AssertionError("계산형 질문은 LLM generate를 타면 안 됩니다.")


class _FakeCalculationEngine:
    def try_answer(self, *, question, retrieved_chunks, metadata_filter=None):
        return type(
            "CalcAnswer",
            (),
            {
                "mode": "budget_difference",
                "answer": "핵심 답변:\n- 차액은 700,000,000원입니다.\n\n계산 근거:\n- 테스트",
            },
        )()


def _chunk(doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=1.0,
        chunk=Chunk(
            chunk_id=f"{doc_id}-chunk-1",
            doc_id=doc_id,
            text="본문",
            text_with_meta="본문",
            char_count=10,
            chunk_index=0,
            metadata={"파일명": doc_id, "사업명": "테스트 사업"},
        ),
    )


def test_chat_pipeline_returns_calculation_result_without_llm() -> None:
    pipeline = RAGChatPipeline(
        retriever=_FakeRetriever([_chunk("doc-a.hwp"), _chunk("doc-b.hwp")]),
        llm=_FailLLM(),
        calculation_engine=_FakeCalculationEngine(),
    )

    result = pipeline.answer("예산 차이는 얼마야?")

    assert result.answer.startswith("핵심 답변:")
    assert result.debug["calculation_mode"] == "budget_difference"
    assert result.token_usage["total"] == 0
