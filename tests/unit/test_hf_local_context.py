from bidmate_rag.providers.llm.hf_local import HFLocalLLM
from bidmate_rag.schema import Chunk, RetrievedChunk


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _FakeGenerator:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.last_call: dict | None = None

    def __call__(self, full_input: str, **kwargs):
        self.last_call = {"full_input": full_input, **kwargs}
        return [{"generated_text": "로컬 응답"}]


def _make_chunk(
    *,
    chunk_id: str,
    text: str,
    budget: str,
    filename: str,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}-doc",
        text=text,
        text_with_meta=text,
        char_count=len(text),
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={
            "사업명": "차세대 ERP 구축",
            "발주 기관": "한국가스공사",
            "파일명": filename,
            "사업 금액": budget,
        },
    )
    return RetrievedChunk(rank=1, score=0.95, chunk=chunk)


def test_hf_local_provider_uses_metadata_aware_context_and_honors_context_limit() -> None:
    generator = _FakeGenerator()
    provider = HFLocalLLM(
        model_name="hf-test",
        provider_name="huggingface",
        generator=generator,
    )
    first_chunk = _make_chunk(
        chunk_id="chunk-1",
        text="핵심 요구사항",
        budget="약 3억원",
        filename="한국가스공사_erp.hwp",
    )
    second_chunk = _make_chunk(
        chunk_id="chunk-2",
        text="두번째 문단 " + ("B" * 400),
        budget="약 5억원",
        filename="한국가스공사_erp_2.hwp",
    )

    result = provider.generate(
        question="예산은 얼마야?",
        context_chunks=[first_chunk, second_chunk],
        history=[],
        generation_config={
            "max_context_chars": 120,
            "max_new_tokens": 64,
            "rewritten_query": "차세대 ERP 구축 사업의 예산은 얼마인가요?",
            "memory_summary": "이전 대화에서 차세대 ERP 구축 사업 개요를 확인했다.",
            "memory_slots": {
                "발주기관": "한국가스공사",
                "사업명": "차세대 ERP 구축",
                "관심속성": "예산",
            },
        },
        system_prompt="SYSTEM",
    )

    full_input = generator.last_call["full_input"]

    assert "[문서: 차세대 ERP 구축 | 한국가스공사 | 한국가스공사_erp.hwp]" in full_input
    assert "차세대 ERP 구축 사업의 예산은 얼마인가요?" in full_input
    assert "이전 대화에서 차세대 ERP 구축 사업 개요를 확인했다." in full_input
    assert "발주기관: 한국가스공사" in full_input
    assert "사업 금액=약 3억원" in full_input
    assert "핵심 요구사항" in full_input
    assert "## 작성 절차" in full_input
    assert "비교형 질문이면 기관/사업별로" in full_input
    assert "문서에 없는 항목" in full_input
    assert "두번째 문단" not in full_input
    assert generator.last_call["max_new_tokens"] == 64
    # 같은 문서의 청크는 문서 헤더 아래 그룹으로 묶이고, 인용 번호는 청크 본문에 유지된다.
    assert result.context.startswith("[문서: 차세대 ERP 구축 | 한국가스공사 | 한국가스공사_erp.hwp]")
    assert "두번째 문단" not in result.context
    assert "\n\n[1] 섹션=요구사항" in result.context
