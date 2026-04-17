from bidmate_rag.providers.llm.openai_compat import OpenAICompatibleLLM
from bidmate_rag.schema import Chunk, RetrievedChunk


class _FakeResponse:
    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = type("Msg", (), {"content": content})()

    def __init__(self, content: str = "응답") -> None:
        self.choices = [self._Choice(content)]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": type("PromptDetails", (), {"cached_tokens": 0})(),
            },
        )()


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_call: dict | None = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def _make_retrieved_chunk() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="핵심 요구사항",
        text_with_meta="핵심 요구사항",
        char_count=7,
        section="요구사항",
        content_type="text",
        chunk_index=0,
        metadata={
            "사업명": "차세대 ERP 구축",
            "발주 기관": "한국가스공사",
            "파일명": "한국가스공사_erp.hwp",
            "사업 금액": "약 3억원",
        },
    )
    return RetrievedChunk(rank=1, score=0.95, chunk=chunk)


def test_openai_provider_builds_metadata_aware_context_block() -> None:
    client = _FakeClient()
    provider = OpenAICompatibleLLM(
        provider_name="openai",
        model_name="gpt-test",
        client=client,
    )

    result = provider.generate(
        question="예산은 얼마야?",
        context_chunks=[_make_retrieved_chunk()],
        history=[],
        generation_config={
            "max_context_chars": 2000,
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

    prompt = client.chat.completions.last_call["messages"][-1]["content"]

    assert "[문서: 차세대 ERP 구축 | 한국가스공사 | 한국가스공사_erp.hwp]" in prompt
    assert "차세대 ERP 구축 사업의 예산은 얼마인가요?" in prompt
    assert "이전 대화에서 차세대 ERP 구축 사업 개요를 확인했다." in prompt
    assert "발주기관: 한국가스공사" in prompt
    assert "사업 금액=약 3억원" in prompt
    assert "핵심 요구사항" in prompt
    assert "## 작성 절차" in prompt
    assert "질문을 먼저 해석" in prompt
    assert "문서에 없는 항목" in prompt
    # 청크 앞에 인용 번호 prefix가 붙는다 — LLM이 답변에서 [1], [2]로 인용할 수 있도록.
    assert result.context.startswith("[문서: 차세대 ERP 구축 | 한국가스공사 | 한국가스공사_erp.hwp]")
    assert "\n\n[1] 섹션=요구사항" in result.context
