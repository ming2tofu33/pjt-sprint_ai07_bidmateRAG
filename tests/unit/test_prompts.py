from bidmate_rag.config.prompts import SYSTEM_PROMPT, build_rag_user_prompt


def test_system_prompt_includes_partial_comparison_and_follow_up_rules() -> None:
    assert "질문이 여러 요청으로 이루어져 있으면 먼저 답변 단위를 나누세요." in SYSTEM_PROMPT
    assert "후속 질문에서" in SYSTEM_PROMPT
    assert "비교형 질문이면 기관/사업별로 정보를 분리해서 정리한 뒤" in SYSTEM_PROMPT
    assert "문서에 없는 항목" in SYSTEM_PROMPT


def test_build_rag_user_prompt_includes_structured_answer_guidance() -> None:
    prompt = build_rag_user_prompt(
        question="고려대학교와 광주과학기술원 사업을 비교해줘",
        context="[1] 샘플 컨텍스트",
    )

    assert "## 작성 절차" in prompt
    assert "질문을 먼저 해석하고, 필요한 답변 단위를 나누세요." in prompt
    assert "비교형 질문이면 기관/사업별로 정보를 분리해서 정리한 뒤 비교 결과를 쓰세요." in prompt
    assert "후속 질문이면 대화 이력에서 직전 대상과 기준을 먼저 해석하세요." in prompt
    assert "문서에 없는 항목" in prompt
