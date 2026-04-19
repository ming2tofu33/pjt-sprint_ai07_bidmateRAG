from bidmate_rag.config.prompts import SYSTEM_PROMPT, build_rag_user_prompt


def test_system_prompt_includes_partial_comparison_and_follow_up_rules() -> None:
    assert "질문이 여러 요청으로 이루어져 있으면" in SYSTEM_PROMPT
    assert "후속 질문에서" in SYSTEM_PROMPT
    assert "비교형 질문이면 기관/사업별로 정보를 분리해서 정리한 뒤" in SYSTEM_PROMPT
    assert "문서에 없는 항목" in SYSTEM_PROMPT
    assert "질문에 직접 답할 수 있는 근거가 있으면 먼저 답하고" in SYSTEM_PROMPT
    assert "첫 문장에서 바로 값/여부/항목을 먼저 말하세요." in SYSTEM_PROMPT
    assert "해석이 정말 불가능할 때만 부족하다고 쓰세요." in SYSTEM_PROMPT
    assert "질문이 요구한 비교 결론" in SYSTEM_PROMPT
    assert "여러 항목 중 일부만 확인되더라도 확인된 항목은 확정적으로 답하세요." in SYSTEM_PROMPT
    assert "`- 비교 결론:` bullet을 먼저 쓰고" in SYSTEM_PROMPT
    assert "단일 최종 판정을 묻는 경우에는 첫 bullet에서 그 판정만 간단히 답하세요." in SYSTEM_PROMPT
    assert "예/아니오 질문은 첫 bullet을 반드시 `예.` 또는 `아니요.`로 시작하세요." in SYSTEM_PROMPT
    assert "조건을 만족하는 대상만 먼저 답하고" in SYSTEM_PROMPT
    assert "기관명/사업명/파일명처럼 답변 대상이 명시되면 그 대상만 우선 사용하세요." in SYSTEM_PROMPT
    assert "`항목명: 없음(문서 미기재)`" in SYSTEM_PROMPT
    assert "문서 근거가 질문 전제와 반대이면" in SYSTEM_PROMPT


def test_build_rag_user_prompt_includes_structured_answer_guidance() -> None:
    prompt = build_rag_user_prompt(
        question="고려대학교와 광주과학기술원 사업을 비교해줘",
        context="[1] 샘플 컨텍스트",
    )

    assert "## 작성 절차" in prompt
    assert "질문을 먼저 해석하고, 필요한 답변 단위를 나누세요." in prompt
    assert "비교형 질문이면 기관/사업별로 정보를 분리해서 정리하고, 첫 bullet 또는 마지막 요약에 비교 결론을 분명히 쓰세요." in prompt
    assert "후속 질문이면 대화 이력과 메모리에서 직전 대상과 기준을 먼저 해석하고, 해석된 대상을 기준으로 바로 답하세요." in prompt
    assert "문서에 없는 항목" in prompt
    assert "답할 수 있는 근거가 있으면 먼저 답하세요." in prompt
    assert "질문이 여러 슬롯이면 `- 항목명: 값` 형태로 나누어 작성하세요." in prompt
    assert "비교형 질문이면 첫 bullet에 `- 비교 결론:`을 작성하세요." in prompt
    assert "일부만 확인되면 확인된 내용부터 확정적으로 답하고" in prompt
    assert "단일 최종 판정(가장 큼/작음, 예/아니오, 어느 기관)을 묻는 경우" in prompt
    assert "예/아니오 질문이면 첫 bullet을 `- 예.` 또는 `- 아니요.`로 시작하세요." in prompt
    assert "조건을 만족하는 기관/사업을 묻는 질문이면 첫 bullet에 정답 대상만 적으세요." in prompt
    assert "기관명/사업명/파일명처럼 답변 대상이 명시되면 그 대상만 우선 사용하고" in prompt
    assert "`항목명: 없음(문서 미기재)` 형식으로 직접 적으세요." in prompt
    assert "예/아니오 질문의 전제가 문서와 반대이면 첫 bullet에서 반대 결론과 실제 사실을 함께 쓰세요." in prompt
