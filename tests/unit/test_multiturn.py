from unittest.mock import MagicMock

from bidmate_rag.retrieval.multiturn import (
    extract_recent_agency_filter,
    rewrite_query_with_history,
)


def _make_mock_llm(
    rewritten_text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> MagicMock:
    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=rewritten_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    mock_llm.model_name = "gpt-5-mini"
    return mock_llm


def test_rewrite_query_with_history_replaces_follow_up_reference_with_recent_topic() -> None:
    rewritten, trace = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"
    assert trace["rewrite_reason"] == "rule_fallback"


def test_rule_only_rewrites_q063_style_implicit_followup_with_recent_topic() -> None:
    query = (
        "그럼 방금 말씀해주신 3초 이내 성능 응답 기준이 면제되는 예외 상황이 있다고 가정하겠습니다. "
        "만약 특정 조건이 만족할 때 해당 디스플레이 성능 기준이 적용되지 않는다면, "
        "파일 크기 측면과 동시 접속자 수 측면에서 각각 그 구체적인 예외 허용 기준은 무엇으로 설계되어 있나요?"
    )

    rewritten, trace = rewrite_query_with_history(
        query=query,
        chat_history=[
            {
                "role": "user",
                "content": "한국한의학연구원 통합정보시스템의 일반 온라인 페이지 디스플레이 요구시간 성능 목표는 어떻게 되나요?",
            },
            {
                "role": "assistant",
                "content": "시스템 성능 요구사항(PER-003)에 따라, 사용자가 일반 페이지를 요청한 시각으로부터 3초 이내에 화면에 완전히 디스플레이 되어야 합니다.",
            },
        ],
        agency_list=["한국한의학연구원"],
        mode="rule_only",
    )

    expected_prefix = "한국한의학연구원 통합정보시스템의 일반 온라인 페이지 디스플레이 요구시간 성능 목표"
    assert rewritten.startswith(f"{expected_prefix} ")
    assert rewritten.endswith(query)
    assert trace["rewrite_reason"] == "rule_fallback"
    assert trace["rewrite_applied"] is True


def test_rule_only_keeps_query_when_followup_discourse_already_has_explicit_agency() -> None:
    query = "그럼 한국한의학연구원 통합정보시스템 성능 기준 예외는 무엇인가요?"

    rewritten, trace = rewrite_query_with_history(
        query=query,
        chat_history=[
            {
                "role": "user",
                "content": "한국한의학연구원 통합정보시스템의 일반 온라인 페이지 디스플레이 요구시간 성능 목표는 어떻게 되나요?",
            },
            {
                "role": "assistant",
                "content": "시스템 성능 요구사항(PER-003)에 따라, 사용자가 일반 페이지를 요청한 시각으로부터 3초 이내에 화면에 완전히 디스플레이 되어야 합니다.",
            },
        ],
        agency_list=["한국한의학연구원"],
        mode="rule_only",
    )

    assert rewritten == query
    assert trace["rewrite_reason"] == "original"
    assert trace["rewrite_applied"] is False


def test_rewrite_query_with_history_uses_llm_for_implicit_followup() -> None:
    mock_llm = _make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준은?")

    rewritten, trace = rewrite_query_with_history(
        query="평가기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다."},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가기준은?"
    mock_llm.rewrite.assert_called_once()
    assert trace["rewrite_reason"] == "llm"
    assert trace["rewrite_applied"] is True


def test_rewrite_query_with_history_includes_slot_memory_in_llm_prompt() -> None:
    mock_llm = _make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준은?")

    rewritten, _trace = rewrite_query_with_history(
        query="평가기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다."},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
        slot_memory={
            "발주기관": "국민연금공단",
            "사업명": "차세대 ERP 사업",
            "관심속성": "평가기준",
        },
    )

    prompt = mock_llm.rewrite.call_args.args[0]
    assert "발주기관: 국민연금공단" in prompt
    assert "사업명: 차세대 ERP 사업" in prompt
    assert "관심속성: 평가기준" in prompt
    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가기준은?"


def test_rewrite_query_with_history_skips_llm_when_no_history() -> None:
    mock_llm = _make_mock_llm("무시되는 응답")

    rewritten, trace = rewrite_query_with_history(
        query="서버 구축 사업 찾아줘",
        chat_history=[],
        agency_list=[],
        llm=mock_llm,
    )

    assert rewritten == "서버 구축 사업 찾아줘"
    mock_llm.rewrite.assert_not_called()
    assert trace["rewrite_reason"] == "original"


def test_rewrite_query_with_history_falls_back_to_rule_based_on_llm_error() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "gpt-5-mini"
    mock_llm.rewrite.side_effect = Exception("timeout")

    rewritten, trace = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"
    assert trace["rewrite_reason"] == "rule_fallback"
    assert trace["rewrite_error"] == "timeout"


def test_extract_recent_agency_filter_prefers_latest_single_agency_from_history() -> None:
    agency_filter = extract_recent_agency_filter(
        chat_history=[
            {"role": "user", "content": "기초과학연구원 사업 알려줘"},
            {"role": "assistant", "content": "설명"},
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert agency_filter == {"발주 기관": "국민연금공단"}


def test_rewrite_query_with_history_passes_config_to_provider() -> None:
    mock_llm = _make_mock_llm("재작성 결과")

    rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
        max_completion_tokens=8000,
        timeout_seconds=60,
    )

    call_kwargs = mock_llm.rewrite.call_args.kwargs
    assert call_kwargs["max_tokens"] == 8000
    assert call_kwargs["timeout"] == 60


def test_rewrite_config_rejects_non_positive_values() -> None:
    """RewriteConfig는 max_completion_tokens/timeout_seconds를 양수로만 받는다."""
    import pytest
    from pydantic import ValidationError

    from bidmate_rag.config.settings import RewriteConfig

    with pytest.raises(ValidationError):
        RewriteConfig(max_completion_tokens=0)

    with pytest.raises(ValidationError):
        RewriteConfig(max_completion_tokens=-1)

    with pytest.raises(ValidationError):
        RewriteConfig(timeout_seconds=0)

    with pytest.raises(ValidationError):
        RewriteConfig(timeout_seconds=-5)

    # 정상값은 그대로 통과.
    cfg = RewriteConfig(max_completion_tokens=1, timeout_seconds=1)
    assert cfg.max_completion_tokens == 1
    assert cfg.timeout_seconds == 1


def test_rewrite_query_with_history_excludes_detailed_assistant_facts_from_prompt() -> None:
    mock_llm = _make_mock_llm(
        "한국한의학연구원 통합정보시스템 성능 기준을 초과하면 수행사는 어떻게 처리해야 하나요?"
    )

    rewrite_query_with_history(
        query="해당 성능 기준을 초과하면 수행사는 어떻게 처리해야 하나요?",
        chat_history=[
            {
                "role": "user",
                "content": "한국한의학연구원 통합정보시스템 성능 테스트 기준을 알려주세요.",
            },
            {
                "role": "assistant",
                "content": "성능 테스트 기준은 평균 2초 이내, 최대 지연은 5초 이내여야 합니다.",
            },
        ],
        agency_list=["한국한의학연구원"],
        llm=mock_llm,
        slot_memory={"발주기관": "한국한의학연구원"},
    )

    prompt = mock_llm.rewrite.call_args.args[0]
    assert "한국한의학연구원 통합정보시스템 성능 테스트 기준을 알려주세요." in prompt
    assert "평균 2초" not in prompt
    assert "최대 지연은 5초" not in prompt
    assert "현재 질문에 없는 새로운 사실" in prompt


import json
from unittest.mock import MagicMock

from bidmate_rag.providers.llm.base import RewriteResponse
from bidmate_rag.retrieval.multiturn import rewrite_query_with_history


def test_llm_rewrite_extracts_section_hint_from_json_response() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 평가 기준",
                "section_hint": "평가 기준",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="평가 기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가 기준"
    assert trace["section_hint"] == "평가 기준"
    assert trace["rewrite_reason"] == "llm"


def test_llm_rewrite_falls_back_when_json_invalid() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text="국민연금공단 차세대 ERP 사업의 평가 기준",  # plain text, no JSON
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="평가 기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가 기준"
    assert trace["section_hint"] is None


def test_llm_rewrite_section_hint_null_when_missing() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {"rewritten_query": "국민연금공단 사업 알려줘", "section_hint": None},
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    _, trace = rewrite_query_with_history(
        query="사업 알려줘",
        chat_history=[{"role": "user", "content": "국민연금공단 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert trace["section_hint"] is None


def test_llm_rewrite_flags_validation_failure_when_new_number_injected() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 사업기간 30일",
                "section_hint": "사업 일정",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="사업기간은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert "30일" not in rewritten
    assert trace["rewrite_validation"] == "failed"
    assert trace["rewrite_reason"] in ("rule_fallback", "original")


def test_llm_rewrite_flags_validation_failure_when_new_year_injected() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "2024년 국민연금공단 차세대 ERP 사업 예산",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert "2024년" not in rewritten
    assert trace["rewrite_validation"] == "failed"


def test_llm_rewrite_validation_passes_when_rewritten_only_adds_context_terms() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 예산",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="예산은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 예산"
    assert trace["rewrite_validation"] == "passed"
    assert trace["rewrite_reason"] == "llm"


def test_rewrite_query_with_history_clears_rejected_section_hint_after_rule_fallback() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "국민연금공단 차세대 ERP 사업의 예산 2024년",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"
    assert trace["rewrite_reason"] == "rule_fallback"
    assert trace["rewrite_validation"] == "failed"
    assert trace["section_hint"] is None


def test_rewrite_query_with_history_clears_rejected_section_hint_when_original_is_kept() -> None:
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=json.dumps(
            {
                "rewritten_query": "2024년 국민연금공단 차세대 ERP 사업 예산",
                "section_hint": "예산",
            },
            ensure_ascii=False,
        ),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    mock_llm.model_name = "gpt-5-mini"

    rewritten, trace = rewrite_query_with_history(
        query="예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "예산은?"
    assert trace["rewrite_reason"] == "original"
    assert trace["rewrite_validation"] == "failed"
    assert trace["section_hint"] is None
