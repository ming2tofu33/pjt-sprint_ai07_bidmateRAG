from bidmate_rag.retrieval.memory import (
    ConversationMemory,
    build_rewrite_safe_slot_memory,
)


def test_memory_keeps_recent_turns_and_extracts_slots() -> None:
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["교육부", "국민연금공단"],
    )

    state = memory.build(
        [
            {"role": "user", "content": "교육부 클라우드 전환 사업 알려줘"},
            {"role": "assistant", "content": "예산은 3억원입니다."},
            {"role": "user", "content": "평가기준도 정리해줘"},
            {"role": "assistant", "content": "기술평가와 배점 항목이 있습니다."},
            {"role": "user", "content": "일정도 알려줘"},
        ],
        current_question="일정도 알려줘",
        rewritten_query="교육부 클라우드 전환 사업의 일정",
    )

    assert len(state["recent_turns"]) == 4
    assert state["summary_buffer"]
    assert state["slot_memory"]["발주기관"] == "교육부"
    assert state["slot_memory"]["사업명"] == "교육부 클라우드 전환 사업"
    assert state["slot_memory"]["예산"] == "3억원"
    assert state["slot_memory"]["관심속성"] == "일정"


def test_memory_supports_legacy_history_shape() -> None:
    memory = ConversationMemory(max_recent_turns=2, max_summary_chars=80)

    state = memory.build(
        [
            {
                "user": "국민연금공단 차세대 ERP 사업 알려줘",
                "assistant": "예산은 5억원입니다.",
            }
        ],
        current_question="예산은?",
        rewritten_query="국민연금공단 차세대 ERP 사업의 예산",
    )

    assert len(state["recent_turns"]) == 2
    assert state["slot_memory"]["예산"] == "5억원"


def test_build_rewrite_safe_slot_memory_keeps_only_context_slots() -> None:
    rewrite_slots = build_rewrite_safe_slot_memory(
        {
            "발주기관": "교육부",
            "사업명": "교육부 클라우드 전환 사업",
            "예산": "3억원",
            "일정": "2024년 6월까지",
            "평가기준": "기술평가 80점",
            "관심속성": "예산",
        }
    )

    assert rewrite_slots == {
        "발주기관": "교육부",
        "사업명": "교육부 클라우드 전환 사업",
        "관심속성": "예산",
    }
