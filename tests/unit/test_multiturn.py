from bidmate_rag.retrieval.multiturn import (
    extract_recent_agency_filter,
    rewrite_query_with_history,
)


def test_rewrite_query_with_history_replaces_follow_up_reference_with_recent_topic() -> None:
    rewritten = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"


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
