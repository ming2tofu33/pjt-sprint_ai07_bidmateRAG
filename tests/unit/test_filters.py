from bidmate_rag.retrieval.filters import (
    extract_metadata_filters,
    extract_project_clues,
    extract_range_filters,
    extract_section_hint,
    is_comparison_query,
    should_fan_out_multi_source_query,
    should_boost_tables,
)


def test_extract_metadata_filters_prefers_exact_agency_match() -> None:
    filters = extract_metadata_filters(
        query="국민연금공단 이러닝시스템 요구사항을 정리해줘",
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert filters == {"발주 기관": "국민연금공단"}


def test_extract_metadata_filters_uses_domain_when_agency_absent() -> None:
    filters = extract_metadata_filters(
        query="교육 관련 사업 찾아줘",
        agency_list=["국민연금공단"],
    )

    assert filters == {"사업도메인": "교육/학습"}


def test_extract_range_and_section_filters() -> None:
    range_filters = extract_range_filters("2024년 5억 이상 사업의 요구사항 알려줘")

    assert range_filters == {"사업 금액": {"$gte": 500000000}, "공개연도": 2024}
    assert extract_section_hint("보안 요구사항 알려줘") == "보안"
    assert (
        extract_section_hint(
            "그 외의 개인 소유 일반 PC나 보조기억장치(USB 등)를 반입하거나 반출해도 되나요?"
    )
        == "보안"
    )
    assert extract_section_hint("이 사업 개요 알려줘") is None
    assert extract_section_hint("이 사업의 추진 배경이 뭐야?") == "사업개요"
    assert should_boost_tables("예산과 일정 표로 정리해줘") is True


def test_is_comparison_query_detects_comparison_and_shared_keywords() -> None:
    assert is_comparison_query("국민연금공단과 기초과학연구원 사업을 비교해줘") is True
    assert is_comparison_query("두 기관의 차액과 공통 요구사항을 각각 알려줘") is True
    assert is_comparison_query("각 기관 요구사항을 각각 정리해줘") is True
    assert is_comparison_query("요구사항을 각각 정리해줘") is False
    assert is_comparison_query("국민연금공단 요구사항 알려줘") is False


def test_should_fan_out_multi_source_query_detects_listing_intent() -> None:
    assert (
        should_fan_out_multi_source_query(
            "(사)벤처기업협회와 (사)한국대학스포츠협의회의 사업에서 항목을 각각 나열해 주세요."
        )
        is True
    )
    assert should_fan_out_multi_source_query("각 사업의 예산을 순서대로 정리해줘") is True
    assert should_fan_out_multi_source_query("국민연금공단 요구사항 알려줘") is False


def test_extract_project_clues_reads_quoted_project_names() -> None:
    query = '"호계체육관 예약 시스템 구축", "평택시 버스정보시스템(BIS) 구축" 사업 중 예산을 비교해줘'

    assert extract_project_clues(query) == [
        "호계체육관 예약 시스템 구축",
        "평택시 버스정보시스템(BIS) 구축",
    ]
