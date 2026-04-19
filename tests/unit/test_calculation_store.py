import pandas as pd

from bidmate_rag.storage.calculation_store import (
    COL_AGENCY,
    COL_BID_END,
    COL_BID_START,
    COL_BODY_MARKDOWN,
    COL_BUDGET,
    COL_CANONICAL_FILE,
    COL_CLEANED_CHARS,
    COL_CLEANED_TEXT,
    COL_DOMAIN,
    COL_FILE_NAME,
    COL_FILE_TYPE,
    COL_INGEST_ENABLED,
    COL_INGEST_FILE,
    COL_IS_DUPLICATE,
    COL_NOTICE_ID,
    COL_ORIGINAL_AGENCY,
    COL_PUBLIC_DATE,
    COL_PUBLIC_YEAR,
    COL_RESOLVED_AGENCY,
    COL_TITLE,
    CalculationStore,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COL_NOTICE_ID: "2024-001",
                COL_TITLE: "차세대 포털 구축",
                COL_BUDGET: 1_000_000_000,
                COL_AGENCY: "고려대학교",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-08-12 11:00:00",
                COL_FILE_TYPE: "pdf",
                COL_FILE_NAME: "korea.pdf",
                COL_CANONICAL_FILE: "korea.pdf",
                COL_INGEST_FILE: "korea.pdf",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "고려대학교",
                COL_ORIGINAL_AGENCY: "고려대학교",
                COL_CLEANED_CHARS: 5000,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: (
                    "배정예산: 850,000,000원\n"
                    "추정가격: 920,000,000원\n"
                    "예정가격: 910,000,000원\n"
                    "기초금액: 900,000,000원\n"
                    "계약금액: 830,000,000원\n"
                    "사업예산: 1,000,000,000원"
                ),
            },
            {
                COL_NOTICE_ID: "2024-002",
                COL_TITLE: "학사 시스템 기능개선",
                COL_BUDGET: 300_000_000,
                COL_AGENCY: "광주과학기술원",
                COL_PUBLIC_DATE: "2024-06-01 00:00:00",
                COL_BID_START: "2024-06-10 09:00:00",
                COL_BID_END: "2024-06-20 18:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "gist.hwp",
                COL_CANONICAL_FILE: "gist.hwp",
                COL_INGEST_FILE: "gist.hwp",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "광주과학기술원",
                COL_ORIGINAL_AGENCY: "광주과학기술원",
                COL_CLEANED_CHARS: 4200,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: (
                    "배정 예산 250,000,000원\n"
                    "추정 가격 280,000,000원\n"
                    "예정 가격 275,000,000원\n"
                    "기초 금액 270,000,000원\n"
                    "계약 금액 260,000,000원\n"
                    "사업예산: 300,000,000원"
                ),
            },
            {
                COL_NOTICE_ID: "2024-003",
                COL_TITLE: "학사 시스템 기능개선 사본",
                COL_BUDGET: 300_000_000,
                COL_AGENCY: "광주과학기술원",
                COL_PUBLIC_DATE: "2024-06-01 00:00:00",
                COL_BID_START: "2024-06-10 09:00:00",
                COL_BID_END: "2024-06-20 18:00:00",
                COL_FILE_TYPE: "hwp",
                COL_FILE_NAME: "gist-dup.hwp",
                COL_CANONICAL_FILE: "gist.hwp",
                COL_INGEST_FILE: "gist.hwp",
                COL_IS_DUPLICATE: True,
                COL_INGEST_ENABLED: False,
                COL_RESOLVED_AGENCY: "광주과학기술원",
                COL_ORIGINAL_AGENCY: "광주과학기술원",
                COL_CLEANED_CHARS: 4100,
                COL_DOMAIN: "교육",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "사업예산: 300,000,000원",
            },
        ]
    )


def test_calculation_store_filters_duplicates_by_default() -> None:
    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(_sample_frame())

        assert store.count(canonical_only=False) == 3
        assert store.count(canonical_only=True) == 2

        facts = store.list_facts()
        assert [fact.file_name for fact in facts] == ["korea.pdf", "gist.hwp"]
    finally:
        store.close()


def test_calculation_store_extracts_budget_kinds() -> None:
    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(_sample_frame())

        fact = store.get_fact("korea.pdf")
        assert fact is not None
        assert fact.allocated_budget == 850_000_000
        assert fact.estimated_price == 920_000_000
        assert fact.planned_price == 910_000_000
        assert fact.base_amount == 900_000_000
        assert fact.contract_amount == 830_000_000
        assert fact.project_budget_labeled == 1_000_000_000
    finally:
        store.close()


def test_calculation_store_budget_comparison_and_aggregation() -> None:
    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(_sample_frame())

        comparison = store.compare_budget("korea.pdf", "gist.hwp")
        assert comparison.difference == 700_000_000
        assert comparison.ratio == round(1_000_000_000 / 300_000_000, 6)

        assert store.sum_budget(year=2024) == 1_300_000_000
        assert store.average_budget(year=2024) == 650_000_000
        assert store.apply_budget_ratio("korea.pdf", 0.3) == 300_000_000
    finally:
        store.close()


def test_calculation_store_supports_budget_kind_aggregation() -> None:
    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(_sample_frame())

        comparison = store.compare_budget("korea.pdf", "gist.hwp", budget_kind="allocated_budget")
        assert comparison.difference == 600_000_000
        assert store.sum_budget(year=2024, budget_kind="estimated_price") == 1_200_000_000
        assert store.sum_budget(year=2024, budget_kind="planned_price") == 1_185_000_000
        assert store.average_budget(year=2024, budget_kind="base_amount") == 585_000_000
        assert store.average_budget(year=2024, budget_kind="contract_amount") == 545_000_000
        assert store.budget_by_kind("korea.pdf", "allocated_budget") == 850_000_000
    finally:
        store.close()


def test_calculation_store_supports_bid_window_lookup() -> None:
    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(_sample_frame())

        fact = store.get_fact("2024-001")
        assert fact is not None
        assert fact.file_name == "korea.pdf"
        assert store.bid_window_days("korea.pdf") == 38.0
    finally:
        store.close()


def test_calculation_store_extracts_amount_when_number_precedes_label_in_table_row() -> None:
    frame = pd.DataFrame(
        [
            {
                COL_NOTICE_ID: "2024-201",
                COL_TITLE: "표 패턴 테스트",
                COL_BUDGET: 0,
                COL_AGENCY: "기관A",
                COL_PUBLIC_DATE: "2024-07-01 00:00:00",
                COL_BID_START: "2024-07-05 11:00:00",
                COL_BID_END: "2024-07-10 11:00:00",
                COL_FILE_TYPE: "pdf",
                COL_FILE_NAME: "table.pdf",
                COL_CANONICAL_FILE: "table.pdf",
                COL_INGEST_FILE: "table.pdf",
                COL_IS_DUPLICATE: False,
                COL_INGEST_ENABLED: True,
                COL_RESOLVED_AGENCY: "기관A",
                COL_ORIGINAL_AGENCY: "기관A",
                COL_CLEANED_CHARS: 100,
                COL_DOMAIN: "테스트",
                COL_PUBLIC_YEAR: 2024,
                COL_BODY_MARKDOWN: "",
                COL_CLEANED_TEXT: "1,100,000,000원 | 예정가격\n900,000,000원 | 기초금액",
            }
        ]
    )

    store = CalculationStore.create()
    try:
        store.rebuild_from_frame(frame)
        fact = store.get_fact("table.pdf")
        assert fact is not None
        assert fact.planned_price == 1_100_000_000
        assert fact.base_amount == 900_000_000
    finally:
        store.close()
