"""Build a SQLite store for calculation-oriented structured fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from bidmate_rag.storage.calculation_store import CalculationStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calculation SQLite store from cleaned documents parquet.")
    parser.add_argument(
        "--input",
        default="data/processed/cleaned_documents.parquet",
        help="Input cleaned_documents parquet path.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sql/calculation_store.sqlite3",
        help="Output SQLite path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 parquet를 찾을 수 없습니다: {input_path}")

    store = CalculationStore.from_parquet(input_path, db_path=output_path)
    try:
        print(f"input: {input_path}")
        print(f"output: {output_path}")
        print(f"rows(all): {store.count(canonical_only=False)}")
        print(f"rows(canonical_only): {store.count(canonical_only=True)}")
        rows = store.list_facts(canonical_only=False)
        print(f"allocated_budget: {sum(1 for row in rows if row.allocated_budget is not None)}")
        print(f"estimated_price: {sum(1 for row in rows if row.estimated_price is not None)}")
        print(f"planned_price: {sum(1 for row in rows if row.planned_price is not None)}")
        print(f"base_amount: {sum(1 for row in rows if row.base_amount is not None)}")
        print(f"contract_amount: {sum(1 for row in rows if row.contract_amount is not None)}")
        print(f"project_budget_labeled: {sum(1 for row in rows if row.project_budget_labeled is not None)}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
