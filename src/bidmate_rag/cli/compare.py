"""CLI for comparing multiple benchmark runs.

Usage::

    # 같은 실험명의 모든 run 비교
    uv run bidmate-compare --experiment generation-compare

    # 임의 run_id 비교 (다른 experiment끼리도 가능)
    uv run bidmate-compare --run-ids bench-aaba1e89 bench-9ae1683f bench-d24fc998

    # 파일로 저장
    uv run bidmate-compare --experiment generation-compare -o /tmp/comp.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from bidmate_rag.tracking.comparison import (
    load_runs_for_comparison,
    render_comparison_markdown,
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="bidmate-compare",
        description="Compare multiple benchmark runs across metrics.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        "-e",
        help="experiment_name으로 모든 run 비교 (parquet 1개 통째로)",
    )
    group.add_argument(
        "--run-ids",
        "-r",
        nargs="+",
        help="run_id 명시 (parquet 디렉토리 전체에서 검색)",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="artifacts/logs/benchmarks",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="마크다운을 파일로 저장 (없으면 stdout)",
    )
    args = parser.parse_args()

    data = load_runs_for_comparison(
        run_ids=args.run_ids,
        experiment_name=args.experiment,
        benchmarks_dir=args.benchmarks_dir,
    )
    md = render_comparison_markdown(data)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"report: {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
