"""CLI entrypoint for generating a markdown experiment report.

Usage::

    uv run bidmate-report --run-id bench-a1b2c3d4
    uv run bidmate-report --run-id bench-a1b2c3d4 \\
        --runs-dir artifacts/logs/runs \\
        --benchmarks-dir artifacts/logs/benchmarks \\
        --output-dir artifacts/reports
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from bidmate_rag.tracking.markdown_report import load_report_data, write_report


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="bidmate-report",
        description="Generate a Notion-friendly markdown report for an evaluation run.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="artifacts/logs/runs")
    parser.add_argument("--benchmarks-dir", default="artifacts/logs/benchmarks")
    parser.add_argument("--embeddings-dir", default="artifacts/logs/embeddings")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override experiment name (otherwise inferred from meta.json or parquet scan).",
    )
    args = parser.parse_args()

    data = load_report_data(
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        benchmarks_dir=args.benchmarks_dir,
        embeddings_dir=args.embeddings_dir,
        experiment_name=args.experiment_name,
    )
    out_path = write_report(data, output_dir=args.output_dir)
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
