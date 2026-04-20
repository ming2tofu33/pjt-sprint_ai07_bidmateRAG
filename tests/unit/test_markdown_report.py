"""Tests for tracking/markdown_report.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

import bidmate_rag.tracking.markdown_report as markdown_report_module
from bidmate_rag.tracking.markdown_report import (
    load_report_data,
    render_markdown,
    write_report,
)


def _make_fixture(
    tmp_path: Path,
    *,
    with_meta: bool = True,
    with_embedding: bool = True,
    judge_skipped: bool = False,
    llm_model: str = "gpt-5-mini",
    notes_path: str | None = None,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    runs_dir = tmp_path / "runs"
    benchmarks_dir = tmp_path / "benchmarks"
    embeddings_dir = tmp_path / "embeddings"
    runs_dir.mkdir()
    benchmarks_dir.mkdir()
    embeddings_dir.mkdir()

    run_id = "bench-test1234"
    exp_name = "test-exp"

    # JSONL with two questions
    jsonl_path = runs_dir / f"{run_id}.jsonl"
    rows = rows or [
        {
            "question_id": "q1",
            "question": "Q1",
            "scenario": "openai",
            "run_id": run_id,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "answer": "A1",
            "retrieved_chunks": [],
            "latency_ms": 1500.0,
            "token_usage": {
                "prompt": 100,
                "completion": 50,
                "total": 150,
                "rewrite_prompt": 20,
                "rewrite_completion": 10,
                "rewrite_total": 30,
            },
            "cost_usd": 0.0012,
            "debug": {"rewrite_cost_usd": 0.0001},
            "judge_scores": {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.7,
                "context_recall": 0.8,
            },
        },
        {
            "question_id": "q2",
            "question": "Q2",
            "scenario": "openai",
            "run_id": run_id,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "answer": "A2",
            "retrieved_chunks": [],
            "latency_ms": 2500.0,
            "token_usage": {
                "prompt": 200,
                "completion": 100,
                "total": 300,
                "rewrite_prompt": 30,
                "rewrite_completion": 20,
                "rewrite_total": 50,
            },
            "cost_usd": 0.0024,
            "debug": {"rewrite_cost_usd": 0.0003},
            "judge_scores": {},
        },
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    # Parquet summary
    summary_row = {
        "experiment_name": exp_name,
        "run_id": run_id,
        "scenario": "openai",
        "provider_label": "openai:gpt-5-mini",
        "num_samples": 2,
        "avg_latency_ms": 2000.0,
        "generation_cost_usd": 0.0036,
        "rewrite_cost_usd": 0.0004,
        "judge_cost_usd": 0.0008,
        "total_cost_usd": 0.0048,
        "hit_rate@5": 0.85,
        "mrr": 0.72,
        "ndcg@5": 0.79,
        "faithfulness": 0.9,
        "answer_relevance": 0.85,
        "context_precision": 0.7,
        "context_recall": 0.8,
    }
    pd.DataFrame([summary_row]).to_parquet(
        benchmarks_dir / f"{exp_name}.parquet", index=False
    )

    # meta.json
    if with_meta:
        meta = {
            "run_id": run_id,
            "experiment_name": exp_name,
            "timestamp_utc": "2026-04-09T05:32:11+00:00",
            "timestamp_kst": "2026-04-09 14:32:11",
            "git": {"commit": "abc1234def", "commit_short": "abc1234", "branch": "main", "dirty": False},
            "configs": {
                "base": "configs/base.yaml",
                "provider": "configs/providers/openai_gpt5mini.yaml",
                "experiment": "configs/experiments/test-exp.yaml",
            },
            "config_snapshot": {
                "project": {
                    "default_chunk_size": 800,
                    "default_chunk_overlap": 100,
                    "default_retrieval_top_k": 5,
                },
                "provider": {
                    "provider": "openai",
                    "model": llm_model,
                    "embedding_model": "text-embedding-3-small",
                    "scenario": "openai",
                    "collection_name": "test-collection",
                },
                "experiment": {
                    "name": exp_name,
                    "chunk_size": 800,
                    "chunk_overlap": 100,
                    "retrieval_top_k": 5,
                },
            },
            "eval_path": "data/eval/eval_v1/eval_batch_02.csv",
            "collection_name": "test-collection",
            "judge_skipped": judge_skipped,
            "generation_cost_usd": 0.0036,
            "rewrite_cost_usd": 0.0004,
            "judge_cost_usd": 0.0 if judge_skipped else 0.0008,
            "total_cost_usd": 0.0040 if judge_skipped else 0.0048,
            "judge_total_cost_usd": 0.0 if judge_skipped else 0.0008,
            "judge_total_tokens": 0 if judge_skipped else 400,
        }
        if notes_path is not None:
            meta["notes_path"] = notes_path
        (runs_dir / f"{run_id}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Embedding meta
    if with_embedding:
        emb_meta = {
            "collection_name": "test-collection",
            "embedding_model": "text-embedding-3-small",
            "total_tokens": 50_000,
            "total_cost_usd": 0.001,
            "num_chunks": 100,
            "built_at": "2026-04-09T03:12:08+00:00",
        }
        (embeddings_dir / "test-collection.json").write_text(
            json.dumps(emb_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return tmp_path


def test_load_report_data_full(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    assert data.experiment_name == "test-exp"
    assert len(data.results) == 2
    assert data.summary_row["hit_rate@5"] == 0.85
    assert data.embedding_meta is not None
    assert data.embedding_meta["total_cost_usd"] == 0.001
    assert data.meta["judge_total_cost_usd"] == 0.0008


def test_load_report_data_reads_notes_from_meta(tmp_path):
    notes_path = tmp_path / "notes" / "example-budget-quality.yaml"
    notes_path.parent.mkdir()
    notes_path.write_text(
        (
            "title: budget-metadata-context\n"
            "overview: 예산 메타데이터를 답변에 반영하는 실험\n"
            "hypothesis:\n"
            "  - 예산이 metadata에만 존재하는 문서도 답변 가능해질 것이다.\n"
            "changes:\n"
            "  - LLM 컨텍스트 헤더에 사업 금액/공개연도/기관유형 추가\n"
            "expected_outcome:\n"
            "  - 예산 질문 무응답 감소\n"
            "next_actions:\n"
            "  - judge 기준으로 실패 유형 재분류\n"
            "failure_cases:\n"
            "  - question_id: q2\n"
            "    why_watch: retrieval miss\n"
        ),
        encoding="utf-8",
    )
    _make_fixture(tmp_path, notes_path=str(notes_path))

    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    assert data.experiment_notes is not None
    assert data.experiment_notes["title"] == "budget-metadata-context"
    assert data.experiment_notes["overview"] == "예산 메타데이터를 답변에 반영하는 실험"
    assert data.experiment_notes["failure_cases"][0]["question_id"] == "q2"


def test_load_report_data_resolves_relative_notes_path_from_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    fake_repo = tmp_path / "fake-repo"
    fake_module_path = fake_repo / "src" / "bidmate_rag" / "tracking" / "markdown_report.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("# fake module path for repo-root resolution\n", encoding="utf-8")

    relative_notes = Path("configs/experiments/notes/relative-budget-notes.yaml")
    notes_file = fake_repo / relative_notes
    notes_file.parent.mkdir(parents=True)
    notes_file.write_text(
        "title: relative-notes-title\noverview: 상대 경로 notes 파일\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(markdown_report_module, "__file__", str(fake_module_path))
    _make_fixture(tmp_path, notes_path=str(relative_notes))

    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    assert data.experiment_notes is not None
    assert data.experiment_notes["title"] == "relative-notes-title"
    assert data.experiment_notes["overview"] == "상대 경로 notes 파일"


def test_load_report_data_handles_malformed_notes_yaml_with_warning(
    tmp_path: Path, caplog
) -> None:
    notes_path = tmp_path / "broken-notes.yaml"
    notes_path.write_text("title: [unterminated\n", encoding="utf-8")
    _make_fixture(tmp_path, notes_path=str(notes_path))

    with caplog.at_level(logging.WARNING):
        data = load_report_data(
            run_id="bench-test1234",
            runs_dir=tmp_path / "runs",
            benchmarks_dir=tmp_path / "benchmarks",
            embeddings_dir=tmp_path / "embeddings",
        )

    md = render_markdown(data)

    assert data.experiment_notes is None
    assert "Experiment notes YAML could not be parsed" in caplog.text
    assert "bench-test1234" in md


def test_render_markdown_autofills_notes_bullets_and_failure_case(tmp_path):
    notes_path = tmp_path / "notes.yaml"
    notes_path.write_text(
        (
            "title: budget-metadata-context\n"
            "overview: 예산이 metadata에만 존재하는 문서를 다루는 실험\n"
            "hypothesis:\n"
            "  - 예산이 metadata에만 존재하는 문서도 답변 가능해질 것이다.\n"
            "  - 비교형 질문의 근거 회수가 늘어날 것이다.\n"
            "changes:\n"
            "  - LLM 컨텍스트 헤더에 사업 금액/공개연도/기관유형 추가\n"
            "  - 다문서 비교 질문에서 기관별 retrieval merge 추가\n"
            "expected_outcome:\n"
            "  - 예산 질문 무응답 감소\n"
            "  - 비교형 질문 retrieval hit 개선\n"
            "next_actions:\n"
            "  - judge 기준으로 실패 유형 재분류\n"
            "failure_cases:\n"
            "  - question_id: q2\n"
            "    why_watch: retrieval miss\n"
        ),
        encoding="utf-8",
    )
    _make_fixture(tmp_path, notes_path=str(notes_path))
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    md = render_markdown(data)
    assert "budget-metadata-context" in md
    assert "예산이 metadata에만 존재하는 문서도 답변 가능해질 것이다." in md
    assert "비교형 질문의 근거 회수가 늘어날 것이다." in md
    assert "LLM 컨텍스트 헤더에 사업 금액/공개연도/기관유형 추가" in md
    assert "예산 질문 무응답 감소" in md
    assert "judge 기준으로 실패 유형 재분류" in md
    assert "### 실패 사례 1" in md
    assert "question_id: q2" in md or "Q2" in md
    assert "retrieval miss" in md


def test_render_markdown_handles_missing_notes_file(tmp_path):
    missing_notes = tmp_path / "missing-notes.yaml"
    _make_fixture(tmp_path, notes_path=str(missing_notes))
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    md = render_markdown(data)
    assert data.experiment_notes is None
    assert "bench-test1234" in md
    assert "### 실패 사례 1" in md
    assert "Q2" in md


def test_render_markdown_without_notes_keeps_manual_prompts(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    md = render_markdown(data)
    assert data.experiment_notes is None
    assert "실험 목적/배경을 입력하세요" in md
    assert "왜 이 변경을 했는지:" in md
    assert "무엇이 좋아질 거라고 봤는지:" in md
    assert "변경 내용을 입력하세요" in md
    assert "기대 결과를 입력하세요" in md
    assert "다음 실험에서 무엇을 바꿀지:" in md
    assert "유지할 것:" in md
    assert "버릴 것:" in md


def test_render_markdown_includes_key_sections(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # 노션 속성 영역
    assert "📋 노션 속성" in md
    assert "bench-test1234" in md
    assert "test-exp" in md
    # 자동 생성 본문
    assert "🤖 자동 생성 본문" in md
    assert "Hit Rate@5" in md
    assert "0.8500" in md  # hit_rate
    # 사람 작성 영역
    assert "✍️ 사람이 작성하는 영역" in md
    assert "## 1. 가설" in md
    # 비용 표시
    assert "0.0036" in md  # generation cost
    assert "0.0004" in md  # rewrite cost
    assert "0.0010" in md  # embedding cost
    assert "0.0008" in md  # judge cost
    assert "0.0058" in md  # total cost including rewrite + embedding
    assert "Rewrite Cost (USD)" in md
    assert "Rewrite Prompt Tokens" in md
    assert "Rewrite Completion Tokens" in md
    assert "Rewrite Total Tokens" in md
    assert "| Total Tokens | 530 |" in md
    # 본문 표 cost 명칭이 노션 속성과 동일하게 "Cost (USD)"로 통일
    assert "**Cost (USD)**" in md
    # 토큰 합계
    assert "300" in md or "150" in md
    # git
    assert "abc1234" in md
    # gpt-5-mini는 reasoning 주의 문구가 표시되어야 함
    assert "gpt-5 계열은 reasoning tokens" in md


def test_fallback_failure_cases_prefer_weak_signals_deterministically(tmp_path):
    rows = [
        {
            "question_id": "q9",
            "question": "strong answer",
            "scenario": "openai",
            "run_id": "bench-test1234",
            "answer": "strong",
            "retrieved_chunks": [{"chunk_id": "c1"}],
            "latency_ms": 120.0,
            "cost_usd": 0.001,
            "judge_scores": {
                "faithfulness": 0.95,
                "answer_relevance": 0.93,
            },
        },
        {
            "question_id": "q2",
            "question": "error case",
            "scenario": "openai",
            "run_id": "bench-test1234",
            "answer": "failed",
            "retrieved_chunks": [{"chunk_id": "c2"}],
            "latency_ms": 900.0,
            "cost_usd": 0.01,
            "error": "timeout while generating answer",
            "judge_scores": {},
        },
        {
            "question_id": "q7",
            "question": "retrieval miss",
            "scenario": "openai",
            "run_id": "bench-test1234",
            "answer": "miss",
            "retrieved_chunks": [],
            "latency_ms": 1500.0,
            "cost_usd": 0.02,
            "judge_scores": {},
        },
        {
            "question_id": "q5",
            "question": "low score",
            "scenario": "openai",
            "run_id": "bench-test1234",
            "answer": "weak",
            "retrieved_chunks": [{"chunk_id": "c3"}],
            "latency_ms": 2000.0,
            "cost_usd": 0.03,
            "judge_scores": {
                "faithfulness": 0.2,
                "answer_relevance": 0.3,
                "context_precision": 0.4,
                "context_recall": 0.25,
            },
        },
    ]
    _make_fixture(tmp_path, rows=rows, judge_skipped=True)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    md = render_markdown(data)
    assert "### 실패 사례 1" in md
    assert "- 질문 ID: q2" in md
    assert "timeout while generating answer" in md
    assert "### 실패 사례 2" in md
    assert "- 질문 ID: q7" in md
    assert "retrieved_chunks가 비어 있음" in md
    assert "- 질문 ID: q5" not in md


def test_render_markdown_handles_missing_embedding(tmp_path):
    _make_fixture(tmp_path, with_embedding=False)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "임베딩 비용 미수집" in md
    assert data.embedding_meta is None


def test_render_markdown_hides_rewrite_rows_when_unused(tmp_path):
    rows = [
        {
            "question_id": "q1",
            "question": "Q1",
            "scenario": "openai",
            "run_id": "bench-test1234",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "answer": "A1",
            "retrieved_chunks": [],
            "latency_ms": 1500.0,
            "token_usage": {"prompt": 100, "completion": 50, "total": 150},
            "cost_usd": 0.0012,
            "debug": {"rewrite_cost_usd": 0.0},
            "judge_scores": {},
        }
    ]
    _make_fixture(tmp_path, rows=rows)
    (tmp_path / "runs" / "bench-test1234.meta.json").write_text(
        json.dumps(
            {
                **json.loads((tmp_path / "runs" / "bench-test1234.meta.json").read_text(encoding="utf-8")),
                "rewrite_cost_usd": 0.0,
                "total_cost_usd": 0.0044,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "experiment_name": "test-exp",
                "run_id": "bench-test1234",
                "scenario": "openai",
                "provider_label": "openai:gpt-5-mini",
                "num_samples": 1,
                "avg_latency_ms": 1500.0,
                "generation_cost_usd": 0.0012,
                "rewrite_cost_usd": 0.0,
                "judge_cost_usd": 0.0008,
                "total_cost_usd": 0.0020,
                "hit_rate@5": 0.85,
                "mrr": 0.72,
                "ndcg@5": 0.79,
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.7,
                "context_recall": 0.8,
            }
        ]
    ).to_parquet(tmp_path / "benchmarks" / "test-exp.parquet", index=False)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )

    md = render_markdown(data)
    assert "Rewrite Prompt Tokens" not in md
    assert "Rewrite Completion Tokens" not in md
    assert "Rewrite Total Tokens" not in md
    assert "Rewrite Cost (USD)" not in md


def test_render_markdown_handles_missing_meta(tmp_path):
    _make_fixture(tmp_path, with_meta=False)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # meta가 없어도 렌더링은 성공해야 함
    assert "bench-test1234" in md
    assert "test-exp" in md
    # config 정보가 비어있어야 함
    assert "(미기록)" in md or "configs/base.yaml" not in md


def test_write_report_creates_file(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    out = write_report(data, output_dir=tmp_path / "reports")
    assert out.exists()
    # 사람 친화 파일명: YYYY-MM-DD_HHMM_{exp}_{model}.md
    # fixture timestamp_kst="2026-04-09 14:32:11", llm_model 기본 "gpt-5-mini"
    assert out.name == "2026-04-09_1432_test-exp_gpt-5-mini.md"
    assert "📋 노션 속성" in out.read_text(encoding="utf-8")


def test_write_report_filename_for_non_gpt5_model(tmp_path):
    _make_fixture(tmp_path, llm_model="gpt-4o-mini")
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    out = write_report(data, output_dir=tmp_path / "reports")
    assert out.name == "2026-04-09_1432_test-exp_gpt-4o-mini.md"


def test_write_report_appends_suffix_on_collision(tmp_path):
    """같은 분에 두 번 돌리면 _2, _3 suffix가 자동으로 붙어야 함."""
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    out1 = write_report(data, output_dir=tmp_path / "reports")
    out2 = write_report(data, output_dir=tmp_path / "reports")
    out3 = write_report(data, output_dir=tmp_path / "reports")
    assert out1.name == "2026-04-09_1432_test-exp_gpt-5-mini.md"
    assert out2.name == "2026-04-09_1432_test-exp_gpt-5-mini_2.md"
    assert out3.name == "2026-04-09_1432_test-exp_gpt-5-mini_3.md"


# ---------------------------------------------------------------------------
# Polish UX 회귀 방지
# ---------------------------------------------------------------------------


def test_judge_skipped_shows_미실행_in_judge_cost(tmp_path):
    """--skip-judge로 돌린 run은 'Judge 비용 (USD) | (미실행)' 으로 표시."""
    _make_fixture(tmp_path, judge_skipped=True)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "| Judge 비용 (USD) | (미실행) |" in md


def test_gpt5_warning_only_for_gpt5_models(tmp_path):
    """non-gpt-5 모델에서는 reasoning 주의 문구가 나타나지 않아야 함."""
    _make_fixture(tmp_path, llm_model="gpt-4o-mini")
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "gpt-5 계열은 reasoning tokens" not in md


def test_no_blank_line_clutter_when_no_warnings(tmp_path):
    """warnings/gpt5 주의문이 모두 비어있을 때 본문에 빈 줄 3개 이상이 없어야 함."""
    _make_fixture(tmp_path, llm_model="gpt-4o-mini")  # gpt5_warning 비활성
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # Cost 표 직후 곧바로 "## 리소스 링크"가 와야 함 (사이에 빈 줄 3개 이상 없어야)
    assert "\n\n\n\n" not in md
