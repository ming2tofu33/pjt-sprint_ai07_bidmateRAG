"""Markdown experiment report generator.

Reads ``artifacts/logs/runs/{run_id}.jsonl``,
``artifacts/logs/benchmarks/{exp_name}.parquet``,
``artifacts/logs/runs/{run_id}.meta.json``, and
``artifacts/logs/embeddings/{collection_name}.json`` and produces a single
markdown file at ``artifacts/reports/{exp_name}_{run_id}.md`` that the team
can copy-paste into Notion.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bidmate_rag.tracking.pricing import is_model_priced, load_pricing
from bidmate_rag.tracking.templates import REPORT_TEMPLATE

logger = logging.getLogger(__name__)


def _fmt_num(value: Any, digits: int = 4, default: str = "N/A") -> str:
    """숫자를 소수점 자릿수로 포맷한다. None이면 기본값 반환."""
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _fmt_int(value: Any, default: str = "N/A") -> str:
    """정수를 천 단위 콤마로 포맷한다. None이면 기본값 반환."""
    if value is None:
        return default
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return default


@dataclass
class ReportData:
    """리포트 생성에 필요한 데이터를 담는 컨테이너."""

    run_id: str
    experiment_name: str
    meta: dict[str, Any]
    summary_row: dict[str, Any]
    results: list[dict[str, Any]]
    embedding_meta: dict[str, Any] | None
    experiment_notes: dict[str, Any] | None
    pricing: dict[str, Any]
    runs_dir: Path
    benchmarks_dir: Path
    embeddings_dir: Path
    extras: dict[str, Any] = field(default_factory=dict)


def _resolve_existing_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        alt = repo_root / candidate
        if alt.exists():
            return alt
    return None


def _load_yaml_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    resolved = _resolve_existing_path(path)
    if resolved is None:
        if path is not None:
            logger.warning("Experiment notes not found: %s", path)
        return None
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.warning("Experiment notes YAML could not be parsed: %s", resolved)
        return None
    if not isinstance(loaded, dict):
        logger.warning("Experiment notes must be a mapping: %s", resolved)
        return None
    return loaded


def load_report_data(
    run_id: str,
    runs_dir: str | Path = "artifacts/logs/runs",
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
    embeddings_dir: str | Path = "artifacts/logs/embeddings",
    experiment_name: str | None = None,
) -> ReportData:
    runs_path = Path(runs_dir)
    benchmarks_path = Path(benchmarks_dir)
    embeddings_path = Path(embeddings_dir)

    # 1. meta.json (있으면)
    meta_file = runs_path / f"{run_id}.meta.json"
    meta: dict[str, Any] = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # 2. experiment_name 결정 (인자 우선, 그다음 meta, 그다음 parquet 스캔)
    exp_name = experiment_name or meta.get("experiment_name")
    if not exp_name:
        exp_name = _scan_benchmarks_for_run(benchmarks_path, run_id)
    if not exp_name:
        raise FileNotFoundError(
            f"Could not determine experiment_name for run_id={run_id}. "
            "Pass --experiment-name or ensure meta.json exists."
        )

    # 3. results jsonl
    jsonl_file = runs_path / f"{run_id}.jsonl"
    if not jsonl_file.exists():
        raise FileNotFoundError(f"Run results not found: {jsonl_file}")
    results = [
        json.loads(line)
        for line in jsonl_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 4. parquet → 해당 run_id 행
    parquet_file = benchmarks_path / f"{exp_name}.parquet"
    summary_row: dict[str, Any] = {}
    if parquet_file.exists():
        frame = pd.read_parquet(parquet_file)
        matching = frame[frame["run_id"] == run_id]
        if not matching.empty:
            summary_row = matching.iloc[-1].to_dict()
    else:
        logger.warning("Benchmark parquet not found: %s", parquet_file)

    # 5. embedding meta (collection_name으로 매칭)
    collection_name = meta.get("collection_name")
    embedding_meta: dict[str, Any] | None = None
    if collection_name:
        emb_file = embeddings_path / f"{collection_name}.json"
        if emb_file.exists():
            embedding_meta = json.loads(emb_file.read_text(encoding="utf-8"))

    experiment_notes = _load_yaml_if_exists(meta.get("notes_path"))

    return ReportData(
        run_id=run_id,
        experiment_name=exp_name,
        meta=meta,
        summary_row=summary_row,
        results=results,
        embedding_meta=embedding_meta,
        experiment_notes=experiment_notes,
        pricing=load_pricing(),
        runs_dir=runs_path,
        benchmarks_dir=benchmarks_path,
        embeddings_dir=embeddings_path,
    )


def _scan_benchmarks_for_run(benchmarks_path: Path, run_id: str) -> str | None:
    if not benchmarks_path.exists():
        return None
    for parquet_file in benchmarks_path.glob("*.parquet"):
        try:
            frame = pd.read_parquet(parquet_file)
        except Exception:  # noqa: BLE001
            continue
        if "run_id" in frame.columns and (frame["run_id"] == run_id).any():
            return parquet_file.stem
    return None


def render_markdown(data: ReportData) -> str:
    ctx = _build_context(data)
    return REPORT_TEMPLATE.format(**ctx)


def _sanitize_filename_component(value: str) -> str:
    """Replace path-unsafe characters in a single filename component."""
    text = str(value or "unknown").strip()
    if not text:
        return "unknown"
    return text.replace("/", "-").replace("\\", "-").replace(" ", "-").replace(":", "-")


def build_report_filename(data: ReportData) -> str:
    """사람이 보기 쉬운 리포트 파일명 생성.

    형식: ``YYYY-MM-DD_HHMM_{experiment_name}_{model}.md``

    예: ``2026-04-10_1429_generation-compare_gpt-5-mini.md``

    timestamp_kst가 meta.json에 없으면 ``unknown_unknown`` prefix로 fallback.
    호출자가 충돌 처리(``_2`` suffix)를 직접 하므로 이 함수는 base 이름만 반환.
    """
    timestamp_kst = data.meta.get("timestamp_kst", "")
    date_part = "unknown"
    time_part = "unknown"
    if timestamp_kst:
        # "2026-04-10 14:29:36" → date="2026-04-10", time="1429"
        parts = timestamp_kst.split(" ")
        if len(parts) == 2:
            date_part = parts[0]
            time_part = parts[1].replace(":", "")[:4]
    provider_cfg = (data.meta.get("config_snapshot") or {}).get("provider") or {}
    model = provider_cfg.get("model") or "unknown"
    exp = _sanitize_filename_component(data.experiment_name)
    model_safe = _sanitize_filename_component(model)
    return f"{date_part}_{time_part}_{exp}_{model_safe}.md"


def write_report(
    data: ReportData,
    output_dir: str | Path = "artifacts/reports",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_report_filename(data)
    out_path = out_dir / base_name
    # 같은 분에 두 번 돌리는 경우 _2, _3, … suffix
    counter = 2
    while out_path.exists():
        stem = base_name[:-3]  # ".md" 제거
        out_path = out_dir / f"{stem}_{counter}.md"
        counter += 1
    out_path.write_text(render_markdown(data), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# context builders
# ---------------------------------------------------------------------------


def _build_context(data: ReportData) -> dict[str, Any]:
    summary = data.summary_row
    meta = data.meta
    notes = data.experiment_notes or {}
    runtime_cfg = meta.get("config_snapshot", {}) or {}
    provider_cfg = runtime_cfg.get("provider", {}) or {}
    experiment_cfg = runtime_cfg.get("experiment", {}) or {}
    project_cfg = runtime_cfg.get("project", {}) or {}
    git = meta.get("git", {}) or {}

    # Costs
    generation_cost = float(
        meta.get(
            "generation_cost_usd",
            sum(float(r.get("cost_usd") or 0.0) for r in data.results),
        )
        or 0.0
    )
    rewrite_cost = float(
        meta.get(
            "rewrite_cost_usd",
            sum(
                float(((r.get("debug") or {}).get("rewrite_cost_usd", 0.0) or 0.0))
                for r in data.results
            ),
        )
        or 0.0
    )
    embedding_cost = float((data.embedding_meta or {}).get("total_cost_usd", 0.0) or 0.0)
    judge_cost = float(meta.get("judge_cost_usd", meta.get("judge_total_cost_usd", 0.0)) or 0.0)
    llm_total_cost = float(
        meta.get("total_cost_usd", generation_cost + rewrite_cost + judge_cost) or 0.0
    )
    grand_total = llm_total_cost + embedding_cost

    # Tokens
    prompt_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("prompt", 0) or 0) for r in data.results
    )
    completion_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("completion", 0) or 0) for r in data.results
    )
    rewrite_prompt_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("rewrite_prompt", 0) or 0) for r in data.results
    )
    rewrite_completion_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("rewrite_completion", 0) or 0)
        for r in data.results
    )
    rewrite_total_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("rewrite_total", 0) or 0) for r in data.results
    )
    total_tokens = prompt_tokens_sum + completion_tokens_sum + rewrite_total_tokens_sum

    # Latency
    latencies_ms = [float(r.get("latency_ms") or 0.0) for r in data.results if r.get("latency_ms")]
    latency_avg_s = (sum(latencies_ms) / len(latencies_ms) / 1000) if latencies_ms else None
    latency_p95_s = _percentile(latencies_ms, 95) / 1000 if latencies_ms else None

    # Models / config
    embedding_model = (
        provider_cfg.get("embedding_model")
        or (data.results[0].get("embedding_model") if data.results else None)
        or "unknown"
    )
    llm_model = (
        provider_cfg.get("model")
        or (data.results[0].get("llm_model") if data.results else None)
        or "unknown"
    )
    chunk_size = experiment_cfg.get("chunk_size") or project_cfg.get("default_chunk_size", "?")
    chunk_overlap = experiment_cfg.get("chunk_overlap") or project_cfg.get(
        "default_chunk_overlap", "?"
    )
    top_k = meta.get("actual_top_k") or experiment_cfg.get("retrieval_top_k") or project_cfg.get("default_retrieval_top_k", 5)
    collection_name = meta.get("collection_name") or provider_cfg.get("collection_name") or "?"
    scenario = summary.get("scenario") or provider_cfg.get("scenario") or "?"
    provider_label = summary.get("provider_label", "?")

    # Eval path
    eval_path = meta.get("eval_path", "?")
    eval_basename = Path(str(eval_path)).name if eval_path != "?" else "?"
    num_samples = int(summary.get("num_samples") or len(data.results) or 0)

    # Cost warnings (priced?) — 빈 줄이 어색하지 않도록 warning이 있을 때만 \n 감싸기
    warnings = []
    if not is_model_priced("llm", llm_model, data.pricing):
        warnings.append(f"⚠️ 생성 모델 `{llm_model}` 단가 미등록 — `configs/pricing.yaml` 갱신 필요")
    if data.embedding_meta and not is_model_priced("embedding", embedding_model, data.pricing):
        warnings.append(
            f"⚠️ 임베딩 모델 `{embedding_model}` 단가 미등록 — `configs/pricing.yaml` 갱신 필요"
        )
    if not data.embedding_meta:
        warnings.append("⚠️ 임베딩 비용 미수집 (build_index를 새 트래킹 코드로 다시 실행 필요)")
    warnings_text = "\n".join(warnings)
    cost_warning = f"\n{warnings_text}\n" if warnings_text else ""

    # gpt-5 reasoning 주의 문구 — 모델이 gpt-5 계열일 때만 표시
    gpt5_warning = ""
    if str(llm_model).startswith("gpt-5"):
        gpt5_warning = (
            "\n> ℹ️ gpt-5 계열은 reasoning tokens가 completion에 포함되어 "
            "cost가 예상보다 높을 수 있습니다.\n"
        )

    # judge 미실행 표시
    judge_skipped = bool(meta.get("judge_skipped"))
    judge_cost_str = "(미실행)" if judge_skipped else _fmt_num(judge_cost, digits=4)
    has_rewrite_usage = rewrite_total_tokens_sum > 0 or rewrite_cost > 0.0
    rewrite_token_rows = ""
    rewrite_cost_row = ""
    if has_rewrite_usage:
        rewrite_token_rows = (
            f"| Rewrite Prompt Tokens | {_fmt_int(rewrite_prompt_tokens_sum)} |\n"
            f"| Rewrite Completion Tokens | {_fmt_int(rewrite_completion_tokens_sum)} |\n"
            f"| Rewrite Total Tokens | {_fmt_int(rewrite_total_tokens_sum)} |\n"
        )
        rewrite_cost_row = f"| Rewrite Cost (USD) | {_fmt_num(rewrite_cost, digits=4)} |\n"

    # Config links
    configs = meta.get("configs", {}) or {}
    config_links = (
        "\n".join(
            f"  - `{configs[k]}`" for k in ("base", "provider", "experiment") if configs.get(k)
        )
        or "  - (미기록)"
    )

    experiment_title = _first_non_empty(notes.get("title"), data.experiment_name)
    experiment_overview = _first_non_empty(
        notes.get("overview"),
        notes.get("summary"),
        notes.get("description"),
    ) or "실험 목적/배경을 입력하세요"
    hypothesis_bullets = _format_bullet_block(
        notes.get("hypothesis"),
        data.experiment_notes,
        placeholder_lines=[
            "왜 이 변경을 했는지:",
            "무엇이 좋아질 거라고 봤는지:",
        ],
    )
    changes_bullets = _format_bullet_block(
        notes.get("changes"),
        data.experiment_notes,
        placeholder_lines=["변경 내용을 입력하세요"],
    )
    expected_outcome_bullets = _format_bullet_block(
        notes.get("expected_outcome"),
        data.experiment_notes,
        placeholder_lines=["기대 결과를 입력하세요"],
    )
    next_actions_bullets = _format_bullet_block(
        notes.get("next_actions"),
        data.experiment_notes,
        placeholder_lines=[
            "다음 실험에서 무엇을 바꿀지:",
            "유지할 것:",
            "버릴 것:",
        ],
    )
    failure_case_blocks = _build_failure_case_blocks(data, notes)

    # Judge metrics from summary first, fall back to results aggregate
    def _get_metric(key: str) -> Any:
        return summary.get(key) if summary else None

    return {
        "experiment_name": data.experiment_name,
        "experiment_title": experiment_title,
        "experiment_overview": experiment_overview,
        "run_id": data.run_id,
        "timestamp_kst": meta.get("timestamp_kst", "N/A"),
        "scenario": scenario,
        "eval_basename": eval_basename,
        "eval_path": eval_path,
        "num_samples": num_samples,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "collection_name": collection_name,
        "provider_label": provider_label,
        "git_commit": git.get("commit", "unknown"),
        "git_commit_short": git.get("commit_short", "unknown"),
        "git_branch": git.get("branch", "unknown"),
        "dirty_marker": "(dirty)" if git.get("dirty") else "",
        # metrics
        "hit_rate": _fmt_num(_get_metric(f"hit_rate@{top_k}")),
        "mrr": _fmt_num(_get_metric("mrr")),
        "ndcg": _fmt_num(_get_metric(f"ndcg@{top_k}")),
        "map": _fmt_num(_get_metric(f"map@{top_k}")),
        "faithfulness": _fmt_num(_get_metric("faithfulness")),
        "answer_relevance": _fmt_num(_get_metric("answer_relevance")),
        "context_precision": _fmt_num(_get_metric("context_precision")),
        "context_recall": _fmt_num(_get_metric("context_recall")),
        # latency
        "latency_avg_s": _fmt_num(latency_avg_s, digits=3),
        "latency_p95_s": _fmt_num(latency_p95_s, digits=3),
        # tokens
        "prompt_tokens_sum": _fmt_int(prompt_tokens_sum),
        "completion_tokens_sum": _fmt_int(completion_tokens_sum),
        "rewrite_token_rows": rewrite_token_rows,
        "total_tokens": _fmt_int(total_tokens),
        # costs
        "generation_cost": _fmt_num(generation_cost, digits=4),
        "rewrite_cost_row": rewrite_cost_row,
        "embedding_cost": (
            _fmt_num(embedding_cost, digits=4) if data.embedding_meta else "N/A (미수집)"
        ),
        "judge_cost": judge_cost_str,
        "grand_total_cost": _fmt_num(grand_total, digits=4),
        "cost_warning": cost_warning,
        "gpt5_warning": gpt5_warning,
        # paths
        "run_jsonl_path": str(data.runs_dir / f"{data.run_id}.jsonl"),
        "benchmark_parquet_path": str(data.benchmarks_dir / f"{data.experiment_name}.parquet"),
        "meta_json_path": str(data.runs_dir / f"{data.run_id}.meta.json"),
        "config_links": config_links,
        # notes-derived report body
        "hypothesis_bullets": hypothesis_bullets,
        "changes_bullets": changes_bullets,
        "expected_outcome_bullets": expected_outcome_bullets,
        "failure_case_blocks": failure_case_blocks,
        "next_actions_bullets": next_actions_bullets,
    }


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _format_bullet_block(
    value: Any,
    notes_present: dict[str, Any] | None,
    *,
    placeholder_lines: list[str],
) -> str:
    if not notes_present:
        return "\n".join(f"- {line}" for line in placeholder_lines)
    if not value:
        return "\n".join(f"- {line}" for line in placeholder_lines)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    bullets = []
    for item in items:
        text = str(item).strip()
        if text:
            bullets.append(f"- {text}")
    return "\n".join(bullets) if bullets else "\n".join(f"- {line}" for line in placeholder_lines)


def _normalize_question_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_failure_case_blocks(data: ReportData, notes: dict[str, Any]) -> str:
    matched_cases = _select_failure_case_entries(data, notes)
    if not matched_cases:
        return ""
    blocks: list[str] = []
    for index, entry in enumerate(matched_cases[:2], start=1):
        row = entry["row"]
        note = entry["note"]
        question_id = row.get("question_id", "?")
        question = row.get("question", "N/A")
        answer = row.get("answer", "N/A")
        retrieved_chunks = row.get("retrieved_chunks") or []
        judge_scores = row.get("judge_scores") or {}
        evidence = note.get("why_watch") or note.get("reason") or note.get("note") or ""
        if not evidence:
            explicit_error = _extract_error_text(row)
            if explicit_error:
                evidence = explicit_error
        if not evidence:
            if not retrieved_chunks:
                evidence = "retrieved_chunks가 비어 있음"
            elif judge_scores:
                numeric_scores = [
                    float(score)
                    for score in judge_scores.values()
                    if isinstance(score, (int, float))
                ]
                evidence = (
                    f"judge score 최저값 {min(numeric_scores):.2f}"
                    if numeric_scores
                    else "judge score 기록 없음"
                )
            else:
                evidence = "run 결과에서 약한 사례로 선별"
        block_lines = [
            f"### 실패 사례 {index}",
            f"- 질문 ID: {question_id}",
            f"- 질문: {question}",
            f"- 실제 결과: {answer}",
            f"- 관찰 포인트: {evidence}",
        ]
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def _select_failure_case_entries(
    data: ReportData, notes: dict[str, Any]
) -> list[dict[str, Any]]:
    results_by_qid = {
        _normalize_question_id(row.get("question_id")): row for row in data.results if row.get("question_id")
    }
    note_cases = notes.get("failure_cases") or []
    selected: list[dict[str, Any]] = []
    matched_qids: set[str] = set()

    if isinstance(note_cases, list):
        for note_case in note_cases:
            if not isinstance(note_case, dict):
                continue
            qid = _normalize_question_id(note_case.get("question_id"))
            if not qid or qid not in results_by_qid:
                continue
            matched_qids.add(qid)
            selected.append({"row": results_by_qid[qid], "note": note_case})

    if selected:
        if len(selected) >= 2:
            return selected[:2]
        for row in _select_fallback_weak_rows(data, exclude_qids=matched_qids):
            selected.append({"row": row, "note": {}})
            if len(selected) >= 2:
                break
        return selected

    return [{"row": row, "note": {}} for row in _select_fallback_weak_rows(data)[:2]]


def _select_fallback_weak_rows(
    data: ReportData, *, exclude_qids: set[str] | None = None
) -> list[dict[str, Any]]:
    exclude_qids = exclude_qids or set()

    def _score(row: dict[str, Any]) -> tuple[int, int, int, float, float, float, str]:
        qid = _normalize_question_id(row.get("question_id"))
        retrieved_chunks = row.get("retrieved_chunks") or []
        judge_scores = row.get("judge_scores") or {}
        explicit_error = 0 if _row_has_explicit_error(row) else 1
        has_retrieval_miss = 0 if not retrieved_chunks else 1
        has_judge_scores = 0 if judge_scores else 1
        if judge_scores:
            numeric_scores = [
                float(score)
                for score in judge_scores.values()
                if isinstance(score, (int, float))
            ]
            score_value = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
        else:
            score_value = 1.0
        latency_ms = float(row.get("latency_ms") or 0.0)
        cost_usd = float(row.get("cost_usd") or 0.0)
        return (
            explicit_error,
            has_retrieval_miss,
            has_judge_scores,
            score_value,
            -latency_ms,
            -cost_usd,
            qid,
        )

    rows = [
        row
        for row in data.results
        if _normalize_question_id(row.get("question_id")) not in exclude_qids
    ]
    return sorted(rows, key=_score)


def _extract_error_text(row: dict[str, Any]) -> str:
    for key in ("error", "error_message", "exception", "failure_reason", "judge_error"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text and text.lower() not in {"ok", "success", "succeeded", "passed", "pass", "none"}:
                return text
        elif value not in (False, "", [], {}, ()):
            return str(value)
    status = str(row.get("status") or "").strip()
    if status.lower() in {"error", "failed", "failure", "exception"}:
        return status
    return ""


def _row_has_explicit_error(row: dict[str, Any]) -> bool:
    return bool(_extract_error_text(row))


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # statistics.quantiles(n=100) → 99 cut points; index pct-1
    try:
        quantiles = statistics.quantiles(sorted_vals, n=100)
        return quantiles[pct - 1]
    except statistics.StatisticsError:
        return sorted_vals[-1]
