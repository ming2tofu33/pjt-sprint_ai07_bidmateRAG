"""평가 탭 — 4개 서브탭 (실행, 디버깅, 비교, 편집)."""

from __future__ import annotations

import json
from pathlib import Path
from app.api.routes import run_benchmark_experiment, load_metadata_options, list_scenario_a_embeddings, list_scenario_a_llms
import pandas as pd
import yaml as _yaml

from app.api.routes import run_benchmark_experiment, load_metadata_options
from bidmate_rag.evaluation.dataset import (
    find_latest_eval_dir,
    normalize_metadata_filter,
)
from bidmate_rag.tracking.markdown_report import load_report_data, write_report

# 평가셋은 ``data/eval/eval_v1/``, ``eval_v2/`` 등 버전 디렉토리에 둡니다.
# UI는 가장 높은 버전을 자동으로 사용 (새 버전 만들면 코드 수정 없이 반영됨).
EVAL_DIR = find_latest_eval_dir()

# Provider 정렬/필터 공통 유틸
B_ORDER = {"gpt-5": 0, "gpt-5-mini": 1, "gpt-5-nano": 2}


def _get_provider_info(path):
    try:
        cfg = _yaml.safe_load(path.read_text())
        return cfg.get("scenario", ""), cfg.get("model", "")
    except Exception:
        return "", ""


def _format_provider(p):
    scenario, model = _get_provider_info(p)
    tag = "🅰️" if scenario == "scenario_a" else "🅱️"
    return f"{tag} {model} ({p.stem})"


def _sort_providers(configs):
    """B 시나리오(gpt-5→mini→nano) 우선, A 시나리오 이름순."""
    return sorted(
        configs,
        key=lambda p: (
            0 if _get_provider_info(p)[0] == "scenario_b" else 1,
            B_ORDER.get(_get_provider_info(p)[1], 99),
            p.stem,
        ),
    )


def _render_scenario_provider_selector(st, list_provider_configs, list_scenario_a_embeddings=None, list_scenario_a_llms=None, key_prefix=""):
    """시나리오 체크박스 + Provider selectbox. 평가실행/디버깅 공통."""
    provider_configs = list_provider_configs()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        show_a = st.checkbox("🅰️ 시나리오 A", value=False, key=f"{key_prefix}_sa")
    with col_s2:
        show_b = st.checkbox("🅱️ 시나리오 B", value=True, key=f"{key_prefix}_sb")

    filtered = []
    for p in provider_configs:
        scenario, _ = _get_provider_info(p)
        if scenario == "scenario_a" and show_a and not (show_a and not show_b):
            filtered.append(p)
        elif scenario == "scenario_b" and show_b:
            filtered.append(p)
    filtered = _sort_providers(filtered)

    provider = None
    # 시나리오 B만 선택 시 Provider selectbox 표시
    if show_b:
        if not filtered:
            st.warning("선택한 시나리오에 Provider가 없습니다.")
            return None, None, None
        provider = st.selectbox(
            "Provider", filtered, format_func=_format_provider, key=f"{key_prefix}_provider"
        )

    # 시나리오 A 전용 설정 (Provider 숨김)
    selected_embedding = None
    selected_llm = None
    if show_a and not show_b and list_scenario_a_embeddings and list_scenario_a_llms:
        embedding_configs = list_scenario_a_embeddings()
        if embedding_configs:
            selected_embedding = st.selectbox(
                "임베딩 모델",
                embedding_configs,
                format_func=lambda p: p.stem,
                key=f"{key_prefix}_embedding",
            )
        llm_configs = list_scenario_a_llms()
        if llm_configs:
            selected_llm = st.selectbox(
                "LLM 모델",
                llm_configs,
                format_func=lambda p: p.stem,
                key=f"{key_prefix}_llm",
            )

    return provider, selected_embedding, selected_llm

EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RUNS_DIR = Path("artifacts/logs/runs")
BENCHMARKS_DIR = Path("artifacts/logs/benchmarks")


def _render_chunking_selector(st, list_chunking_configs, key_prefix=""):
    # 청킹 전략 selectbox
    chunking_configs = list_chunking_configs()
    default_idx = next((i for i, p in enumerate(chunking_configs) if "1000_150" in p.stem), 0)
    return st.selectbox(
        "청킹 전략",
        chunking_configs,
        index=default_idx,
        format_func=lambda p: p.stem,
        key=f"{key_prefix}_chunking",
    )


def _parse_json_field(value) -> object:
    """CSV에서 읽은 JSON 문자열 필드를 파싱한다."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s or s == "[]" or s == "{}":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def load_eval_set_from_csv(path: Path) -> list[dict]:
    """CSV 파일에서 평가셋을 로딩한다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        q = {
            "id": str(row.get("id", "")),
            "type": str(row.get("type", "A")),
            "difficulty": str(row.get("difficulty", "중")),
            "question": str(row.get("question", "")),
            "ground_truth_answer": str(row.get("ground_truth_answer", "")),
            "ground_truth_docs": _parse_json_field(row.get("ground_truth_docs")) or [],
            "metadata_filter": _parse_json_field(row.get("metadata_filter")),
            "history": _parse_json_field(row.get("history")),
        }
        records.append(q)
    return records


def load_eval_set() -> list[dict]:
    """평가셋을 로딩한다. CSV가 있으면 CSV 우선, 없으면 JSON."""
    csv_files = sorted(EVAL_DIR.glob("eval_batch_*.csv"))
    if csv_files:
        # 가장 최신 CSV 로딩
        all_records = []
        for csv_path in csv_files:
            all_records.extend(load_eval_set_from_csv(csv_path))
        return all_records
    if EVAL_SET_PATH.exists():
        return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    return []


def save_eval_set(data: list[dict], fmt: str = "csv") -> Path:
    """평가셋을 저장한다."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        save_path = EVAL_DIR / "eval_set_edited.csv"
        rows = []
        for q in data:
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "difficulty": q.get("difficulty", "중"),
                    "question": q["question"],
                    "ground_truth_answer": q.get("ground_truth_answer", ""),
                    "ground_truth_docs": json.dumps(
                        q.get("ground_truth_docs", []), ensure_ascii=False
                    ),
                    "metadata_filter": json.dumps(
                        q.get("metadata_filter") or {}, ensure_ascii=False
                    ),
                    "history": json.dumps(q.get("history") or [], ensure_ascii=False),
                }
            )
        pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
        return save_path
    else:
        EVAL_SET_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return EVAL_SET_PATH


def render_eval_tabs(
    st,
    run_live_query,
    list_provider_configs,
    list_chunking_configs,
    list_scenario_a_embeddings, # 시나리오 A 임베딩 옵션 로딩 함수
    list_scenario_a_llms, # 시나리오 A LLM 옵션 로딩 함수
    load_benchmark_frames,
    load_run_records,
):
    """평가 탭 메인 렌더러."""

    # 평가셋 파일 선택
    csv_files = sorted(EVAL_DIR.glob("eval_batch_*.csv"))
    all_eval_files = csv_files
    if EVAL_SET_PATH.exists():
        all_eval_files = list(all_eval_files) + [EVAL_SET_PATH]

    if all_eval_files:
        selected_file = st.selectbox(
            "📄 평가셋 파일",
            all_eval_files,
            format_func=lambda p: f"{p.name} ({p.stat().st_size // 1024}KB)",
            key="eval_file_select",
        )
        if selected_file and selected_file != st.session_state.get("_loaded_eval_file"):
            if selected_file.suffix == ".csv":
                st.session_state.eval_set = load_eval_set_from_csv(selected_file)
            else:
                st.session_state.eval_set = json.loads(selected_file.read_text(encoding="utf-8"))
            st.session_state["_loaded_eval_file"] = selected_file
            st.toast(f"{selected_file.name} 로딩: {len(st.session_state.eval_set)}개", icon="📄")

    run_tab, debug_tab, compare_tab, edit_tab = st.tabs(
        ["🏃 평가 실행", "🔍 질문 디버깅", "⚖️ 결과 비교", "✏️ 평가셋 편집"]
    )

    # 평가셋 session 관리
    if "eval_set" not in st.session_state:
        st.session_state.eval_set = load_eval_set()
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = {}

    eval_set = st.session_state.eval_set

    # ── 서브탭 1: 평가 실행 ──
    with run_tab:
        _render_run_tab(st, eval_set, run_live_query, list_provider_configs, list_chunking_configs, list_scenario_a_embeddings, list_scenario_a_llms, ) # 시나리오 A 옵션 전달 추가

    # ── 서브탭 2: 질문 디버깅 ──
    with debug_tab:
        _render_debug_tab(
            st, eval_set, run_live_query, list_provider_configs, list_chunking_configs, list_scenario_a_embeddings, list_scenario_a_llms   # 시나리오 A 옵션 전달 추가
        )

    # ── 서브탭 3: 결과 비교 ──
    with compare_tab:
        _render_compare_tab(st, load_benchmark_frames, load_run_records)

    # ── 서브탭 4: 평가셋 편집 ──
    with edit_tab:
        _render_edit_tab(st, eval_set)


def _render_run_tab(st, eval_set, run_live_query, list_provider_configs, list_chunking_configs, list_scenario_a_embeddings, list_scenario_a_llms): # 시나리오 A 옵션 인자 추가
    """평가셋 일괄 실행 탭. CLI(``bidmate-eval``)와 정확히 같은 코드 경로를 호출한다.

    UX 단순화 원칙:
    - top_k/필터링 등 옵션은 CLI 인자와 같은 것만 제공
    - 평가셋 파일 자체를 디스크 경로 기반으로 사용 (session 편집은 '편집' 탭에서
      먼저 저장 후 실행)
    - 비즈니스 로직은 0줄 — ``run_benchmark_experiment``가 모든 일을 처리
    """
    st.subheader("평가셋 일괄 실행")

    if not eval_set:
        st.warning("평가셋이 비어있습니다. '평가셋 편집' 탭에서 질문을 추가하세요.")
        return

    eval_file: Path | None = st.session_state.get("_loaded_eval_file")
    if eval_file is None:
        st.info("위쪽에서 평가셋 파일을 먼저 선택하세요.")
        return

    st.caption(
        f"실행 대상: `{eval_file.name}` (총 {len(eval_set)}개 — 편집한 내용을 평가하려면 "
        "'평가셋 편집' 탭에서 저장 후 다시 로딩하세요)"
    )

    provider, selected_embedding, selected_llm = _render_scenario_provider_selector( 
    st, list_provider_configs, list_scenario_a_embeddings, list_scenario_a_llms, key_prefix="run") # 시나리오 A 임베딩/LLM 선택 추가
    if provider is None and selected_embedding is None:
        return
    chunking = _render_chunking_selector(
        st, list_chunking_configs, key_prefix="run"
    )  # 청킹 선택 추가

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        skip_judge = st.checkbox(
            "Judge 끄기 (faithfulness 등)",
            value=False,
            key="run_skip_judge",
            help="LLM judge는 추가 API 호출이라 비용/시간이 늘어납니다. 빠른 실행에 사용",
        )
    with opt_col2:
        judge_model = st.selectbox(
            "Judge 모델",
            ["gpt-4o-mini", "gpt-5-mini"],
            disabled=skip_judge,
            key="run_judge_model",
        )

    if not st.button("▶️ 평가 실행", type="primary", key="run_eval_btn"):
        return

    progress_bar = st.progress(0, text="평가 실행 중...")

    def _on_progress(done: int, total: int, sample) -> None:
        progress_bar.progress(done / total, text=f"[{done}/{total}] {sample.question[:40]}...")

    try:
        artifacts = run_benchmark_experiment(
            evaluation_path=eval_file,
            provider_config_path=provider,
            experiment_config_path=chunking,  # 청킹 전략 전달
            skip_judge=skip_judge,
            judge_model=judge_model,
            progress_callback=_on_progress,
            embedding_config_path=selected_embedding,  # 시나리오 A 임베딩 전달,
            llm_config_path=selected_llm,  # 시나리오 A LLM 전달
        )
    except Exception as exc:
        progress_bar.empty()
        st.error(f"평가 실행 실패: {exc}")
        return

    progress_bar.empty()

    # compare_tab은 list[dict]를 기대하므로 legacy 호환 형식으로 저장
    legacy_rows = [
        {
            "id": r.question_id,
            "type": "",
            "difficulty": "",
            "question": r.question,
            "answer_preview": (r.answer or "")[:100],
            "chunks": len(r.retrieved_chunks),
            "tokens": int((r.token_usage or {}).get("total", 0) or 0),
            "latency_ms": round(r.latency_ms),
            "ground_truth": "",
        }
        for r in artifacts.benchmark.results
    ]
    st.session_state.eval_results[artifacts.run_id] = legacy_rows

    st.success(f"실행 완료: {len(artifacts.benchmark.results)}건 (run_id: `{artifacts.run_id}`)")

    _render_run_artifacts(st, artifacts)


def _render_run_artifacts(st, artifacts) -> None:
    """평가 실행 결과를 메트릭 카드 + 표 + 리포트 다운로드로 렌더링."""
    metrics = artifacts.metrics or {}
    results = artifacts.benchmark.results

    # 비용/토큰/지연 집계
    generation_cost = sum(float(r.cost_usd or 0.0) for r in results)
    judge_cost = float(artifacts.judge_total_cost_usd or 0.0)
    prompt_tokens = sum(int((r.token_usage or {}).get("prompt", 0) or 0) for r in results)
    completion_tokens = sum(int((r.token_usage or {}).get("completion", 0) or 0) for r in results)
    total_tokens = prompt_tokens + completion_tokens
    latencies_ms = [float(r.latency_ms or 0.0) for r in results]
    avg_latency_s = sum(latencies_ms) / len(latencies_ms) / 1000 if latencies_ms else 0.0

    # ── 검색 메트릭 카드 ──
    st.markdown("#### 🔍 검색 품질")
    cols = st.columns(4)
    cols[0].metric("Hit Rate@5", _fmt_metric(metrics.get("hit_rate@5")))
    cols[1].metric("MRR", _fmt_metric(metrics.get("mrr")))
    cols[2].metric("nDCG@5", _fmt_metric(metrics.get("ndcg@5")))
    cols[3].metric("MAP@5", _fmt_metric(metrics.get("map@5")))

    # ── Judge 메트릭 카드 ──
    if not artifacts.judge_skipped:
        st.markdown("#### 🤖 답변 품질 (LLM Judge)")
        cols = st.columns(4)
        cols[0].metric("Faithfulness", _fmt_metric(metrics.get("faithfulness")))
        cols[1].metric("Answer Relevance", _fmt_metric(metrics.get("answer_relevance")))
        cols[2].metric("Context Precision", _fmt_metric(metrics.get("context_precision")))
        cols[3].metric("Context Recall", _fmt_metric(metrics.get("context_recall")))
    else:
        st.info("Judge가 꺼져 있어 답변 품질 메트릭은 계산하지 않았습니다.")

    # ── 비용/토큰/지연 카드 ──
    st.markdown("#### 💰 비용·토큰·지연")
    cols = st.columns(4)
    cols[0].metric("생성 비용", f"${generation_cost:.4f}")
    cols[1].metric("Judge 비용", f"${judge_cost:.4f}" if not artifacts.judge_skipped else "—")
    cols[2].metric("총 토큰", f"{total_tokens:,}")
    cols[3].metric("평균 지연", f"{avg_latency_s:.2f}초")

    # ── 질문별 결과 표 ──
    st.markdown("#### 📋 질문별 결과")
    rows = []
    for r in results:
        judge = r.judge_scores or {}
        rows.append(
            {
                "id": r.question_id,
                "answer_preview": (r.answer or "")[:80],
                "chunks": len(r.retrieved_chunks),
                "tokens": int((r.token_usage or {}).get("total", 0) or 0),
                "latency_ms": round(r.latency_ms),
                "cost_usd": round(r.cost_usd or 0.0, 6),
                "faithfulness": judge.get("faithfulness"),
                "answer_relevance": judge.get("answer_relevance"),
                "error": bool(r.error),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    # ── 마크다운 리포트 생성 ──
    st.markdown("#### 📄 노션 리포트")
    st.caption(
        f"jsonl: `{artifacts.run_path}` · meta: `{artifacts.meta_path}` · "
        f"parquet: `{artifacts.summary_path}`"
    )
    if st.button("📋 노션 마크다운 리포트 생성", key=f"gen_report_{artifacts.run_id}"):
        try:
            data = load_report_data(run_id=artifacts.run_id)
            report_path = write_report(data)
            md_text = report_path.read_text(encoding="utf-8")
            st.success(f"리포트 생성 완료: `{report_path}`")
            st.download_button(
                "⬇️ 마크다운 다운로드",
                data=md_text,
                file_name=report_path.name,
                mime="text/markdown",
                key=f"dl_report_{artifacts.run_id}",
            )
            with st.expander("📝 리포트 미리보기", expanded=False):
                st.markdown(md_text)
        except Exception as exc:
            st.error(f"리포트 생성 실패: {exc}")


def _fmt_metric(value) -> str:
    """None은 'N/A', 그 외는 0.0000 형태로."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _render_debug_tab(st, eval_set, run_live_query, list_provider_configs, list_chunking_configs, list_scenario_a_embeddings, list_scenario_a_llms): # 시나리오 A 옵션 인자 추가
    """질문별 디버깅 탭. 평가셋에서 질문 하나를 선택해 검색/생성 결과와 메트릭을 상세히 보여준다."""
    st.subheader("질문별 디버깅")

    if not eval_set:
        st.warning("평가셋이 비어있습니다.")
        return

    # 필터
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox(
            "유형 필터", ["전체"] + sorted(set(q["type"] for q in eval_set)), key="debug_type"
        )
    with col2:
        diff_filter = st.selectbox(
            "난이도 필터",
            ["전체"] + sorted(set(q.get("difficulty", "중") for q in eval_set)),
            key="debug_diff",
        )

    filtered = eval_set
    if type_filter != "전체":
        filtered = [q for q in filtered if q["type"] == type_filter]
    if diff_filter != "전체":
        filtered = [q for q in filtered if q.get("difficulty", "중") == diff_filter]

    if not filtered:
        st.info("조건에 맞는 질문이 없습니다.")
        return

    selected_q = st.selectbox(
        "질문 선택",
        filtered,
        format_func=lambda q: (
            f"[{q['id']}|{q['type']}|{q.get('difficulty', '중')}] {q['question'][:50]}"
        ),
        key="debug_question",
    )

    if not selected_q:
        return

    # 설정
    provider, selected_embedding, selected_llm = _render_scenario_provider_selector(
    st, list_provider_configs, list_scenario_a_embeddings, list_scenario_a_llms, key_prefix="debug") # 시나리오 A 임베딩/LLM 선택 추가
    if provider is None and selected_embedding is None:
        return
    top_k = st.slider("Top-K", 1, 20, 5, key="debug_topk")
    chunking = _render_chunking_selector(
        st, list_chunking_configs, key_prefix="debug"
    )  # 청킹 선택 추가

    if st.button("🔍 이 질문 실행", type="primary", key="debug_run_btn"):
        with st.status("디버깅 실행 중...", expanded=True) as status:
            status.write("🔍 검색 중...")
            try:
                # 평가셋의 metadata_filter는 영문 키 → ChromaDB 한국어 키로 정규화
                # (cli/eval.py가 사용하는 normalize_metadata_filter와 동일 로직)
                raw_filter = selected_q.get("metadata_filter")
                normalized_filter = normalize_metadata_filter(
                    raw_filter if isinstance(raw_filter, dict) else None,
                    question=selected_q["question"],
                    agency_list=load_metadata_options().get("agencies", [])
                )
                # history는 list[{role, content}] 형식이면 그대로 전달
                history = selected_q.get("history") or None
                if not isinstance(history, list):
                    history = None

                result = run_live_query(
                    question=selected_q["question"],
                    provider_config_path=provider,
                    experiment_config_path=chunking,  # 청킹 전략 전달
                    top_k=top_k,
                    metadata_filter=normalized_filter,
                    chat_history=history,
                    embedding_config_path=selected_embedding,  # 시나리오 A 임베딩 전달,
                    llm_config_path=selected_llm,  # 시나리오 A LLM 전달
                )
                status.update(label="완료", state="complete")

                # 1) 검색 단계
                st.markdown("### 1단계: 검색")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**의도된 필터** (metadata_filter)")
                    st.json(selected_q.get("metadata_filter") or "자동 추출")
                with col2:
                    st.markdown("**기대 문서** (ground_truth_docs)")
                    st.json(selected_q.get("ground_truth_docs", []))

                if result.retrieved_chunks:
                    chunks_data = [
                        {
                            "순위": c.rank,
                            "유사도": round(c.score, 4),
                            "사업명": c.chunk.metadata.get("사업명", "")[:25],
                            "발주기관": c.chunk.metadata.get("발주기관", ""),
                            "도메인": c.chunk.metadata.get("사업도메인", ""),
                            "유형": c.chunk.content_type,
                            "내용": c.chunk.text[:120],
                        }
                        for c in result.retrieved_chunks
                    ]
                    st.dataframe(chunks_data, width="stretch")

                    # 정답 문서 포함 여부
                    retrieved_docs = [c.chunk.metadata.get("사업명", "") for c in result.retrieved_chunks]
                    retrieved_filenames = [c.chunk.metadata.get("파일명", "") for c in result.retrieved_chunks]
                    expected = selected_q.get("ground_truth_docs", [])
                    if expected:
                        found = [
                            e for e in expected if
                            any(e in d or d in e for d in retrieved_docs) or  # 사업명 부분 매칭
                            any(e == f or e in f for f in retrieved_filenames)  # 파일명 매칭
                        ]
                        if len(found) == len(expected):
                            st.success(f"검색 성공: 기대 문서 {len(found)}/{len(expected)}건 포함")
                        else:
                            st.warning(f"검색 부분 성공: 기대 문서 {len(found)}/{len(expected)}건만 포함")
                # 2) 생성 단계
                st.markdown("### 2단계: 생성 — 답변 vs 정답 비교")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**생성된 답변**")
                    st.text_area(
                        "답변", result.answer[:2000], height=300, key="debug_answer", disabled=True
                    )
                with col2:
                    st.markdown("**정답 (Ground Truth)**")
                    st.text_area(
                        "정답",
                        selected_q.get("ground_truth_answer", "없음"),
                        height=300,
                        key="debug_gt",
                        disabled=True,
                    )

                # 3) 메트릭
                st.markdown("### 3단계: 메트릭")
                cols = st.columns(4)
                cols[0].metric("검색 청크", f"{len(result.retrieved_chunks)}개")
                cols[1].metric("컨텍스트", f"{len(result.context) if result.context else 0:,}자")
                cols[2].metric("토큰", f"{result.token_usage.get('total', 0):,}")
                cols[3].metric("응답시간", f"{round(result.latency_ms)}ms")

            except Exception as e:
                status.update(label="오류", state="error")
                st.error(str(e))


def _render_compare_tab(st, load_benchmark_frames, load_run_records):
    st.subheader("결과 비교")

    # session에 저장된 run 결과 사용
    eval_runs = st.session_state.get("eval_results", {})

    # 파일 기반 결과도 로딩
    run_files = sorted(RUNS_DIR.glob("*.jsonl")) if RUNS_DIR.exists() else []

    all_sources = {}
    for run_id, results in eval_runs.items():
        all_sources[f"[세션] {run_id}"] = pd.DataFrame(results)
    for rf in run_files:
        try:
            records = [json.loads(line) for line in rf.read_text().splitlines() if line.strip()]
            if records:
                all_sources[f"[파일] {rf.stem}"] = pd.DataFrame(records)
        except Exception:
            pass

    if len(all_sources) < 2:
        st.info(
            "비교하려면 최소 2개 이상의 평가 결과가 필요합니다. '평가 실행' 탭에서 다른 설정으로 실행해보세요."
        )
        if all_sources:
            st.caption(f"현재 결과: {len(all_sources)}개")
            for name, df in all_sources.items():
                with st.expander(name):
                    st.dataframe(df, width="stretch")
        return

    col1, col2 = st.columns(2)
    source_names = list(all_sources.keys())
    with col1:
        left = st.selectbox("왼쪽 결과", source_names, index=0, key="compare_left")
    with col2:
        right_default = 1 if len(source_names) > 1 else 0
        right = st.selectbox("오른쪽 결과", source_names, index=right_default, key="compare_right")

    left_df = all_sources[left]
    right_df = all_sources[right]

    # 나란히 비교
    st.markdown("### 질문별 비교")
    if "id" in left_df.columns and "id" in right_df.columns:
        merged = left_df.merge(right_df, on="id", suffixes=("_좌", "_우"), how="outer")
        display_cols = [
            c
            for c in merged.columns
            if "id" in c or "answer" in c or "tokens" in c or "latency" in c
        ]
        st.dataframe(merged[display_cols] if display_cols else merged, width="stretch")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{left}**")
            st.dataframe(left_df, width="stretch")
        with col2:
            st.markdown(f"**{right}**")
            st.dataframe(right_df, width="stretch")

    # 집계 비교
    st.markdown("### 집계 비교")
    if "type" in left_df.columns and "type" in right_df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{left}** — 유형별")
            if "tokens" in left_df.columns:
                st.dataframe(left_df.groupby("type")["tokens"].mean().round(0), width="stretch")
        with col2:
            st.markdown(f"**{right}** — 유형별")
            if "tokens" in right_df.columns:
                st.dataframe(right_df.groupby("type")["tokens"].mean().round(0), width="stretch")


def _render_edit_tab(st, eval_set):
    st.subheader("평가셋 편집")
    st.caption(f"현재 {len(eval_set)}개 질문 (세션 편집 중 — 파일 저장은 별도)")

    # 질문 목록 테이블
    if eval_set:
        table_data = [
            {
                "ID": q["id"],
                "유형": q["type"],
                "난이도": q.get("difficulty", "중"),
                "질문": q["question"][:50],
                "기대문서": ", ".join(q.get("ground_truth_docs", []))[:30],
            }
            for q in eval_set
        ]
        st.dataframe(table_data, width="stretch")

    # 편집할 질문 선택 또는 새로 추가
    action = st.radio(
        "작업", ["기존 질문 편집", "새 질문 추가", "질문 삭제"], horizontal=True, key="edit_action"
    )

    if action == "기존 질문 편집" and eval_set:
        selected_idx = st.selectbox(
            "편집할 질문",
            range(len(eval_set)),
            format_func=lambda i: f"[{eval_set[i]['id']}] {eval_set[i]['question'][:50]}",
            key="edit_select",
        )
        q = eval_set[selected_idx]
        _render_question_form(st, q, key_prefix="edit", edit_mode=True, idx=selected_idx)

    elif action == "새 질문 추가":
        new_q = {
            "id": f"Q{len(eval_set) + 1:03d}",
            "type": "A",
            "difficulty": "중",
            "question": "",
            "ground_truth_answer": "",
            "ground_truth_docs": [],
            "metadata_filter": None,
            "history": None,
        }
        _render_question_form(st, new_q, key_prefix="new", edit_mode=False, idx=None)

    elif action == "질문 삭제" and eval_set:
        del_idx = st.selectbox(
            "삭제할 질문",
            range(len(eval_set)),
            format_func=lambda i: f"[{eval_set[i]['id']}] {eval_set[i]['question'][:50]}",
            key="del_select",
        )
        if st.button("🗑️ 삭제", type="secondary", key="del_btn"):
            st.session_state.eval_set.pop(del_idx)
            st.toast("질문 삭제됨", icon="🗑️")
            st.rerun()

    # 파일 저장/로딩
    st.divider()

    # 소스 파일 선택
    csv_files = sorted(EVAL_DIR.glob("eval_batch_*.csv"))
    source_options = [f.name for f in csv_files]
    if EVAL_SET_PATH.exists():
        source_options.append(EVAL_SET_PATH.name)

    if source_options:
        st.caption(f"사용 가능한 평가 파일: {', '.join(source_options)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        save_fmt = st.selectbox("저장 형식", ["csv", "json"], key="save_fmt")
        if st.button("💾 파일에 저장", type="primary", use_container_width=True, key="save_eval"):
            path = save_eval_set(st.session_state.eval_set, fmt=save_fmt)
            st.success(f"저장 완료: {path} ({len(st.session_state.eval_set)}개)")
    with col2:
        if st.button("🔄 파일에서 다시 로딩", use_container_width=True, key="reload_eval"):
            st.session_state.eval_set = load_eval_set()
            st.toast(f"로딩 완료: {len(st.session_state.eval_set)}개 질문", icon="🔄")
            st.rerun()
    with col3:
        # CSV 업로드
        uploaded = st.file_uploader(
            "CSV 업로드", type=["csv"], key="upload_csv", label_visibility="collapsed"
        )
        if uploaded:
            import io

            df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8-sig")))
            new_records = []
            for _, row in df.iterrows():
                new_records.append(
                    {
                        "id": str(row.get("id", "")),
                        "type": str(row.get("type", "A")),
                        "difficulty": str(row.get("difficulty", "중")),
                        "question": str(row.get("question", "")),
                        "ground_truth_answer": str(row.get("ground_truth_answer", "")),
                        "ground_truth_docs": _parse_json_field(row.get("ground_truth_docs")) or [],
                        "metadata_filter": _parse_json_field(row.get("metadata_filter")),
                        "history": _parse_json_field(row.get("history")),
                    }
                )
            st.session_state.eval_set = new_records
            st.toast(f"업로드 완료: {len(new_records)}개 질문", icon="📤")
            st.rerun()


def _render_question_form(st, q: dict, key_prefix: str, edit_mode: bool, idx: int | None):
    """질문 편집 폼."""
    col1, col2, col3 = st.columns(3)
    with col1:
        q_id = st.text_input("ID", value=q["id"], key=f"{key_prefix}_id")
    with col2:
        q_type = st.selectbox(
            "유형",
            ["A", "B", "C", "D", "E"],
            index=["A", "B", "C", "D", "E"].index(q.get("type", "A")),
            key=f"{key_prefix}_type",
        )
    with col3:
        q_diff = st.selectbox(
            "난이도",
            ["하", "중", "상"],
            index=["하", "중", "상"].index(q.get("difficulty", "중")),
            key=f"{key_prefix}_diff",
        )

    q_question = st.text_area(
        "질문", value=q.get("question", ""), height=80, key=f"{key_prefix}_question"
    )
    q_gt_answer = st.text_area(
        "정답 (Ground Truth)",
        value=q.get("ground_truth_answer", ""),
        height=120,
        key=f"{key_prefix}_gt",
    )
    q_gt_docs = st.text_input(
        "기대 문서 (쉼표 구분)",
        value=", ".join(q.get("ground_truth_docs", [])),
        key=f"{key_prefix}_docs",
    )
    q_meta_filter = st.text_input(
        "메타데이터 필터 (JSON)",
        value=json.dumps(q.get("metadata_filter") or {}, ensure_ascii=False),
        key=f"{key_prefix}_filter",
    )

    # C유형: history
    q_history = None
    if q_type == "C":
        history_str = json.dumps(q.get("history") or [], ensure_ascii=False, indent=2)
        q_history_raw = st.text_area(
            "대화 히스토리 (JSON)", value=history_str, height=150, key=f"{key_prefix}_history"
        )
        try:
            q_history = json.loads(q_history_raw) if q_history_raw.strip() else None
        except json.JSONDecodeError:
            st.warning("히스토리 JSON 형식이 올바르지 않습니다.")
            q_history = q.get("history")

    # 저장 버튼
    btn_label = "✏️ 수정 반영" if edit_mode else "➕ 질문 추가"
    if st.button(btn_label, type="primary", key=f"{key_prefix}_save_btn"):
        updated = {
            "id": q_id,
            "type": q_type,
            "difficulty": q_diff,
            "question": q_question,
            "ground_truth_answer": q_gt_answer,
            "ground_truth_docs": [d.strip() for d in q_gt_docs.split(",") if d.strip()],
            "metadata_filter": json.loads(q_meta_filter)
            if q_meta_filter.strip() and q_meta_filter.strip() != "{}"
            else None,
            "history": q_history,
        }

        if edit_mode and idx is not None:
            st.session_state.eval_set[idx] = updated
            st.toast(f"질문 {q_id} 수정됨", icon="✏️")
        else:
            st.session_state.eval_set.append(updated)
            st.toast(f"질문 {q_id} 추가됨", icon="➕")
        st.rerun()
