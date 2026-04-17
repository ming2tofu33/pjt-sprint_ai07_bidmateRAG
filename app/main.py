"""BidMate RAG — Streamlit UI with chat interface and sidebar controls."""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from app.api.routes import (
    list_provider_configs,
    list_chunking_configs,
    list_scenario_a_embeddings,
    list_scenario_a_llms,
    load_benchmark_frames,
    load_metadata_options,
    load_run_records,
    run_live_query,
)
from app.eval_ui import render_eval_tabs

EXAMPLE_QUESTIONS = [
    "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘",
    "한국원자력연구원 선량평가시스템 고도화 사업의 목적을 알려줘",
    "고려대학교 차세대 포털이랑 광주과학기술원 학사 시스템을 비교해줘",
    "교육 관련 사업 찾아줘",
    "5억 이상 대규모 시스템 구축 사업이 있어?",
    "기초과학연구원 극저온시스템에서 AI 기반 예측 요구사항이 있나?",
]


def _build_chat_history(messages: list[dict]) -> list[dict[str, str]]:
    """Run 직전까지의 user/assistant 메시지만 멀티턴 history로 정리한다."""
    history: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        history.append({"role": role, "content": content})
    return history


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="BidMate RAG",
        page_icon="📄",
        layout="wide",
    )

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"queries": 0, "total_tokens": 0, "total_latency": 0}

    # 메타데이터 옵션 로딩 (캐싱)
    meta_options = load_metadata_options()

    # ── 사이드바 ──
    with st.sidebar:
        st.header("⚙️ 설정")

        # 1) 시나리오 + Provider 선택
        provider_configs = list_provider_configs()
        if not provider_configs:
            st.warning("Provider config가 없습니다.")
            st.stop()

        # 시나리오 체크박스
        st.subheader("🔬 시나리오")
        col_a, col_b = st.columns(2)
        with col_a:
            show_a = st.checkbox("A (로컬)", value=False)
        with col_b:
            show_b = st.checkbox("B (API)", value=True)

        # 시나리오별 필터링 + 정렬
        import yaml as _yaml
        def _get_scenario(path):
            try:
                return _yaml.safe_load(path.read_text()).get("scenario", "")
            except Exception:
                return ""

        def _get_model(path):
            try:
                return _yaml.safe_load(path.read_text()).get("model", "")
            except Exception:
                return ""

        # B 시나리오 정렬: gpt-5 → gpt-5-mini → gpt-5-nano
        B_ORDER = {"gpt-5": 0, "gpt-5-mini": 1, "gpt-5-nano": 2}

        filtered_configs = []
        for p in provider_configs:
            scenario = _get_scenario(p)
            if scenario == "scenario_a" and show_a:
                filtered_configs.append(p)
            elif scenario == "scenario_b" and show_b:
                filtered_configs.append(p)
            elif scenario not in ("scenario_a", "scenario_b"):
                if show_a or show_b:
                    filtered_configs.append(p)

        # 정렬: B(gpt-5 순서) → A(이름순)
        filtered_configs.sort(key=lambda p: (
            0 if _get_scenario(p) == "scenario_b" else 1,
            B_ORDER.get(_get_model(p), 99),
            p.stem,
        ))

        if not filtered_configs:
            st.warning("선택한 시나리오에 해당하는 Provider가 없습니다.")
            st.stop()

        def _format_provider(p):
            scenario = _get_scenario(p)
            model = _get_model(p)
            tag = "🅰️" if scenario == "scenario_a" else "🅱️"
            return f"{tag} {model} ({p.stem})"

        selected_provider = st.selectbox(
            "LLM Provider",
            filtered_configs,
            format_func=_format_provider,
        )

        # 2) 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("Top-K (검색 청크 수)", min_value=1, max_value=20, value=5)

        search_mode = st.radio(
            "검색 모드",
            ["🤖 자동", "🎛️ 수동", "🔓 필터 없음"],
            captions=[
                "질문에서 발주기관/도메인 자동 감지",
                "아래에서 직접 필터 조합 선택",
                "벡터 유사도만 사용 (디버깅용)",
            ],
        )

        # 3) 메타데이터 필터
        manual_filters = {}
        if search_mode == "🎛️ 수동":
            st.subheader("🏷️ 필터 조합")
            st.caption("여러 필터를 동시에 적용할 수 있습니다.")

            # 발주 기관 (ChromaDB 메타 키는 공백 포함이라 정확히 일치시켜야 매칭됨)
            if meta_options["agencies"]:
                selected_agency = st.selectbox("발주 기관", ["전체"] + meta_options["agencies"])
                if selected_agency != "전체":
                    manual_filters["발주 기관"] = selected_agency

            # 사업도메인 (multiselect)
            if meta_options["domains"]:
                selected_domains = st.multiselect("사업도메인 (복수 선택 가능)", meta_options["domains"])
                if selected_domains:
                    if len(selected_domains) == 1:
                        manual_filters["사업도메인"] = selected_domains[0]
                    else:
                        manual_filters["사업도메인"] = {"$in": selected_domains}

            # 기관유형 (multiselect)
            if meta_options["agency_types"]:
                selected_types = st.multiselect("기관유형 (복수 선택 가능)", meta_options["agency_types"])
                if selected_types:
                    if len(selected_types) == 1:
                        manual_filters["기관유형"] = selected_types[0]
                    else:
                        manual_filters["기관유형"] = {"$in": selected_types}

            # 사업 금액 범위 (ChromaDB 메타 키는 '사업 금액' — 공백 포함)
            budget_filter = st.selectbox("사업 금액", ["전체", "1억 이하", "1~5억", "5~10억", "10억 이상"])
            if budget_filter == "1억 이하":
                manual_filters["사업 금액"] = {"$lte": 100_000_000}
            elif budget_filter == "1~5억":
                manual_filters["사업 금액"] = {"$gte": 100_000_000, "$lte": 500_000_000}
            elif budget_filter == "5~10억":
                manual_filters["사업 금액"] = {"$gte": 500_000_000, "$lte": 1_000_000_000}
            elif budget_filter == "10억 이상":
                manual_filters["사업 금액"] = {"$gte": 1_000_000_000}

            # 적용된 필터 뱃지 표시
            if manual_filters:
                tags = []
                for k, v in manual_filters.items():
                    if isinstance(v, dict):
                        tags.append(f"`{k}: {v}`")
                    else:
                        tags.append(f"`{k}: {v}`")
                st.markdown(f"**적용 필터 ({len(manual_filters)}개):** " + " · ".join(tags))
            else:
                st.info("필터를 선택하면 여기에 표시됩니다.")

        elif search_mode == "🔓 필터 없음":
            manual_filters = {"_no_filter": True}
            st.caption("메타데이터 필터 없이 순수 벡터 유사도로만 검색합니다.")

        # 4) 생성 설정
        st.subheader("✏️ 생성 설정")
        max_context_chars = st.slider(
            "컨텍스트 최대 길이",
            min_value=2000, max_value=16000, value=8000, step=1000,
            help="검색된 청크를 LLM에 보낼 때 최대 글자 수",
        )

        # 시스템 프롬프트 편집
        from bidmate_rag.config.prompts import SYSTEM_PROMPT as DEFAULT_PROMPT
        with st.expander("📝 시스템 프롬프트", expanded=False):
            custom_prompt = st.text_area(
                "프롬프트 편집",
                value=st.session_state.get("custom_prompt", DEFAULT_PROMPT),
                height=250,
                key="prompt_editor",
                help="수정 후 질문하면 변경된 프롬프트가 적용됩니다",
            )
            st.session_state["custom_prompt"] = custom_prompt

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button("↩️ 기본값 복원", use_container_width=True, key="reset_prompt"):
                    st.session_state["custom_prompt"] = DEFAULT_PROMPT
                    st.rerun()
            with col_p2:
                prompt_changed = custom_prompt.strip() != DEFAULT_PROMPT.strip()
                if prompt_changed:
                    st.caption("✏️ 수정됨")
        if show_a and not show_b:
            st.subheader("🅰️ 시나리오 A 설정")
            
            # 임베딩 모델 선택
            embedding_configs = list_scenario_a_embeddings()
            if embedding_configs:
                selected_embedding = st.selectbox(
                    "임베딩 모델",
                    embedding_configs,
                    format_func=lambda p: p.stem,
                    key="scenario_a_embedding",
                )
            else:
                st.warning("임베딩 모델 설정이 없습니다.")
                selected_embedding = None

            # LLM 모델 선택
            llm_configs = list_scenario_a_llms()
            if llm_configs:
                selected_llm = st.selectbox(
                    "LLM 모델",
                    llm_configs,
                    format_func=lambda p: p.stem,
                    key="scenario_a_llm",
                )
            else:
                st.warning("LLM 모델 설정이 없습니다.")
                selected_llm = None
        # 청킹 전략 선택    
        st.subheader("📦 청킹 전략")
        chunking_configs = list_chunking_configs()
        default_idx = next(
            (i for i, p in enumerate(chunking_configs) if "1000_150" in p.stem), 0
        )
        selected_chunking = st.selectbox(
            "청킹 전략 선택",
            chunking_configs,
            index=default_idx,
            format_func=lambda p: p.stem,
        )

        # 5) 모델 정보
        st.subheader("📊 모델 정보")
        try:
            import yaml
            config = yaml.safe_load(selected_provider.read_text())
            st.markdown(f"""
- **Provider**: `{config.get('provider', '-')}`
- **Model**: `{config.get('model', '-')}`
- **Embedding**: `{config.get('embedding_model', '-')}`
- **Scenario**: `{config.get('scenario', '-')}`
            """)
        except Exception:
            pass

        # 5) 응답 통계
        stats = st.session_state.session_stats
        if stats["queries"] > 0:
            st.subheader("📈 세션 통계")
            cols = st.columns(2)
            cols[0].metric("질문 수", f"{stats['queries']}회")
            cols[1].metric("총 토큰", f"{stats['total_tokens']:,}")
            cols = st.columns(2)
            avg_latency = stats["total_latency"] / stats["queries"]
            cols[0].metric("평균 응답", f"{avg_latency / 1000:.1f}초")
            est_cost = stats["total_tokens"] * 0.15 / 1_000_000
            cols[1].metric("예상 비용", f"${est_cost:.4f}")

        st.divider()

        # 대화 히스토리 내보내기
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 초기화", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_stats = {"queries": 0, "total_tokens": 0, "total_latency": 0}
                st.session_state.pop("pending_example", None)
                st.rerun()
        with col2:
            if st.session_state.messages:
                export = []
                for msg in st.session_state.messages:
                    entry = {"role": msg["role"], "content": msg["content"]}
                    if msg.get("metadata"):
                        entry["metadata"] = {
                            k: v for k, v in msg["metadata"].items() if k != "retrieved"
                        }
                    export.append(entry)
                st.download_button(
                    "💾 내보내기",
                    data=json.dumps(export, ensure_ascii=False, indent=2),
                    file_name="bidmate_rag_chat.json",
                    mime="application/json",
                    width="stretch",
                )

        st.caption("BidMate RAG v0.1 — 시나리오 B Baseline")

    # ── 탭 구성 ──
    demo_tab, docs_tab, eval_tab = st.tabs(["💬 라이브 데모", "📁 문서 목록", "📊 평가"])

    # ── 탭 1: 채팅 UI ──
    with demo_tab:
        # 채팅 스타일 CSS 주입
        st.markdown("""
        <style>
        /* 채팅 메시지 영역 — 고정 높이 + 스크롤 */
        section[data-testid="stChatFlow"] {
            height: 65vh;
            overflow-y: auto;
            padding-bottom: 1rem;
        }

        /* 채팅 입력창 — 하단 고정 강화 */
        .stChatInput {
            position: sticky;
            bottom: 0;
            background: var(--background-color);
            z-index: 100;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
        }

        /* 사용자 메시지 — 오른쪽 정렬감 */
        [data-testid="stChatMessage"][data-testid-type="user"] {
            background-color: rgba(59, 130, 246, 0.08);
            border-radius: 12px;
            margin-bottom: 0.5rem;
        }

        /* 어시스턴트 메시지 */
        [data-testid="stChatMessage"][data-testid-type="assistant"] {
            background-color: rgba(128, 128, 128, 0.05);
            border-radius: 12px;
            margin-bottom: 0.5rem;
        }

        /* 메시지 간 간격 */
        .stChatMessage {
            margin-bottom: 0.75rem;
        }

        /* 예시 질문 버튼 스타일 */
        .example-container .stButton > button {
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            font-size: 0.85rem;
            padding: 0.4rem 0.8rem;
            transition: all 0.15s ease;
        }
        .example-container .stButton > button:hover {
            border-color: rgba(59, 130, 246, 0.6);
            background-color: rgba(59, 130, 246, 0.08);
        }
        </style>
        """, unsafe_allow_html=True)

        # 빈 상태: 예시 질문
        if not st.session_state.messages:
            st.markdown("#### 📄 RFP 문서 질의응답")
            st.caption("RFP 문서 내용에 대해 질문하세요. 사업명이나 발주기관을 언급하면 더 정확합니다.")
            st.markdown("")

            # 예시 질문 칩 — 2행 3열
            st.markdown('<div class="example-container">', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                label = q[:28] + "..." if len(q) > 28 else q
                if cols[i % 3].button(label, key=f"example_{i}", use_container_width=True):
                    st.session_state["pending_example"] = q
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 메시지가 있을 때 — 간결한 헤더
            st.caption(f"💬 대화 {len([m for m in st.session_state.messages if m['role']=='user'])}회")

        # 기존 메시지 렌더링
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    _render_metadata_expander(st, msg["metadata"])

        # 예시 질문 처리
        pending = st.session_state.pop("pending_example", None)
        prompt = pending or st.chat_input("질문을 입력하세요 (예: 국민연금공단 이러닝시스템 요구사항)")

        if prompt:
            chat_history = _build_chat_history(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("RAG 파이프라인 실행 중...", expanded=True)
                try:
                    status.write("🔍 관련 문서 검색 중...")

                    # 수동 필터 전달
                    filters_to_pass = manual_filters if manual_filters else None

                    # 시스템 프롬프트 (수정된 경우만 전달)
                    active_prompt = st.session_state.get("custom_prompt", "")
                    prompt_override = active_prompt if active_prompt.strip() != DEFAULT_PROMPT.strip() else None

                    result = run_live_query(
                        question=prompt,
                        provider_config_path=selected_provider,
                        experiment_config_path=selected_chunking,
                        top_k=top_k,
                        manual_filters=filters_to_pass,
                        chat_history=chat_history,
                        system_prompt=prompt_override,
                        max_context_chars=max_context_chars,
                    )
                    status.write("✏️ 답변 생성 완료")
                    status.update(label="완료", state="complete", expanded=False)

                    st.markdown(result.answer)

                    retrieved_records = []
                    if result.retrieved_chunks:
                        retrieved_records = [
                            {
                                "순위": c.rank,
                                "유사도": round(c.score, 4),
                                "사업명": c.chunk.metadata.get("사업명", "")[:25],
                                "발주기관": c.chunk.metadata.get("발주기관", ""),
                                "도메인": c.chunk.metadata.get("사업도메인", ""),
                                "유형": c.chunk.content_type,
                                "내용": c.chunk.text[:100],
                            }
                            for c in result.retrieved_chunks
                        ]

                    meta = {
                        "chunks": len(result.retrieved_chunks),
                        "context_chars": len(result.context) if result.context else 0,
                        "tokens": result.token_usage.get("total", 0),
                        "prompt_tokens": result.token_usage.get("prompt", 0),
                        "completion_tokens": result.token_usage.get("completion", 0),
                        "latency": round(result.latency_ms),
                        "model": getattr(result, "llm_model", "-"),
                        "retrieved": retrieved_records,
                        "applied_filter": getattr(result, "metadata_filter", None) or filters_to_pass,
                        "filter_info": {
                            "검색 모드": search_mode,
                            "적용 필터": str(filters_to_pass) if filters_to_pass else "자동 추출",
                            "청킹 전략": selected_chunking.stem,
                            "Top-K": top_k,
                            "컨텍스트 길이": f"{max_context_chars:,}자",
                            "프롬프트": "커스텀" if prompt_override else "기본",
                        },
                        "context_preview": result.context if hasattr(result, "context") else "",
                        "system_prompt": getattr(result, "system_prompt", ""),
                    }

                    _render_metadata_expander(st, meta)

                    st.toast(
                        f"답변 완료 — {meta['tokens']:,}토큰, {meta["latency"] / 1000:.1f}초",
                        icon="✅",
                    )

                    # 세션 통계 업데이트
                    st.session_state.session_stats["queries"] += 1
                    st.session_state.session_stats["total_tokens"] += meta["tokens"]
                    st.session_state.session_stats["total_latency"] += meta["latency"]

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "metadata": meta,
                    })

                except Exception as exc:
                    status.update(label="오류 발생", state="error", expanded=False)
                    error_msg = str(exc)
                    st.error(f"오류가 발생했습니다: {error_msg}")
                    st.info("💡 해결 방법:\n"
                            "- `.env` 파일에 OPENAI_API_KEY가 설정되어 있는지 확인\n"
                            "- 사이드바에서 올바른 Provider를 선택했는지 확인\n"
                            "- ChromaDB 인덱스가 생성되어 있는지 확인 (`05_embedding` 실행)")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"오류: {error_msg}",
                    })

    # ── 탭 2: 문서 목록 ──
    with docs_tab:
        st.subheader("RFP 문서 목록")
        # 가장 최근 실험별 metadata를 자동 사용 (없으면 top-level fallback)
        from bidmate_rag.evaluation.dataset import find_latest_metadata_path
        chunks_path = find_latest_metadata_path()
        if not chunks_path.exists():
            st.info("문서 데이터가 없습니다. 파이프라인을 실행해주세요:\n\n"
                    "`uv run python scripts/ingest_data.py`")
        else:
            import pandas as pd
            docs_df = pd.read_parquet(chunks_path)

            # 필터 영역
            filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
            with filter_col1:
                search_term = st.text_input("🔍 검색", placeholder="사업명 또는 발주기관 키워드...")
            with filter_col2:
                domain_options = ["전체"] + sorted(docs_df["사업도메인"].dropna().unique().tolist()) if "사업도메인" in docs_df.columns else ["전체"]
                doc_domain = st.selectbox("도메인", domain_options, key="doc_domain")
            with filter_col3:
                type_options = ["전체"] + sorted(docs_df["기관유형"].dropna().unique().tolist()) if "기관유형" in docs_df.columns else ["전체"]
                doc_type = st.selectbox("기관유형", type_options, key="doc_type")

            # 필터 적용
            if search_term:
                docs_df = docs_df[
                    docs_df["사업명"].str.contains(search_term, case=False, na=False) |
                    docs_df["발주 기관"].str.contains(search_term, case=False, na=False)
                ]
            if doc_domain != "전체" and "사업도메인" in docs_df.columns:
                docs_df = docs_df[docs_df["사업도메인"] == doc_domain]
            if doc_type != "전체" and "기관유형" in docs_df.columns:
                docs_df = docs_df[docs_df["기관유형"] == doc_type]

            st.caption(f"총 {len(docs_df)}건")

            display_cols = ["사업명", "발주 기관", "사업 금액", "기관유형", "사업도메인", "정제_글자수"]
            available_cols = [c for c in display_cols if c in docs_df.columns]
            display = docs_df[available_cols].copy()
            if "사업 금액" in display.columns:
                display["사업 금액"] = display["사업 금액"].apply(
                    lambda x: f"{x/1e8:.1f}억" if x and x > 0 else "-"
                )
            st.dataframe(display, width="stretch", height=400)

            # 문서 상세 보기
            if len(docs_df) > 0:
                selected_doc = st.selectbox("문서 상세 보기", docs_df["사업명"].tolist())
                if selected_doc:
                    doc = docs_df[docs_df["사업명"] == selected_doc].iloc[0]
                    with st.expander(f"📄 {selected_doc}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("발주기관", doc.get("발주 기관", "-"))
                        budget = doc.get("사업 금액", 0)
                        col2.metric("사업금액", f"{budget/1e8:.1f}억" if budget and budget > 0 else "-")
                        col3.metric("본문 글자수", f"{doc.get('정제_글자수', 0):,}자")
                        col4.metric("도메인", doc.get("사업도메인", "-"))

                        if "사업 요약" in doc.index and doc["사업 요약"]:
                            st.markdown(f"**사업 요약**: {doc['사업 요약']}")

                        if "본문_정제" in doc.index:
                            st.text_area("본문 미리보기 (앞 2000자)", doc["본문_정제"][:2000], height=300, key="doc_preview")

                        # 이 문서에 대해 질문하기
                        agency = doc.get("발주 기관", "")
                        project = doc.get("사업명", "")
                        if st.button("💬 이 문서에 대해 질문하기", key="ask_doc"):
                            st.session_state["pending_example"] = f"{agency} {project} 사업 요구사항을 정리해 줘"
                            st.toast("💬 라이브 데모 탭으로 이동하세요", icon="👆")
                            st.rerun()

    # ── 탭 3: 평가 ──
    with eval_tab:
        render_eval_tabs(st, run_live_query, list_provider_configs, list_chunking_configs, list_scenario_a_embeddings, list_scenario_a_llms, load_benchmark_frames, load_run_records)


def _render_debug_panel(st_module, meta: dict) -> None:
    """4단계 파이프라인 디버그 패널."""

    # 요약 메트릭 (항상 표시)
    cols = st_module.columns(4)
    cols[0].metric("검색 청크", f"{meta.get('chunks', '-')}개")
    cols[1].metric("컨텍스트", f"{meta.get('context_chars', 0):,}자")
    cols[2].metric("토큰", f"{meta.get('tokens', 0):,}")
    cols[3].metric("응답 시간", f"{meta.get('latency', 0) / 1000:.1f}초")

    # 1단계: 필터 추출
    with st_module.expander("1️⃣ 필터 추출", expanded=False):
        filter_info = meta.get("filter_info", {})
        if filter_info:
            for k, v in filter_info.items():
                st_module.markdown(f"- **{k}**: `{v}`")
        else:
            applied = meta.get("applied_filter")
            if applied:
                st_module.json(applied)
            else:
                st_module.caption("필터 없음 (벡터 검색만 사용)")

    # 2단계: 검색 결과
    with st_module.expander("2️⃣ 검색 결과", expanded=False):
        if meta.get("retrieved"):
            st_module.dataframe(meta["retrieved"], width="stretch")
        else:
            st_module.caption("검색 결과 없음")

    # 3단계: 컨텍스트 + 프롬프트
    import hashlib
    _uid = hashlib.md5(str(meta).encode()).hexdigest()[:8]
    with st_module.expander("3️⃣ 컨텍스트 & 프롬프트", expanded=False):
        if meta.get("context_preview"):
            st_module.text_area(
                "컨텍스트 (앞 2000자)",
                meta["context_preview"][:2000],
                height=200,
                disabled=True,
                key=f"ctx_{_uid}",
            )
        if meta.get("system_prompt"):
            st_module.text_area(
                "시스템 프롬프트",
                meta["system_prompt"],
                height=150,
                disabled=True,
                key=f"sys_{_uid}",
            )

    # 4단계: LLM 응답 메타
    with st_module.expander("4️⃣ LLM 응답 상세", expanded=False):
        detail_cols = st_module.columns(3)
        detail_cols[0].markdown(f"**모델**: `{meta.get('model', '-')}`")
        detail_cols[1].markdown(f"**입력 토큰**: `{meta.get('prompt_tokens', '-')}`")
        detail_cols[2].markdown(f"**출력 토큰**: `{meta.get('completion_tokens', '-')}`")


def _render_metadata_expander(st_module, meta: dict) -> None:
    """디버그 패널 래퍼 (호환성 유지)."""
    _render_debug_panel(st_module, meta)


def main() -> None:
    if not _running_under_streamlit():
        print("BidMate RAG scaffold is ready.")
        return
    _render_streamlit_app()


if __name__ == "__main__":
    main()
