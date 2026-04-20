# app/ — Streamlit UI 규칙

## 구조

```
app/
├── main.py           # 메인 앱 (사이드바 + 3탭)
├── eval_ui.py        # 평가 탭 (4개 서브탭)
└── api/routes.py     # UI 헬퍼 (파이프라인 호출, 데이터 로딩)
```

## 실행

```bash
PYTHONPATH=. uv run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

`PYTHONPATH=.` 필수 — `app.api.routes` import에 필요.

## UI 구성

### 사이드바
- 시나리오 A/B 체크박스
- Provider 선택 (B: gpt-5→mini→nano 순서)
- 검색 설정 (Top-K, 검색 모드, 메타데이터 필터)
- 생성 설정 (컨텍스트 길이, 시스템 프롬프트 편집)
- 세션 통계 + 초기화/내보내기

### 탭
1. **💬 라이브 데모**: 채팅 UI + 디버그 패널 (4단계)
2. **📁 문서 목록**: RFP 문서 브라우징 + 필터 + 상세
3. **📊 평가**: 실행/디버깅/비교/편집 4개 서브탭

## 코딩 규칙

### Streamlit 버전 호환
- `st.dataframe()`: `width="stretch"` 사용 (`use_container_width` 삭제됨)
- `st.button()`: `use_container_width=True` 여전히 유효
- `st.download_button()`: `width="stretch"` 사용

### session_state 키
- `messages`: 채팅 히스토리
- `session_stats`: 질문수/토큰/시간 누적
- `custom_prompt`: 수정된 시스템 프롬프트
- `pending_example`: 예시 질문 클릭 시 전달
- `eval_set`: 평가셋 (session 편집용)
- `eval_results`: 평가 실행 결과 (run_id → results)

### 디버그 패널 key 충돌 방지
text_area의 key에 `hashlib.md5(str(meta)).hexdigest()[:8]` 사용.
latency 같은 중복 가능한 값을 key로 쓰지 말 것.

### Provider 정렬
`eval_ui.py`의 `_render_scenario_provider_selector()`를 사용.
B 시나리오(gpt-5→mini→nano) 우선, A 시나리오 이름순.

### 평가셋 파일
- 메인 위치: `data/eval/eval_v1/eval_batch_*.csv` (CSV-first, 버전 디렉토리)
- 새 버전 추가 시 `data/eval/eval_v2/`처럼 디렉토리만 만들면 코드가 자동으로
  가장 높은 버전을 사용 (`bidmate_rag.evaluation.dataset.find_latest_eval_dir`)
- JSON 폴백: `data/eval/eval_set.json`
- session에서 편집 → "💾 저장"으로 파일 반영
