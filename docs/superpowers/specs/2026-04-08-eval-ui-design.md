# 평가 UI 설계 — BidMate RAG

**Date**: 2026-04-08
**Status**: Approved

## 개요

현재 "📊 평가 비교" 탭(결과 파일 뷰어)을 **능동적 평가 도구**로 확장한다.
4개 서브탭으로 구성하여 평가셋 실행, 디버깅, 비교, 편집을 UI에서 수행 가능하게 한다.

## 평가셋 스키마

```json
{
  "id": "Q001",
  "type": "A",
  "difficulty": "중",
  "question": "국민연금공단이 발주한 이러닝시스템 사업 요구사항을 정리해 줘",
  "ground_truth_answer": "교육운영(FUR-001~005), 시스템기능(SFR-001~008)...",
  "ground_truth_docs": ["이러닝시스템 운영 용역"],
  "metadata_filter": {"발주기관": "국민연금공단"},
  "history": null
}
```

| 컬럼 | 설명 | 목적 |
|---|---|---|
| id | 질문 고유 식별자 | 데이터 관리 |
| type | A~E 유형 | 유형별 통계 |
| difficulty | 상/중/하 | 난이도별 집계 |
| question | 사용자 질문 | 테스트 입력 |
| ground_truth_answer | 사람 작성 정답 | 생성 평가 기준 |
| ground_truth_docs | 정답 문서 리스트 | 검색 평가 기준 |
| metadata_filter | 의도된 필터 | 필터 정확도 측정 |
| history | C유형 대화 내역 | 맥락 유지 검증 |

## 서브탭 구조

```
📊 평가
├── 🏃 평가 실행
├── 🔍 질문 디버깅
├── ⚖️ 결과 비교
└── ✏️ 평가셋 편집
```

### 서브탭 1: 🏃 평가 실행

- Provider config + Top-K + 검색 모드 선택
- 실행 범위: 전체 / 유형별 / 난이도별
- 진행률 바 + 질문별 실시간 결과
- 완료 시 요약: 유형별 + 난이도별 + 필터 정확도
- 결과 자동 저장 (artifacts/logs/)

### 서브탭 2: 🔍 질문 디버깅

- 질문 선택 (유형/난이도 필터)
- 3단계 표시:
  - 검색: 적용 필터 vs 의도된 필터(metadata_filter) 비교, Top-k 청크
  - 생성: 답변 vs ground_truth_answer 나란히
  - Judge: 4개 점수 + reasoning
- "재실행" 버튼

### 서브탭 3: ⚖️ 결과 비교

- 좌/우 2개 run 선택
- 비교 축: 모델, Top-K, 검색 모드 자유 조합
- 질문별 나란히 + 유형별/난이도별 집계 차트

### 서브탭 4: ✏️ 평가셋 편집

- 질문 목록 테이블
- 편집 폼 (전 필드)
- C유형 history JSON 편집기
- session 편집 → 파일 저장 분리

## 기술 사항

- 평가셋 파일: `data/eval/eval_set.json`
- 결과 저장: `artifacts/logs/runs/*.jsonl` + `artifacts/logs/benchmarks/*.parquet`
- 실행 엔진: `app/api/routes.py` → `evaluation/runner.py`
- UI: Streamlit `st.tabs` 중첩

## 질문 유형별 평가 기준

| 유형 | 기준 | 설명 |
|---|---|---|
| A. 단일문서 | 추출 정밀성 | 수치 데이터 오차 없이 추출 |
| B. 다문서비교 | 정보 통합력 | 여러 RFP 대조하여 차이점 요약 |
| C. 후속질문 | 문맥 유지력 | 지시어 정확히 이해 |
| D. 무응답 | 근거 기반 답변 | 할루시네이션 없이 거부 |
| E. 모호한질의 | 검색 견고성 | 부정확한 입력에도 올바른 문서 검색 |
