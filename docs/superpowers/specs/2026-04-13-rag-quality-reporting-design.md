# RAG 답변 품질 개선 + 노션 실험 로그 강화 설계 — BidMate RAG

**Date**: 2026-04-13
**Status**: Approved

## 개요

현재 파이프라인은 정답 문서를 찾고도 답변에 필요한 핵심 사실을 회수하지 못하는 경우가 있다. 특히 예산/연도/기관 비교 질문에서 본문에 직접 없는 값이 `metadata`에는 존재하는데, LLM 컨텍스트에는 이 값이 노출되지 않아 답변이 "문서에서 찾을 수 없음"으로 끝나는 패턴이 반복된다.

동시에 기존 노션 친화 마크다운 리포트는 메트릭과 비용은 잘 남기지만, "왜 이 실험을 했고 무엇을 바꿨는지"를 사람이 다시 채워야 한다. 이번 변경은 **답변 품질 개선**과 **실험 맥락 자동 기록**을 하나의 흐름으로 묶는다.

## 목표

- 예산/연도/기관유형처럼 `metadata`에 저장된 사실을 답변에 직접 활용할 수 있게 한다.
- 다문서 비교 질문에서 검색 결과가 비거나 한 문서로 치우치는 패턴을 줄인다.
- 기존 `bidmate-report` 결과물에 실험 가설, 변경점, 기대 효과, 대표 실패 사례, 다음 액션을 자동 삽입한다.
- 기존 CLI/UI 흐름과 산출물 위치(`artifacts/logs`, `artifacts/reports`)는 유지한다.

## 비목표

- Notion API로 페이지를 직접 생성하거나 수정하지 않는다.
- 별도 실험 관리 시스템이나 DB를 새로 만들지 않는다.
- 대규모 reranker 도입이나 새 임베딩 모델 교체는 이번 범위에 포함하지 않는다.

## 문제 정리

1. `RetrievedChunk.chunk.metadata`에는 `사업 금액`, `공개연도`, `기관유형`, `파일명`이 있지만, 현재 LLM 컨텍스트는 `[사업명 | 발주기관] + chunk.text`만 전달한다.
2. 비교형 질문은 단일 벡터 검색 한 번으로 처리되어, 다중 기관/다중 문서 질문에서 필요한 문서별 근거를 균형 있게 확보하지 못한다.
3. 섹션/표 관련 heuristic 함수가 일부 존재하지만 실제 retrieval orchestration에는 충분히 연결되지 않았다.
4. 기존 노션 리포트는 템플릿의 "사람이 작성하는 영역"이 비어 있어, 실험 맥락 보존이 약하다.

## 설계 요약

이번 변경은 두 축으로 구성한다.

1. **Quality path**
   - 컨텍스트 헤더에 핵심 metadata를 노출한다.
   - 비교형 질문에 대해 문서별 근거 확보를 위한 retrieval fallback을 추가한다.
   - 기존 section/table heuristic을 실제 retrieval 흐름에 연결한다.

2. **Reporting path**
   - 실험 노트 파일(`notes.yaml`)을 별도로 둔다.
   - 평가 실행 시 notes 경로를 run metadata에 기록한다.
   - `bidmate-report`가 notes 파일과 실제 run 결과를 함께 읽어 노션용 markdown을 더 풍부하게 생성한다.

## 상세 설계

### 1. 메타데이터 기반 컨텍스트 강화

대상 파일:
- `src/bidmate_rag/providers/llm/openai_compat.py`
- `src/bidmate_rag/providers/llm/hf_local.py`

변경 내용:
- `_build_context()`를 provider별 중복 함수로 두지 않고 공통 helper로 정리한다.
- 각 청크 앞에 아래와 같은 metadata header를 붙인다.

```text
[출처: 사업명 | 발주기관 | 파일명]
[메타: 사업 금액=..., 공개연도=..., 기관유형=..., 사업도메인=...]
```

규칙:
- 값이 비어 있거나 `nan` 성격의 값은 출력하지 않는다.
- `사업 금액`은 사람이 읽기 좋은 형식으로 포맷한다.
- metadata header는 chunk 본문보다 앞에 두어 예산/연도 질문에서 LLM이 바로 읽게 한다.
- `max_context_chars`는 유지하되, 헤더 길이를 고려해 청크 개수 산정이 안정적으로 되도록 한다.

기대 효과:
- 본문에는 금액이 없지만 metadata에 있는 예산 정보를 답변에 활용할 수 있다.
- 출처/문서명 식별이 쉬워져 비교형 답변의 정확도와 설명 가능성이 올라간다.

### 2. 비교형 질문 retrieval fallback

대상 파일:
- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/retrieval/filters.py`
- 필요 시 `src/bidmate_rag/storage/metadata_store.py`

변경 내용:
- explicit `metadata_filter`가 `{"발주 기관": {"$in": [...]}}` 또는 다중 문서 조건일 때, 전체 검색 1회만 하지 않고 **문서/기관별 소규모 검색을 나눠 수행한 뒤 merge**한다.
- merge 시 동일 문서 청크만 상위권을 독점하지 않도록 간단한 diversification 규칙을 둔다.
- `extract_section_hint()`가 반환한 값은 `where_document` substring 검색만 쓰지 말고, `section` metadata와 함께 활용한다.
- 예산/금액/일정/평가처럼 구조화된 질문은 가능한 경우 해당 section 청크를 우선 확보한다.
- 기존 `should_boost_tables()`는 이번 범위에서 최소 기능으로 실제 검색 결과 정렬에 반영한다.

세부 규칙:
- 기본 경로는 기존 single query retrieval을 유지한다.
- 다중 비교 경로는 아래 조건 중 하나일 때만 활성화한다.
  - explicit filter에 `$in`이 포함된 경우
  - 질문에서 2개 이상 기관이 감지된 경우
- 문서/기관별 검색 결과는 round-robin 방식으로 섞어 최종 `top_k`를 채운다.
- 중복 chunk_id는 제거한다.

기대 효과:
- 비교 질문에서 한 문서만 검색되거나 아예 빈 검색으로 끝나는 패턴이 줄어든다.
- 문서별 핵심 근거를 골고루 확보해 차이/합산/우선순위 질문에 대응하기 쉬워진다.

### 3. 실험 노트 파일 추가

새 파일 위치:
- `configs/experiments/notes/<note_name>.yaml`

예시 구조:

```yaml
title: budget-metadata-context
hypothesis:
  - 예산이 metadata에만 존재하는 문서도 답변 가능해질 것이다.
changes:
  - LLM 컨텍스트 헤더에 사업 금액/공개연도/기관유형 추가
  - 다문서 비교 질문에서 기관별 retrieval merge 추가
expected_outcome:
  - 예산 질문 무응답 감소
  - 비교형 질문 retrieval hit 개선
next_actions:
  - judge 기준으로 실패 유형 재분류
failure_cases:
  - question_id: Q007
    why_watch: 다문서 예산 차액 계산
  - question_id: Q014
    why_watch: 다문서 합산 + 공통요소 추출
```

설계 결정:
- notes는 실험 config와 분리된 별도 파일로 관리한다.
- 실험 config에는 optional `notes_path`만 추가한다.
- notes 파일이 없으면 기존 동작을 유지한다.

### 4. Runtime / meta 기록

대상 파일:
- `src/bidmate_rag/config/settings.py`
- `src/bidmate_rag/evaluation/pipeline.py`
- `scripts/run_experiment.py` 경유 config snapshot

변경 내용:
- `ExperimentConfig`에 `notes_path: str | None`를 추가한다.
- 평가 실행 시 `run meta.json`에 `notes_path`를 기록한다.
- `config_snapshot`에도 남겨 나중에 어떤 실험 노트와 연결된 run인지 추적 가능하게 한다.

호환성:
- 기존 experiment yaml은 `notes_path`가 없어도 그대로 동작한다.

### 5. 노션 친화 report 자동 채움

대상 파일:
- `src/bidmate_rag/tracking/markdown_report.py`
- `src/bidmate_rag/tracking/templates.py`
- `src/bidmate_rag/cli/report.py`

변경 내용:
- `load_report_data()`가 `meta.json`의 `notes_path`를 읽어 notes 파일을 로드한다.
- `ReportData`에 `experiment_notes` 필드를 추가한다.
- 템플릿의 "사람이 작성하는 영역"을 완전 자동화하지는 않되, 아래 항목은 notes + run 결과로 자동 채운다.
  - 실험 목적
  - 가설
  - 변경점
  - 기대 효과
  - 대표 실패 사례
  - 다음 액션
- 대표 실패 사례는 우선순위를 다음 순서로 정한다.
  1. notes 파일에 명시된 `failure_cases.question_id`
  2. retrieval miss (`retrieved_chunks == []`)
  3. judge score가 낮은 샘플

출력 원칙:
- notes가 있으면 해당 내용을 우선 사용한다.
- notes가 없으면 기존 템플릿의 빈 섹션을 유지한다.
- run 결과에서 자동 채운 실패 사례는 질문/실제 답변/관찰 포인트 중심으로 짧게 요약한다.

## 데이터 흐름

```text
experiment.yaml
  └── notes_path -> configs/experiments/notes/foo.yaml

run_experiment / bidmate-eval
  └── runtime config load
      └── execute_evaluation()
          └── runs/{run_id}.meta.json 에 notes_path 기록

bidmate-report --run-id ...
  └── load meta.json
      └── notes.yaml 로드
      └── jsonl/parquet 결과 로드
      └── notes + metrics + 실패 사례를 합쳐 markdown 생성
```

## 오류 처리

- `notes_path`가 config에 있으나 파일이 없으면:
  - report 생성은 실패시키지 않는다.
  - warning 로그를 남기고 notes 없는 리포트로 생성한다.
- notes 파일 schema가 잘못되면:
  - 필수 필드 없는 경우에도 가능한 값만 사용한다.
  - report 전체 실패보다 부분 생성을 우선한다.
- retrieval fallback이 빈 결과를 내면:
  - 기존 single query 결과로 한 번 더 폴백한다.
- metadata 값이 `nan`, 빈 문자열, `None`이면:
  - 컨텍스트 헤더에서 생략한다.

## 테스트 전략

### Unit tests

- metadata header가 금액/연도/기관유형을 올바르게 렌더링하는지
- 다중 기관 filter에서 retrieval merge가 수행되는지
- 단일 질문에서는 기존 retrieval 경로가 유지되는지
- notes 파일 로딩이 정상/누락/부분 필드 케이스에서 모두 안전한지
- report가 notes 내용을 markdown에 반영하는지

### Integration tests

- 예산 질문에서 metadata 기반 답변이 가능한지
- 다문서 비교 질문에서 `retrieved_chunks`가 비지 않도록 개선되는지
- `bidmate-report`가 notes_path가 있는 run에 대해 자동 채움 섹션을 생성하는지

## 구현 순서

1. context builder 공통화 + metadata header 추가
2. retrieval 비교형 fallback 및 section/table 연결
3. experiment config에 `notes_path` 추가
4. report loader/template에 notes 연동
5. representative failure 자동 선택 로직 추가
6. unit/integration test 추가

## 리스크와 대응

- 리스크: metadata header가 너무 길어져 context budget을 잠식할 수 있음
  - 대응: 출력 필드 수를 제한하고 값 포맷을 짧게 유지
- 리스크: 비교형 retrieval merge가 기존 점수 정렬과 충돌할 수 있음
  - 대응: 다중 비교 질문에서만 조건부 활성화
- 리스크: report 템플릿이 과도하게 길어질 수 있음
  - 대응: 실패 사례는 최대 2건, bullet도 짧게 유지

## 성공 기준

- 기존 실패 예시(Q001, Q007 계열)에서 metadata 기반 답변 회수율이 개선된다.
- 다문서 비교 질문에서 `retrieved_chunks == []` 사례 수가 감소한다.
- 생성된 markdown 리포트가 노션에 붙여넣기 전 추가 수작업 없이 실험 맥락을 전달할 수 있다.
