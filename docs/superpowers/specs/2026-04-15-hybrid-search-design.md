# 멀티턴 메모리형 하이브리드 검색 설계

## 요약

이번 설계의 목표는 단순한 `history 붙은 후속 질문 처리` 수준을 넘어서, 아래 구조를 기준으로 멀티턴 검색과 응답 생성을 다시 정렬하는 것이다.

`사용자 질문 + 대화 이력 -> LLM Query Rewriting -> Hybrid Retrieval(Dense + Sparse) -> 선택적 Cross-Encoder Reranking -> Summary Buffer Memory + Slot Memory -> LLM 응답 생성`

핵심은 두 가지다.

1. `그거 평가기준 알려줘` 같은 짧은 후속 질문을 검색 가능한 독립 쿼리로 바꾼다.
2. 검색이 끝난 뒤에도 대화 문맥을 잃지 않도록 메모리 계층을 별도로 유지한다.

이 설계는 `추가 YAML 파일을 만들지 않고`, 기존 `configs/retrieval.yaml`만 확장해서 제어하는 것을 전제로 한다.

## 현재 상태와 문제

현재 코드베이스에는 아래 요소가 일부만 존재한다.

- 최근 대화 일부를 참고한 재작성
- Dense + Sparse 기반 하이브리드 검색
- 선택적 Cross-Encoder 재정렬

하지만 아래 요소는 아직 없다.

- Summary Buffer Memory
- 구조화된 Slot Memory
- CLI에서 재작성 결과와 메모리 상태를 바로 확인하는 디버그 출력
- 멀티턴 추가 비용을 분리해서 보여주는 비용 기록

즉 지금 구조는 `최근 몇 턴 참고 + 재작성 + 검색` 수준이며, 사용자가 원하는 `메모리 포함 멀티턴 구조`와는 아직 거리가 있다. 따라서 현재 상태로는 멀티턴이 제대로 도는지 간접적으로만 추정할 수 있고, 확실하게 확인하기 어렵다.

## 목표 구조

### 1. 사용자 질문 + 대화 이력

입력은 항상 아래 두 가지를 받는다.

- 현재 질문 `query`
- 이전 대화 `chat_history`

`chat_history`는 검색 단계와 생성 단계 양쪽에서 사용되지만, 역할은 다르다.

- 검색 단계: 독립 검색 쿼리 복원
- 생성 단계: 최근 원문 유지 + 요약/슬롯 메모리 공급

### 2. LLM Query Rewriting

후속 질문이 들어오면 먼저 LLM이 이를 독립 검색 쿼리로 바꾼다.

예:

- 입력 질문: `그거 평가기준 알려줘`
- 대화 이력: `서버 구축 입찰 사업 알려줘`
- 재작성 결과: `서버 구축 입찰 사업의 기술 평가기준`

재작성 단계가 보존해야 하는 정보:

- 발주기관
- 사업명 또는 문서 대상
- 비교 대상
- 질문 의도
  - 예: 평가기준, 예산, 일정, 유지보수, 보안, 하도급

재작성 모드:

- `rule_only`
- `llm_only`
- `llm_with_rule_fallback`

권장 기본값:

- `llm_with_rule_fallback`

폴백 순서:

1. LLM 재작성
2. 규칙 기반 재작성
3. 원본 질문 유지

### 3. Hybrid Retrieval

재작성된 쿼리를 기준으로 Dense 검색과 BM25 Sparse 검색을 함께 수행한다.

구성:

- Dense 검색: 기존 벡터 스토어 사용
- Sparse 검색: `chunks.parquet` 기반 BM25
- 후보 결합: RRF

데이터 소스:

- `data/processed/chunks.parquet`
- 실험별 경로가 있으면 동일 규칙으로 해석

여기서 중요한 점은 dense와 sparse가 반드시 같은 chunk 세계를 공유해야 한다는 것이다. 별도 문서 소스를 도입하지 않는다.

### 4. 선택적 Cross-Encoder Reranking

Cross-Encoder는 유지하되 강제 단계로 두지 않는다.

- `reranker_model: null` -> OFF
- `reranker_model: <model>` -> ON

이 단계는 `질문 + 검색된 청크` 기준으로 재순위화만 담당한다. 메모리 처리는 여기 넣지 않는다.

### 5. Summary Buffer Memory + Slot Memory

이 단계가 이번 설계에서 가장 중요한 신규 요소다.

#### Summary Buffer Memory

역할:

- 최근 몇 턴은 원문 그대로 유지
- 오래된 턴은 요약으로 압축

권장 동작:

- 최근 3~4턴: 원문 유지
- 그 이전 턴: 요약 버퍼로 압축
- 요약은 대화가 길어질수록 누적 갱신

이 메모리는 `긴 대화에서 맥락을 잃지 않는 것`이 목적이지, 검색 쿼리 재작성을 대신하는 것이 아니다.

#### Slot Memory

역할:

- 대화에서 반복적으로 등장하는 핵심 값만 구조화해서 저장

예:

- `발주기관: 교육부`
- `사업명: 서버 구축 입찰 사업`
- `예산: 3억`
- `관심분야: 클라우드`
- `현재 비교 대상: 평택시 BIS vs 안양시 예약시스템`

권장 슬롯:

- 발주기관
- 사업명
- 공고/문서 식별자
- 예산
- 일정
- 평가기준
- 사용자가 직전 턴에서 추적한 속성

메모리 단계의 출력:

- `recent_turns`
- `summary_buffer`
- `slot_memory`

### 6. LLM 응답 생성

최종 응답 생성에는 아래를 함께 전달한다.

- 현재 질문
- 재작성된 쿼리
- 검색 결과 컨텍스트
- 최근 원문 턴
- 요약 메모리
- 슬롯 메모리

즉 생성 단계는 단순히 `검색 결과 + 최근 4턴`만 보는 구조에서 벗어나야 한다.

## 설정 원칙

새 YAML 파일은 만들지 않는다. 기존 `configs/retrieval.yaml`만 확장한다.

유지하는 항목:

- `reranker_model`
- `hybrid.enabled`
- `hybrid.dense_pool_multiplier`
- `hybrid.sparse_pool_multiplier`
- `hybrid.rrf_k`
- `enable_multiturn`

추가가 필요한 항목:

- `rewrite.mode`
- `memory.enabled`
- `memory.summary_buffer.max_recent_turns`
- `memory.summary_buffer.max_summary_tokens`
- `memory.slot_memory.enabled`
- `debug_trace.enabled`

문서상 원칙:

- 기능 토글은 기존 YAML 안에 넣는다.
- 멀티턴 전용 별도 retrieval YAML은 만들지 않는다.

## 진입 경로

1차 구현의 진입 경로는 CLI만 대상으로 한다.

대상:

- `scripts/run_rag.py`

이유:

- 멀티턴 동작 확인이 가장 빠르다.
- 재작성/메모리/비용 trace를 바로 확인할 수 있다.
- Streamlit, 웹 API보다 영향 범위가 작다.

2차에서 연결할 경로:

- Streamlit
- 웹 API

## 디버그 출력과 비용 기록

현재 구조에서는 멀티턴이 실제로 도는지 확인하기 어렵기 때문에, CLI에서 아래 정보를 바로 볼 수 있어야 한다.

필수 디버그 출력:

- `original_query`
- `rewritten_query`
- `rewrite_applied`
- `retrieved_chunks_before_rerank`
- `retrieved_chunks_after_rerank`
- `memory_summary`
- `memory_slots`
- `final_answer`

필수 비용 기록:

- `rewrite_prompt_tokens`
- `rewrite_completion_tokens`
- `rewrite_total_tokens`
- `rewrite_cost_usd`
- `generation_cost_usd`
- `total_cost_usd`

비용 기록 원칙:

- 재작성 비용은 생성 비용과 분리해서 보인다.
- CLI에서 한 번의 질의 결과로 바로 확인 가능해야 한다.
- 나중에 평가나 리포트로 넘길 때도 같은 필드를 재사용한다.

## 구현 단위 제안

### 재작성 계층

- `src/bidmate_rag/retrieval/multiturn.py`

책임:

- LLM 재작성
- 규칙 기반 폴백
- 재작성 trace 생성

### 하이브리드 검색 계층

- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/retrieval/hybrid.py`
- `src/bidmate_rag/retrieval/reranker.py`

책임:

- Dense/Sparse 후보 수집
- RRF 융합
- 선택적 재정렬

### 메모리 계층

- 신규 파일 권장: `src/bidmate_rag/retrieval/memory.py`

책임:

- 최근 턴 유지
- summary buffer 생성/갱신
- slot memory 추출/갱신

### 생성 계층

- `src/bidmate_rag/pipelines/chat.py`
- `src/bidmate_rag/providers/llm/openai_compat.py`

책임:

- 검색 결과 + 메모리 조합
- 최종 프롬프트 구성
- 비용 기록 결합

### CLI 확인 경로

- `scripts/run_rag.py`

책임:

- history 입력 받기
- 재작성 결과 출력
- 메모리 출력
- 비용 출력

## 비범위

이번 문서에서 바로 하지 않는 것:

- 멀티턴 전용 평가셋 재구성
- 대화 단위 judge
- LangSmith / DeepEval / Ragas 직접 연동
- 웹/Streamlit UX 반영

## 검증 기준

1차 검증은 CLI에서 아래가 보이면 된다.

1. 후속 질문이 독립 쿼리로 재작성된다.
2. 검색 결과가 재작성된 쿼리를 기준으로 나온다.
3. 메모리 요약과 슬롯이 함께 출력된다.
4. 최종 답변이 메모리를 반영한다.
5. 재작성 비용과 생성 비용이 분리되어 표시된다.

즉, 이번 목표는 `평가`가 아니라 `멀티턴 구조가 실제로 보이게 만드는 것`이다.
