# 멀티턴 메모리형 하이브리드 검색 계획

1. 기존 `configs/retrieval.yaml`만 기준으로 사용하고, 멀티턴 전용 YAML 파일은 새로 만들지 않는다.
2. 목표 파이프라인은 아래 순서를 따른다.
   - 사용자 질문 + 대화 이력
   - LLM Query Rewriting
   - Hybrid Retrieval(Dense + Sparse)
   - 선택적 Cross-Encoder Reranking
   - Summary Buffer Memory + Slot Memory
   - LLM 응답 생성
3. 쿼리 재작성 단계는 후속 질문을 독립 검색 쿼리로 바꾸되, 실패 시 기존 규칙 기반 재작성으로 안전하게 폴백한다.
4. 하이브리드 검색은 기존 `chunks.parquet`를 공통 소스로 사용하며, dense 검색과 BM25 sparse 검색을 RRF로 융합한다.
5. Cross-Encoder 재정렬은 유지하되 필수 단계로 강제하지 않는다.
   - `reranker_model: null`이면 OFF
   - `reranker_model: <model>`이면 ON
6. 메모리 단계는 검색 이후에 배치한다.
   - 최근 턴 원문은 그대로 유지한다.
   - 오래된 턴은 summary buffer로 요약한다.
   - 핵심 슬롯은 구조화된 값으로 유지한다.
7. 슬롯 메모리는 최소 아래 항목을 다룬다.
   - 발주기관
   - 사업명
   - 예산
   - 일정
   - 평가기준
   - 사용자가 반복적으로 추적하는 관심 속성
8. 최종 응답 생성에는 아래 3가지를 함께 넣는다.
   - 재작성된 쿼리
   - 검색 결과
   - 메모리 요약 및 슬롯
9. CLI에서 멀티턴이 실제로 동작하는지 확인할 수 있도록 디버그 출력과 비용 기록을 남긴다.
   - `original_query`
   - `rewritten_query`
   - `retrieved_chunks_before_rerank`
   - `retrieved_chunks_after_rerank`
   - `memory_summary`
   - `memory_slots`
   - `rewrite_cost_usd`
   - `generation_cost_usd`
   - `total_cost_usd`
10. 1차 확인 경로는 CLI로 제한한다.
    - `run_rag.py`에서 history 입력을 받고
    - 재작성 결과와 메모리 상태를 출력할 수 있게 한다.
11. Streamlit과 웹 경로는 CLI에서 동작이 확인된 뒤 같은 코어 로직을 연결한다.
12. 멀티턴 평가는 나중 단계로 미루되, 지금 구조에서 바로 이어서 평가할 수 있도록 trace와 cost 기록은 미리 남긴다.
