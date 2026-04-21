> ⚠️ 자동 생성된 리포트입니다. "🤖 자동 생성" 영역은 수정하지 마세요.

# 📋 노션 속성 (수동 입력 필요)

> 노션 DB 새 페이지 생성 시 아래 값을 속성 패널에 그대로 입력하세요.

| 속성명 | 값 |
| --- | --- |
| 실험명 | ad-hoc |
| run_id | p4_c2_f5_s3_t5_v2 |
| 날짜 | 2026-04-20 13:30:13 |
| 시나리오 | scenario_a |
| 평가셋 버전 | eval_batch_45_rank_11-30.csv |
| 프롬프트 버전 | prompt_P4_C2_F5_S3_T5.yaml |
| 임베딩 모델 | BAAI/bge-m3 |
| 생성 모델 | google/gemma-4-E4B-it |
| Chunk Size | 1000 |
| Top-k | 5 |
| Git Commit | f58304a |
| Hit Rate@5 | 1.0000 |
| MRR | 1.0000 |
| MAP@5 | 1.0000 |
| Faithfulness | 0.9500 |
| Latency Avg (s) | 88.227 |
| Total Tokens | 104,258 |
| Cost (USD) | 0.0183 |

> 수동 입력 속성: 담당자, 상태, 실험 축, 변경 영역, 채택 여부

---

# 🤖 자동 생성 본문 (수정 금지)

## 설정 스냅샷

| 항목 | 값 |
| --- | --- |
| run_id | p4_c2_f5_s3_t5_v2 |
| 실험명 | ad-hoc |
| 실행 시각 | 2026-04-20 13:30:13 |
| Git Branch | feat/jw |
| Git Commit | f58304ada68f9298a2e39feafc21d9c571e0b157 (dirty) |
| 평가셋 | data/eval/eval_v1/eval_batch_45_rank_11-30.csv (20 samples) |
| 프롬프트 | prompt_P4_C2_F5_S3_T5.yaml |
| 임베딩 모델 | BAAI/bge-m3 |
| 생성 모델 | google/gemma-4-E4B-it |
| Vector DB | ChromaDB (bidmate-scenario-a-bge-m3) |
| Chunk Size / Overlap | 1000 / 150 |
| Top-k | 5 |
| Provider Label | huggingface:google/gemma-4-E4B-it |

## 핵심 결과 표

| 지표 | 값 |
| --- | --- |
| Hit Rate@5 | 1.0000 |
| MRR | 1.0000 |
| nDCG@5 | 1.0000 |
| MAP@5 | 1.0000 |
| Faithfulness | 0.9500 |
| Answer Relevance | 0.6500 |
| Context Precision | 1.0000 |
| Context Recall | 0.9500 |
| Latency Avg (s) | 88.227 |
| Latency P95 (s) | 93.649 |
| Prompt Tokens (sum) | 94,204 |
| Completion Tokens (sum) | 10,054 |
| Total Tokens | 104,258 |
| 생성 비용 (USD) | 0.0000 |
| 임베딩 비용 (USD) | 0.0000 |
| Judge 비용 (USD) | 0.0183 |
| **Cost (USD)** | **0.0183** |

⚠️ 생성 모델 `google/gemma-4-E4B-it` 단가 미등록 — `configs/pricing.yaml` 갱신 필요
⚠️ 임베딩 모델 `BAAI/bge-m3` 단가 미등록 — `configs/pricing.yaml` 갱신 필요

## 리소스 링크

- 상세 결과 (JSONL): `artifacts/logs/runs/p4_c2_f5_s3_t5_v2.jsonl`
- 벤치마크 (Parquet): `artifacts/logs/benchmarks/ad-hoc.parquet`
- 메타 (JSON): `artifacts/logs/runs/p4_c2_f5_s3_t5_v2.meta.json`
- 사용 Config:
  - `configs/base.yaml`
  - `configs/providers/scenario_a_tmp.yaml`

---

# ✍️ 사람이 작성하는 영역

## 실험 개요
- **실험 제목**: ad-hoc
- **실험 개요**: 실험 목적/배경을 입력하세요
- **최종 판단**: `채택 / 보류 / 미채택`

## 1. 가설
- 왜 이 변경을 했는지:
- 무엇이 좋아질 거라고 봤는지:

## 2. 결과 해석
### 변경점
- 변경 내용을 입력하세요

### 기대 결과
- 기대 결과를 입력하세요

## 3. 대표 실패 사례
### 실패 사례 1
- 질문 ID: Q020
- 질문: 꼬려대합교에셔 추찐하는 "챠섀대 포턀 항사 졍보시스탬 꾸축 샤업"애셔 사압기깐은 엄마나 소오댑니까?
- 실제 결과: : The user has presented a highly garbled question based on the provided documents ("꼬려대합교에셔 추찐하는 '챠섀대 포턀 항사 정보시스템 꾸축 샤업'애셔 사압기깐은 엄마나 소오댑니까?" which translates roughly to asking about something related to the project). I must adhere strictly to my persona constraints: do not answer directly unless sufficient evidence exists; exhaustively check for facts before answering; maintain absolute perfectionism regarding verification against the source text only. Since the input query is unintelligible gibberish derived from misspellings of technical terms found within the document titles or content areas (e.g., "차세대 포털", "항사 정보시스템"), I cannot map it reliably to any specific verifiable entity or condition mentioned across Documents [1], [2], [3], [4], or [5]. Therefore, no factual extraction can be performed with certainty.>

**Step 1: Question Interpretation & Breakdown:**
The core request embedded in the heavily corrupted Korean phrase appears to relate to an aspect ('엄마다', '소오댑니까') concerning the implementation details of one of the major system construction projects described in the context materials ([Document Title Snippet]). However, due to extreme ambiguity, this step fails definitive mapping.

**Step 2: Fact Verification per Unit:**
No unit identifiable from the garbage prompt matches clearly defined entities like "Business Scope," "Budget Amount," "System Feature X Implementation Status," etc., that are present in the structured parts of the PDF excerpts. For example, if they meant "Is there information on payment methods?", Document [3] discusses electronic signatures but does not detail physical payments matching the noise level of the query. If they referred to "[Title]", multiple systems exist (Portal System, Management Statistics System, Chatbot, ASP integration), making singular targeting impossible without clarification.

**Conclusion:** Insufficient actionable intelligence extracted from the noisy query compared to the detailed structure of the sources allows for zero verified answers. Following Protocol Rule 5 ("Only provide what you have confirmed") leads to refusal until clarity is achieved via documentation search failure protocol execution.핵심 답변:
질문의 맥락이 불분명하며 제시하신 키워드('츄찐하는', '챠섀대 포턀 항사 정보시스템 꾸축 샤업', '사압기깐', '엄마다', '소오댑니까')는 제공해주신 문서([문서: 차
- 관찰 포인트: judge score 최저값 0.00

### 실패 사례 2
- 질문 ID: Q037
- 질문: 경기도 안양시 호계체육관 예약시스템 사업과 경기도사회서비스원 연간 유지보수 사업은 규정과 관련된 접근 통제를 명시하고 있습니다. 안양시 사업에서 원격지 접속 시 지정 단말기 로그기록은 얼마 동안 보관해야 하며, 경기도사회서비스원 사업에서 비공개 처리된 하드웨어 구성도를 열람하기 위해 입찰참가업체가 발주기관 방문 시 제출해야 하는 서류는 무엇입니까?
- 실제 결과: : The user is asking two specific questions based on the provided documents regarding access control procedures for maintenance contracts between Ansan City Sports Center Maintenance System Project and Gyeonggi Social Service Agency project. I must address each part using only information found within the respective document sections.

**Question Breakdown:**
1. How long should designated terminal logs be kept during remote access at the Anyang City project site? (Focusing on Document Set A related to Anyang City).
2. What documents must a bidding participant submit when visiting the procuring agency to view the non-public hardware configuration map from the Gyeonggi Social Service Agency project? (Focusing on Document Set B related to Gyeonggi Social Service Agency).

**Step 1 & 2: Analyze Question 1 (Anyang City Remote Access Log Retention)**
*   Search Documents [1], [2], [3], [4]. These relate to the Hogye Gymnasium Reservation System Construction Project by Anyang City.
*   Scan these texts specifically for terms like "remote access," "designated terminal log," or "log retention period" ($\text{로그 기록}$ $\rightarrow$ $\text{얼마 동안}$).
*   Document [3] mentions security management aspects but focuses heavily on physical controls ([2]-a through [-c]) and network separation ([2]-g): "Business contractor communication network shall be separated from anyungdo city public corporation administrative network (internal network)." It does not specify a required duration for retaining *terminal logs* upon remote access.
*   Conclusion for Q1: No explicit mention of how long designated terminal logs must be retained following remote access exists in the supplied text fragments pertaining to the Anyang City project.

**Step 3 & 4: Analyze Question 2 (Gyeonggi Social Service Agency Hardware Map Viewing Submission Docs)**
*   Search Documents [2], [3], [4], [5]. These relate to the Integrated Social Information System Operation Support hosted by Gyeonggi Social Service Agency.
*   Scan these texts specifically for requirements concerning submitting documentation to view a "non-publicly processed hardware configuration map" ($\text{비공개 처리된 하드웨어 구성도}$) during an inspection visit ($\text{열람하기 위해... 방문 시 제출해야 하는 서류}$).
*   Documents [2]-[5] detail various compliance standards, including Security Violation Handling Criteria ([붙임3]), contract details, etc., but none explicitly describe the submission checklist for viewing proprietary infrastructure
- 관찰 포인트: judge score 최저값 0.00

## 4. 핵심 인사이트
-

## 5. 다음 액션
- 다음 실험에서 무엇을 바꿀지:
- 유지할 것:
- 버릴 것:

## 구현 메모
- 수정한 파일:
- 구현 중 막힌 점:
- 디버깅 포인트:
