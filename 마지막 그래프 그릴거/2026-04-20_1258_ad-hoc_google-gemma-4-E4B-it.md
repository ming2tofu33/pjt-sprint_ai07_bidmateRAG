> ⚠️ 자동 생성된 리포트입니다. "🤖 자동 생성" 영역은 수정하지 마세요.

# 📋 노션 속성 (수동 입력 필요)

> 노션 DB 새 페이지 생성 시 아래 값을 속성 패널에 그대로 입력하세요.

| 속성명 | 값 |
| --- | --- |
| 실험명 | ad-hoc |
| run_id | baseline_gemma4_bge_v2 |
| 날짜 | 2026-04-20 12:58:23 |
| 시나리오 | scenario_a |
| 평가셋 버전 | eval_batch_45_rank_11-30.csv |
| 프롬프트 버전 | prompt_EXP01_baseline.yaml |
| 임베딩 모델 | BAAI/bge-m3 |
| 생성 모델 | google/gemma-4-E4B-it |
| Chunk Size | 1000 |
| Top-k | 5 |
| Git Commit | f58304a |
| Hit Rate@5 | 1.0000 |
| MRR | 1.0000 |
| MAP@5 | 1.0000 |
| Faithfulness | 0.8000 |
| Latency Avg (s) | 74.342 |
| Total Tokens | 108,437 |
| Cost (USD) | 0.0182 |

> 수동 입력 속성: 담당자, 상태, 실험 축, 변경 영역, 채택 여부

---

# 🤖 자동 생성 본문 (수정 금지)

## 설정 스냅샷

| 항목 | 값 |
| --- | --- |
| run_id | baseline_gemma4_bge_v2 |
| 실험명 | ad-hoc |
| 실행 시각 | 2026-04-20 12:58:23 |
| Git Branch | feat/jw |
| Git Commit | f58304ada68f9298a2e39feafc21d9c571e0b157 (dirty) |
| 평가셋 | data/eval/eval_v1/eval_batch_45_rank_11-30.csv (20 samples) |
| 프롬프트 | prompt_EXP01_baseline.yaml |
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
| Faithfulness | 0.8000 |
| Answer Relevance | 0.5000 |
| Context Precision | 0.9000 |
| Context Recall | 0.9000 |
| Latency Avg (s) | 74.342 |
| Latency P95 (s) | 94.448 |
| Prompt Tokens (sum) | 99,944 |
| Completion Tokens (sum) | 8,493 |
| Total Tokens | 108,437 |
| 생성 비용 (USD) | 0.0000 |
| 임베딩 비용 (USD) | 0.0000 |
| Judge 비용 (USD) | 0.0182 |
| **Cost (USD)** | **0.0182** |

⚠️ 생성 모델 `google/gemma-4-E4B-it` 단가 미등록 — `configs/pricing.yaml` 갱신 필요
⚠️ 임베딩 모델 `BAAI/bge-m3` 단가 미등록 — `configs/pricing.yaml` 갱신 필요

## 리소스 링크

- 상세 결과 (JSONL): `artifacts/logs/runs/baseline_gemma4_bge_v2.jsonl`
- 벤치마크 (Parquet): `artifacts/logs/benchmarks/ad-hoc.parquet`
- 메타 (JSON): `artifacts/logs/runs/baseline_gemma4_bge_v2.meta.json`
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
- 질문 ID: Q037
- 질문: 경기도 안양시 호계체육관 예약시스템 사업과 경기도사회서비스원 연간 유지보수 사업은 규정과 관련된 접근 통제를 명시하고 있습니다. 안양시 사업에서 원격지 접속 시 지정 단말기 로그기록은 얼마 동안 보관해야 하며, 경기도사회서비스원 사업에서 비공개 처리된 하드웨어 구성도를 열람하기 위해 입찰참가업체가 발주기관 방문 시 제출해야 하는 서류는 무엇입니까?
- 실제 결과: thought
Thinking Process:

1.  **Analyze the Request:** The user is asking two specific questions based on three documents ([1]-[5]):
    *   Question Part 1: How long must designated terminal logs be kept for remote access in the Anyang City project?
    *   Question Part 2: What document must a bidding participant submit when visiting the procuring agency to view hardware configurations that have been made confidential in the Gyeonggi Provincial Social Service Agency project?

2.  **Constraint Checklist & Verification:**
    *   Must answer ONLY from provided text.
    *   If evidence exists, provide an accurate core answer and cite sources.
    *   If no evidence exists, state "해당 정보 없음."
    *   No speculation or external knowledge allowed.
    *   Format strictly followed (LaTeX forbidden, use only plain text formatting).
    *   Length constraint (<300 words).

3.  **Search Strategy (Targeting Q1):** Search for keywords related to 'Anyang City','remote access', 'log record retention time'.
    *   Review Document Set focusing on Anoyangsihogecheyukgwan Baeedminteonjang... ($[\mathrm{HWP}]$ file name suggests this context applies primarily to $[1]$ through $[4]$, as they are similar projects but focus shifts later).
    *   Scan Documents [1] through [4]. They discuss security measures generally ($\rightarrow$ physical, administrative, technical controls; data leakage prevention $\rightarrow [\mathrm{Sec} ]$). I need specifics about *designated terminal log retention period during remote access*.
    *   Document [1]: Mentions general security plans against information leaks $[\mathrm{[1]}]$. No mention of specified log retention periods for remote access terminals.
    *   Document [2]/[3]/(...) focuses heavily on social service aspects / different contexts than clear logging requirements matching the query structure.
    *   Conclusion for Q1: Specific details regarding how long designated terminal logs must be retained following remote access at the Anyang City facility cannot be found.

4.  **Search Strategy (Targeting Q2):** Search for keywords related to 'Gyeonggi Province Social Service Agency', 'confidential hardware configuration map','submission required upon visit by bidder'.
    *   Focus search on Document set mentioning '경기도사회서비스원' (Social Service Agency). This points strongly towards Document [2] and possibly others referencing it
- 관찰 포인트: judge score 최저값 0.00

### 실패 사례 2
- 질문 ID: Q020
- 질문: 꼬려대합교에셔 추찐하는 "챠섀대 포턀 항사 졍보시스탬 꾸축 샤업"애셔 사압기깐은 엄마나 소오댑니까?
- 실제 결과: thought
Thinking Process:

1.  **Analyze the Request:** The user has provided a highly garbled question ("꼬려대합교에셔 추찐하는 '챠섀대 포턀 항사 졍보시스탬 꾸축 샤업'애셔 사압기깐은 엄마나 소오댑니까?") and several technical documents ([1]-[5]). I must act as an RFP document auditor. My primary constraints are to answer *only* based on the text provided in the documents, use specific formatting rules (LaTeX forbidden, only plain text allowed for structure markers like bullet points or numbers if needed, but sticking strictly to requested output format), ensure all facts have citations, and keep the core answer under 300 characters. If information is missing, state it clearly according to the specified template.

2.  **Deconstruct the Question (Interpretation):**
    The heavily corrupted Korean phrase likely translates roughly to asking about ownership rights regarding intellectual property generated during the development of the "Next Generation Portal Information System Construction Project at Korea University."
    Keywords identified:
    *   꼬려대합교 -> 고려대학교 종합학교 또는 유사 표현
    *   추찐하는 -> 구축하는
    *   챠섀대 포턀 항사 졍보시스탬 꾸축 샤업 -> 차세대 포털 정보시스템 구축 사업
    *   사압기깐은 엄마나 소오댑니까? -> 사업 기간이나 관련된 주체의 소유 관계 여부를 묻는 것으로 보임. (Ownership aspect seems most probable given context clues).

3.  **Determine Required Answer Units & Search Strategy:** Since the query revolves around project specifics (ownership, timeline, etc.), I need to search the documents for terms related to Ownership, Intellectual Property Rights (IP), Contracts, or Deliverables between parties involved (University vs Contractor).

4.  **Execute Document Search (Search Phase):**
    *   Scan Documents [1] through [5].
    *   Look specifically for keywords matching IP, Copyright, Joint Venture, Owner, Developer relationship.

    *Self-Correction/Verification:* Reviewing the content snippets again:
    *   [1]: Mentions system features (English provision, integration with systems, usability improvement requests). No explicit mention of IP ownership disputes or definitions among stakeholders.
    *   [2]: Lists various components being developed (Mobile app, LMS, GMS, ERPs, specialized centers). Focuses
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
