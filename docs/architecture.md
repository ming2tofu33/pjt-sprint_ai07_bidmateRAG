# Architecture — BidMate RAG Baseline (시나리오 B)

## 전체 파이프라인

```mermaid
flowchart TB
    subgraph 데이터 준비
        A[RFP 원본 문서\n101개 HWP/PDF] --> B[kordoc 파싱\nNode.js CLI]
        B --> C[마크다운 텍스트\n평균 106,566자/문서]
        C --> D[텍스트 정제\n6종 노이즈 제거]
        D --> E[정제된 마크다운\n평균 99,105자/문서]
    end

    subgraph 인덱싱
        E --> F[2단계 청킹\nh1 분리 + 500자 병합]
        F --> G[14,035개 청크\n평균 1,037자]
        G --> H[메타데이터 보강\n기관유형/도메인/기술스택]
        H --> I[text-embedding-3-small\n1,536차원 벡터]
        I --> J[(ChromaDB\n13,951개 벡터\n+ 메타데이터)]
    end

    subgraph 검색 & 생성
        K[사용자 질문 + 대화 이력] --> L[LLM Query Rewriting\n실패 시 규칙 기반 폴백]
        L --> M[Hybrid Retrieval\nDense + Sparse + RRF]
        J --> M
        M --> N{Cross-Encoder\n사용 여부}
        N -->|ON| O[재정렬]
        N -->|OFF| P[그대로 진행]
        O --> Q[Summary Buffer Memory\n+ Slot Memory]
        P --> Q
        Q --> R[컨텍스트 + 메모리 조합\n출처 태그 포함]
        R --> S[gpt-5-mini\n시스템 프롬프트]
        S --> U[답변 생성\n출처 명시]
    end

    subgraph 멀티턴 대화
        U --> V[답변 출력]
        V --> W{후속 질문?}
        W -->|예| X1[최근 턴 유지\n오래된 턴 요약\n슬롯 갱신]
        X1 --> K
        W -->|아니오| Y[종료]
    end
```

## 데이터 흐름

```mermaid
flowchart LR
    A[data/raw/rfp/\n101개 문서] -->|kordoc| B[parsed_documents\n.parquet]
    B -->|정제| C[cleaned_documents\n.parquet]
    C -->|청킹 + 메타| D[chunks\n.parquet\n14,035개]
    D -->|임베딩| E[(artifacts/\nchroma_db/\n384MB)]
    E -->|검색| F[Top-k 청크]
    F -->|생성| G[RAG 답변]
```

## 주요 컴포넌트 상세

### 1. 파싱 (02_preprocessing)

```mermaid
flowchart LR
    A[HWP 96건] -->|kordoc CLI| B[마크다운]
    C[PDF 4건] -->|kordoc CLI| B
    D[DOCX 1건] -->|kordoc CLI| B
    B --> E[parsed_documents.parquet\n메타데이터 12개 컬럼 + 본문]
```

- **파서**: kordoc (Node.js) — subprocess로 호출
- **출력**: 마크다운 (# 헤딩, 표, 목차 구조 보존)
- **성능**: 파일당 0.9초, CSV 대비 평균 28배 텍스트 추출
- **폴백**: hwp-hwpx-parser (Python) — kordoc 실패 시 교차 검증

### 2. 정제 (03_cleaning)

```mermaid
flowchart LR
    A[파싱된 마크다운] --> B[kordoc Warning 제거]
    B --> C["&lt;br&gt; → \\n"]
    C --> D[중복 셀 축소]
    D --> E[PUA/NBSP 제거]
    E --> F[공백 정규화\n표 행 제외]
    F --> G[정제된 마크다운]
```

| 노이즈 | 제거량 | 제거율 |
|---|---|---|
| `<br>` 태그 | 102,922개 | 100% |
| 연속줄바꿈 | 26,437개 | 100% |
| PUA 문자 | 2,831개 | 100% |
| 중복 셀 행 | 6,583개 | 99.6% |
| 연속공백 | 136,475개 | 27.6% (표 내부 보존) |

### 3. 청킹 (04_chunking)

```mermaid
flowchart TB
    A[정제된 마크다운] --> B[1단계: h1 헤딩 분리]
    B --> C{섹션 크기\n< 500자?}
    C -->|예| D[다음 섹션과 병합]
    C -->|아니오| E{표 블록?}
    D --> C
    E -->|예| F[표 보존\n초과 시 헤더 반복 분할]
    E -->|아니오| G[RecursiveCharacterTextSplitter\n1,000자 / 150자 오버랩]
    F --> H[메타데이터 부착]
    G --> H
    H --> I[chunks.parquet\n14,035개]
```

**메타데이터 스키마**:

| 필드 | 출처 | 용도 |
|---|---|---|
| 사업명, 발주기관, 공고번호 | CSV 원본 | 출처 표시, 필터링 |
| 사업금액 | CSV 원본 | 금액 범위 필터 |
| 기관유형 | 규칙 분류 (v2) | "대학교 사업만" 필터 |
| 사업도메인 | 사업명 + 본문 키워드 | "교육 관련" 필터 |
| 기술스택 | 본문 키워드 | "AI 관련" 필터 |
| 공개연도 | 공개일자 변환 | "2024년" 필터 |
| section | 청킹 시 헤딩 | 섹션 필터 |
| content_type | 표/텍스트 구분 | 표 부스팅 |
| text_with_meta | 프리픽스 + 본문 | 임베딩 입력 |

### 4. 임베딩 + 벡터 DB (05_embedding)

```mermaid
flowchart LR
    A[14,035개 청크] -->|50자 미만 제외| B[13,951개]
    B -->|text_with_meta| C[OpenAI API\ntext-embedding-3-small]
    C --> D[1,536차원 벡터]
    D --> E[(ChromaDB\nPersistentClient\ncosine 유사도)]
    F[12개 메타데이터\n필드] --> E
```

- **임베딩 입력**: `[발주기관: X | 사업명: Y]\n본문...` (프리픽스 ~5%)
- **비용**: ~$0.15 (207원)
- **저장 크기**: 384MB

### 5. 검색 (06_retrieval, 07_generation)

#### Naive Baseline

```mermaid
flowchart LR
    A[질문] --> B{발주기관\n감지?}
    B -->|예| C[where 필터]
    B -->|아니오| D[필터 없음]
    C & D --> E[코사인 유사도\nTop-k]
    E --> F[컨텍스트]
```

#### Enhanced (5가지 고도화)

```mermaid
flowchart TB
    A[질문] --> B[발주기관/도메인/기관유형 필터]
    A --> C[금액/연도 범위 필터]
    A --> D[섹션 키워드 추출]
    A --> E{필터\n있음?}
    E -->|없음| F[사업 요약 100개로\n문서 Top-3 특정]
    E -->|있음| G[필터 적용]
    F & G --> H[벡터 검색\n+ 섹션 필터]
    H --> I{정형 데이터\n질문?}
    I -->|예| J[표 청크 점수 ×1.2]
    I -->|아니오| K[그대로]
    J & K --> L[Top-k 결과]
    L --> M[인접 청크 확장\nchunk_index ±1]
    M --> N[최종 컨텍스트]
```

| 개선 | 효과 (비교 테스트) |
|---|---|
| 사업도메인 필터 | "교육 관련" → 교육/학습 4건 정확 검색 |
| 금액 범위 필터 | "5억 이상" → Naive 1건 → Enhanced **3건** |
| 섹션 필터 | 요구사항 질문에 요구사항 섹션 청크 우선 |
| 표 부스팅 | 정형 데이터 질문에 표 청크 상위 노출 |
| 인접 청크 | 잘린 표/문단의 뒷부분 보완 |

### 5-1. 멀티턴 검색 (LLM Query Rewriting + Hybrid Retrieval)

```mermaid
flowchart TB
    A[후속 질문\n예: 평가기준은?] --> B{chat_history\n존재?}
    B -->|없음| C[원본 쿼리 그대로]
    B -->|있음| D[LLM Query Rewriting\n5초 타임아웃]
    D -->|성공| E[재작성된 쿼리\n교육부 서버 구축 사업의 기술 평가기준]
    D -->|실패| F[규칙 기반 폴백\n또는 원본 쿼리]
    E --> G[Hybrid 검색\nDense + Sparse + RRF]
    F --> G
    C --> G
    G --> H{Cross-Encoder\n사용 여부}
    H -->|ON| I[재정렬]
    H -->|OFF| J[그대로 진행]
    I --> K[메모리 단계로 전달]
    J --> K
```

**3단계 폴백 체인:**
1. **LLM 재작성** → 성공 시 사용
2. **규칙 기반 재작성** → LLM 실패 시 폴백
3. **원본 쿼리 유지** → 규칙도 적용 불가 시 그대로 사용

| 시나리오 | 재작성 방식 | 예시 |
|----------|------------|------|
| 지시어 없는 후속 | LLM 재작성 | "예산은?" → "교육부 서버 구축 사업의 예산" |
| 지시어 있는 후속 | LLM 재작성 (또는 규칙 폴백) | "그 사업 평가기준은?" → "교육부 서버 구축 사업의 평가기준" |
| 첫 번째 질문 | 스킵 (LLM 호출 없음) | "서버 구축 사업 찾아줘" → 그대로 |
| LLM 타임아웃 | 규칙 기반 폴백 | "그 사업 예산은?" → "서버 구축 사업 예산은?" |

### 5-2. 메모리 계층 (Summary Buffer Memory + Slot Memory)

```mermaid
flowchart TB
    A[최근 대화 이력] --> B[최근 3~4턴 원문 유지]
    A --> C[오래된 턴 요약\nSummary Buffer]
    A --> D[핵심 값 추출\nSlot Memory]
    B --> E[메모리 상태]
    C --> E
    D --> E
```

메모리 계층은 검색을 대체하지 않는다. 검색으로 가져온 근거를 `대화 문맥과 다시 연결`하는 역할을 한다.

**Summary Buffer**

- 최근 몇 턴은 원문 그대로 유지
- 오래된 턴은 요약으로 압축
- 긴 대화에서도 토큰 폭증을 줄이면서 문맥을 유지

**Slot Memory**

- 발주기관
- 사업명
- 예산
- 일정
- 평가기준
- 사용자가 현재 추적 중인 관심 속성

예시:

- `발주기관: 교육부`
- `사업명: 서버 구축 입찰 사업`
- `예산: 3억`
- `관심분야: 클라우드`

### 6. 생성 (07_generation)

```mermaid
flowchart LR
    A[시스템 프롬프트] --> B[gpt-5-mini]
    C[최근 원문 턴] --> B
    D[Summary Buffer] --> B
    E[Slot Memory] --> B
    F[컨텍스트\n출처 태그 포함] --> B
    G[재작성된 쿼리] --> B
    H[사용자 질문] --> B
    B --> I[답변\n핵심→표→세부→부재]
```

**생성 단계 입력 원칙**:
- 검색 결과만 넘기지 않는다.
- 재작성된 쿼리를 함께 넘긴다.
- 최근 원문 턴, Summary Buffer, Slot Memory를 함께 넘긴다.
- 답변 비용은 재작성 비용과 분리해서 기록한다.

### 7. 평가 (08_evaluation)

```mermaid
flowchart TB
    A[평가셋 12개 질문\nA/B/C/D/E 5유형] --> B[RAG 파이프라인 실행]
    B --> C[검색 지표\nHit Rate / MRR / nDCG]
    B --> D[LLM Judge\nFaithfulness / Relevance\nContext Precision / Recall]
    B --> E[성능 지표\nLatency / Token Cost]
    C & D & E --> F[종합 대시보드]
```

**Baseline 평가 결과**:

| 지표 | 점수 |
|---|---|
| Hit Rate@5 (필터) | 1.00 |
| Hit Rate@5 (벡터만) | 0.78 |
| MRR (필터) | 1.00 |
| nDCG@5 (필터) | 1.00 |
| Faithfulness | 71점 |
| Relevance | 100점 |
| Context Precision | 88점 |
| Context Recall | 79점 |
| 무응답 정확도 | 100% |
| 평균 응답 시간 | 25.1초 |

## 기술 스택

| 구분 | 기술 | 용도 |
|---|---|---|
| 파싱 | kordoc (Node.js) | HWP/PDF → 마크다운 |
| 청킹 | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | 2단계 하이브리드 |
| 임베딩 | text-embedding-3-small (OpenAI) | 1,536차원 벡터 |
| 벡터 DB | ChromaDB (PersistentClient) | 코사인 유사도 + 메타 필터 |
| LLM | gpt-5-mini (OpenAI) | 답변 생성 + LLM Judge |
| 언어 | Python 3.12 + uv | 패키지 관리 |
| 실험 | Jupyter Notebook | 01~08 파이프라인 |

## 비용

| 항목 | 비용 |
|---|---|
| 전체 임베딩 (13,951개) | ~$0.15 (207원) |
| 질문 1건 — 싱글턴 (생성만) | ~$0.001 (1원) |
| 질문 1건 — 멀티턴 (재작성 + 메모리 + 생성) | 약 $0.0012~$0.0018 |
| 평가 12건 (생성+Judge) | ~$0.10 (135원) |
| **총 1회 실행** | **~$0.25 (340원)** |

> 멀티턴 비용 기록은 질의 단위로 아래를 분리해 남기는 것을 목표로 한다.
> `rewrite_prompt_tokens`, `rewrite_completion_tokens`, `rewrite_total_tokens`, `rewrite_cost_usd`, `generation_cost_usd`, `total_cost_usd`
> 현재 목표는 이 정보가 최소한 CLI에서 직접 보이게 만드는 것이다.
