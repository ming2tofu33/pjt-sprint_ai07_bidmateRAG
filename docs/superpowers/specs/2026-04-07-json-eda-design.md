# JSON EDA 설계 스펙 — 청킹 전략 결정을 위한 탐색적 데이터 분석

**날짜:** 2026-04-07  
**목적:** `data/json/` 100개 파일의 구조와 텍스트 특성을 파악하여 최적의 청킹 전략(파라미터 + 방식)을 데이터 기반으로 결정한다.  
**결과물:** `notebooks/eda_json.ipynb`

---

## 배경

현재 `Chunker`(`src/bidmate_rag/preprocessing/chunker.py`)는 kordoc이 생성한 JSON의 `markdown` 필드만 사용하여 `RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)`로 단순 분할한다. JSON에는 `blocks`(heading/paragraph/table 구조체), `outline` 필드도 포함되어 있으나 활용되지 않는다.

이 EDA는 두 가지 질문에 답한다:
1. 현재 파라미터(500/50)가 실제 데이터에 적합한가?
2. markdown 단순 분할 vs blocks 구조 활용 중 어느 전략이 더 나은 청크를 만드는가?

---

## 데이터 개요

- 위치: `data/json/` (100개 `.json` 파일)
- 원본: HWP 96개, PDF 4개, DOCX 1개를 `npx kordoc`으로 파싱한 결과
- JSON 스키마:
  - `markdown` (str): 문서 전체를 마크다운으로 변환한 flat 텍스트 (50K~150K자)
  - `blocks` (list): `{type, text/table, pageNumber, level?, style?}` 구조체 목록. 타입은 `heading`, `paragraph`, `table`
  - `outline` (list): heading 블록만 추출한 목차
  - `metadata` (dict): `{version, pageCount}`
  - `source_file` (str): 원본 파일명

---

## 노트북 구조

### Section 0: Setup & Data Loading

- 필요한 라이브러리 import (`pandas`, `matplotlib`, `seaborn`, `langchain_text_splitters` 등)
- `data/json/` 전체 로딩
- 파일별 요약 DataFrame 생성: `filename`, `markdown_len`, `block_count`, `heading_count`, `paragraph_count`, `table_count`, `page_count`

---

### Section 1: 데이터 파악 (Data Overview)

**1-1. 파일 레벨 기본 통계**
- 100개 파일 요약 테이블 (markdown 길이 기준 상위 10개 / 하위 10개)
- markdown 길이 분포 히스토그램
- 블록 타입 비율(heading/paragraph/table) 전체 합산 + 파일별 분포 박스플롯

**1-2. 페이지 수 분포**
- `metadata.pageCount` 분포 확인
- HWP 파싱 시 pageCount가 1로 나오는 특성 파악 (청킹 시 페이지 기반 분할 불가 여부 확인)

**1-3. 테이블 분석**
- 테이블 rows × cols 분포 히트맵
- 빈 셀 비율 분포 (표 품질 파악)
- 테이블 내용 샘플 5개 렌더링 (어떤 RFP 정보가 표에 있는지 육안 확인)

---

### Section 2: 전략 A — Markdown 기반 청킹

현재 `Chunker` 방식의 실제 동작을 검증한다.

**2-1. 현재 설정 시뮬레이션 (chunk_size=500, overlap=50)**
- 100개 파일에 `RecursiveCharacterTextSplitter` 적용
- 파일당 청크 수 분포 히스토그램
- 청크 길이(글자수) 분포 히스토그램

**2-2. 파라미터 감도 분석**
- chunk_size `[300, 500, 800]` × overlap `[0, 50, 100]` 조합별 청크 수 / 평균 청크 길이 비교 테이블
- 권장 범위 도출을 위한 시각화

**2-3. 경계 품질 분석**
- 청크가 헤딩 중간에 잘리는 비율 측정
- "나쁜 청크" 예시 3개 출력 (문맥이 끊긴 실제 청크)
- "좋은 청크" 예시 3개 출력 (자연스러운 경계)

**2-4. 테이블 처리 문제**
- markdown 테이블(`|---|---|`)이 청크 경계에서 잘렸을 때의 결과 예시
- 테이블이 청크 경계에 걸리는 빈도 수치화

---

### Section 3: 전략 B — Blocks 기반 청킹

`blocks` 구조를 활용해 의미 경계를 지키는 청킹 방식을 탐색한다.

**3-1. Heading 경계 기반 섹션 분할**
- heading 블록을 기준으로 섹션 묶기 (heading → 다음 heading 사이의 paragraph들을 한 섹션으로)
- 섹션 길이 분포 히스토그램
- 너무 긴 섹션(2000자 이상) 비율, 너무 짧은 섹션(50자 미만) 비율 수치화

**3-2. 긴 섹션 처리 전략 탐색**
- 긴 섹션을 하위 heading(level 2, 3)으로 추가 분할 가능한지 확인
- 추가 분할 전/후 청크 길이 분포 비교

**3-3. 테이블 Atomic 처리**
- 테이블 블록을 독립 청크로 분리했을 때의 결과
- 테이블 청크 길이 분포 (너무 큰 테이블: 행 단위 분할 vs 통째로 유지)
- 테이블 청크 샘플 3개 출력

**3-4. Heading 레벨 분포 재확인**
- heading이 수백 개인 이유 분석
- `fontSize`, `level` 기준으로 의미 있는 섹션 헤딩 vs HWP 스타일 노이즈 heading 구분 가능성 탐색

---

### Section 4: 비교 결론

**4-1. 청크 길이 분포 비교**
- 전략 A vs 전략 B 청크 길이 분포 오버레이
- 평균, 중앙값, 표준편차, 최대/최소 비교 테이블

**4-2. 경계 품질 비교**
- 전략 A: 헤딩/테이블 경계에서 잘리는 청크 비율
- 전략 B: 섹션 경계를 지킨 청크 비율
- 대표 예시 나란히 출력

**4-3. 권장 파라미터 정리**
- 전략 A 권장 `chunk_size` / `overlap` 값
- 전략 B 권장 "긴 섹션 추가 분할 임계값"

**4-4. 전략 선택 가이드**

| 상황 | 권장 전략 |
|---|---|
| 빠른 베이스라인, 단순 구현 | A (markdown) |
| 테이블 정보 검색 품질 중요 | B (blocks) |
| 시나리오 A/B 비교 실험 | 두 전략 모두 실험 |

---

## 제약 및 주의사항

- `data/json/` 파일은 Git에 포함되지 않으므로, 노트북 실행 전 로더를 먼저 실행해야 함
- `pageCount`가 대부분 1이므로 페이지 기반 분할은 전략으로 채택하지 않음
- EDA 결과는 `Chunker` 구현 개선의 직접 입력값으로 사용됨
