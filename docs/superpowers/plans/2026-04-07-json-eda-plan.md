# 구현 계획 — JSON EDA 노트북 (`notebooks/eda_json.ipynb`)

**스펙:** `docs/superpowers/specs/2026-04-07-json-eda-design.md`  
**결과물:** `notebooks/eda_json.ipynb`

---

## 사전 확인

- [ ] `uv sync --group dev` 실행 (jupyter, matplotlib, seaborn 포함 여부 확인)
- [ ] `data/json/` 에 JSON 파일 100개 존재 확인

필요 패키지가 없으면 `pyproject.toml`에 추가:
```toml
[dependency-groups]
dev = [
    ...
    "jupyter>=1.1.1",
    "matplotlib>=3.10.1",
    "seaborn>=0.13.2",
]
```

---

## Step 1 — Section 0: Setup & Data Loading

**파일:** `notebooks/eda_json.ipynb` (신규 생성)

```python
# Cell 1: imports
import json, os, glob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cell 2: 100개 JSON 로딩 → summary DataFrame 생성
# 컬럼: filename, markdown_len, block_count, heading_count,
#        paragraph_count, table_count, page_count
```

완료 기준: `df.shape == (100, 7)` 확인 셀 출력

---

## Step 2 — Section 1: 데이터 파악

**1-1. 파일 레벨 기본 통계**
- `df.describe()` 출력
- markdown_len 기준 상위 10 / 하위 10 테이블
- markdown_len 히스토그램 (bins=20)
- 블록 타입 비율 박스플롯 (heading/paragraph/table 각각)

**1-2. 페이지 수 분포**
- `page_count` value_counts() 바 차트
- 대부분 1인지 확인 → 주석으로 결론 기록

**1-3. 테이블 분석**
- 전체 테이블 블록 추출 → rows, cols, 빈셀비율 계산
- rows × cols 히트맵
- 빈셀 비율 히스토그램
- 테이블 내용 샘플 5개: `display(pd.DataFrame(table['cells']))` 형태로 렌더링

완료 기준: 3개 서브섹션 모두 차트 출력, 테이블 샘플 5개 보임

---

## Step 3 — Section 2: 전략 A (Markdown 청킹)

**2-1. 현재 설정 시뮬레이션**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# 100개 파일 전체 청킹 → chunks_a 리스트
```
- 파일당 청크 수 히스토그램
- 청크 길이 히스토그램

**2-2. 파라미터 감도 분석**
- 3×3 grid (chunk_size × overlap) 결과 DataFrame
- 히트맵으로 시각화 (평균 청크 수 기준)

**2-3. 경계 품질 분석**
```python
# 헤딩 텍스트가 청크 경계에 걸리는지 확인
# 판정 기준: 청크 시작/끝 50자 안에 "# " 패턴 등장 여부
```
- 나쁜 청크 3개, 좋은 청크 3개 출력

**2-4. 테이블 경계 문제**
- `|---` 패턴이 청크 중간에 등장하는 빈도 수치화
- 잘린 테이블 예시 출력

완료 기준: 4개 서브섹션 차트/출력 완성

---

## Step 4 — Section 3: 전략 B (Blocks 청킹)

**3-1. Heading 경계 기반 섹션 분할**
```python
def split_by_headings(blocks):
    # heading 등장 시 새 섹션 시작
    # 섹션 = heading 텍스트 + 이후 paragraph 텍스트 합산
    ...
```
- 섹션 길이 히스토그램
- 2000자 이상 / 50자 미만 비율 출력

**3-2. 긴 섹션 처리**
- 하위 heading(level ≥ 2)으로 재분할 후 길이 분포 비교 (before/after 오버레이)

**3-3. 테이블 Atomic 처리**
- 테이블 블록 → 독립 청크로 분리
- 청크 길이 분포 + 샘플 3개 출력

**3-4. Heading 레벨 분포 재확인**
- heading level 값 분포 바 차트
- fontSize 분포 히스토그램
- fontSize 기준(예: 120pt 이상)으로 필터링 시 heading 수 변화 확인

완료 기준: 4개 서브섹션 완성

---

## Step 5 — Section 4: 비교 결론

**4-1. 청크 길이 분포 오버레이**
```python
plt.hist(chunks_a_lengths, alpha=0.5, label='전략 A')
plt.hist(chunks_b_lengths, alpha=0.5, label='전략 B')
```

**4-2. 경계 품질 수치 비교 테이블**
```python
pd.DataFrame({
    '전략': ['A (markdown)', 'B (blocks)'],
    '헤딩 경계 위반율': [...],
    '테이블 경계 위반율': [...],
})
```

**4-3. 권장 파라미터 마크다운 셀로 정리**
- 전략 A 권장 chunk_size / overlap
- 전략 B 긴 섹션 분할 임계값

**4-4. 전략 선택 가이드 테이블 출력**

완료 기준: 모든 비교 차트 및 결론 셀 완성, 노트북 처음부터 끝까지 Restart & Run All 성공

---

## 완료 체크리스트

- [ ] `notebooks/eda_json.ipynb` 생성
- [ ] Section 0~4 전체 셀 실행 오류 없음
- [ ] Restart & Run All 성공
- [ ] 권장 파라미터 및 전략 결론이 Section 4에 명확히 기록됨
- [ ] 노트북 커밋
