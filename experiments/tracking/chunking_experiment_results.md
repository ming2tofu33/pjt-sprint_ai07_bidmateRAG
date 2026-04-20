# 청킹 실험 결과 분석

## 실험 개요

| 항목 | 내용 |
|------|------|
| 실험 일자 | 2026-04-10 |
| 대상 문서 | RFP 문서 99개 (케빈랩 파일 제외) |
| 임베딩 모델 | text-embedding-3-small (OpenAI) |
| 비교 대상 | Recursive (베이스라인) vs Semantic Percentile |

---

## 실험 설정

### Recursive (베이스라인)
```yaml
chunk_size: 1000
chunk_overlap: 150
min_section_size: 500
max_table_size: 1500
chunking_strategy: recursive
separators: ["\n\n", "\n", ". ", " ", ""]
headers_to_split: [("#", "h1")]
```

### Semantic Percentile
```yaml
chunk_size: 1000
chunk_overlap: 0
min_section_size: 500
max_table_size: 1500
chunking_strategy: semantic
breakpoint_threshold_type: percentile
breakpoint_threshold_amount: 95
headers_to_split: [("#", "h1")]
embedding_model: text-embedding-3-small
```

---

## 결과 비교

| 항목 | Recursive (베이스라인) | Semantic Percentile | 차이 |
|------|----------------------|---------------------|------|
| 총 청크 수 | 6,423개 | 6,296개 | -127개 |
| 평균 글자수 | 1,303자 | 1,309자 | +6자 |
| 최소 글자수 | 50자 | 40자 | -10자 |
| 최대 글자수 | 4,380자 | 4,380자 | 동일 |
| 텍스트 청크 | 1,136개 | 1,009개 | -127개 |
| 테이블 청크 | 5,287개 | 5,287개 | 동일 |

---

## 인사이트

### 1. 청크 수 감소 (-127개)
텍스트 청크만 127개 줄었어요. Semantic이 의미 단위로 더 자연스럽게 묶어서 불필요하게 잘리는 청크가 줄어든 결과예요.

### 2. 테이블 청크 동일 (5,287개)
테이블은 Semantic 적용 대상이 아니라 기존 방식 그대로 처리돼요. 하이브리드 구조가 의도대로 동작하는 것을 확인했어요.

### 3. 평균 글자수 거의 동일 (1,303 vs 1,309자)
청크 수는 줄었지만 평균 크기는 비슷해요. 정보 손실 없이 더 자연스럽게 잘린 거예요.

### 4. 데이터셋 특성
전체 청크 중 테이블 비율이 매우 높아요:
```
테이블: 5,287 / 6,423 = 82%
텍스트: 1,136 / 6,423 = 18%
```
RFP 문서 특성상 표가 많아서 Semantic 효과가 텍스트 청크에만 제한적으로 적용됐어요.

---

## 결론

Semantic 청킹이 정상 동작하며, 텍스트 위주 문서에서 더 큰 효과를 기대할 수 있어요.

현재 데이터셋은 테이블 비율(82%)이 높아서 차이가 크지 않지만, 의미 단위로 더 자연스럽게 분할되는 것은 확인됐어요.

---

## 향후 실험 계획

| 실험 | 설정 | 상태 |
|------|------|------|
| Recursive (베이스라인) | chunk_size=1000, overlap=150 | ✅ 완료 |
| Semantic Percentile | threshold=95 | ✅ 완료 |
| Semantic Standard Deviation | threshold=3 | ⬜ 예정 |
| Semantic Interquartile | threshold=1.5 | ⬜ 예정 |

---

## 참고

- 실험 코드: `src/bidmate_rag/preprocessing/chunker.py`
- 실험 설정: `configs/chunking/`
- 결과 저장: `data/processed/semantic-percentile/`
