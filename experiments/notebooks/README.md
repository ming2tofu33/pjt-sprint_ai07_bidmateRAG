# Experiment Notebooks - 재현 가이드

## 사전 준비

```bash
# 1. 패키지 설치
uv sync --group dev

# 2. API 키 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. kordoc 설치 (HWP 파싱용, Node.js 필요)
npm install -g kordoc

# 4. Jupyter 커널 등록
uv run python -m ipykernel install --user --name bidmate-rag --display-name "BidMate RAG"

# 5. 원본 데이터 배치
# data/raw/rfp/ 에 RFP 문서 101개 (HWP/PDF)
# data/raw/metadata/data_list.csv
```

## 실행 순서

노트북은 **반드시 순서대로** 실행해야 합니다. 각 단계의 출력이 다음 단계의 입력이 됩니다.

```
01_eda.ipynb              → 데이터 탐색 (파일 출력 없음)
02_preprocessing.ipynb    → parsed_documents.parquet
03_cleaning.ipynb         → cleaned_documents.parquet
04_chunking.ipynb         → chunks.parquet
05_embedding.ipynb        → artifacts/chroma_db/ (ChromaDB, ~380MB)
06_retrieval.ipynb        → 검색 테스트 (in-memory)
07_generation.ipynb       → RAG 답변 생성 (in-memory)
08_evaluation.ipynb       → 평가 지표 + 대시보드
```

## 예상 소요

- 전체 실행: 약 30~40분 (05번 임베딩이 가장 오래 걸림)
- API 비용: 임베딩 ~$0.15 + 평가 질문 ~$0.10 = **총 ~$0.25 (약 340원)**

## 의사결정 기록

각 단계의 핵심 설계 결정과 근거는 [docs/decision-log.md](../../docs/decision-log.md)에 정리되어 있습니다.
