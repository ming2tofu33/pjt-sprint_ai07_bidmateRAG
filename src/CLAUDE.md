# src/ — 파이프라인 코드 규칙

## 구조

```
bidmate_rag/
├── loaders/          # 문서 파싱 (kordoc → pdfplumber 폴백)
├── preprocessing/    # cleaner (6종 노이즈 정제) + chunker (ChunkingConfig)
├── providers/        # 임베딩 (OpenAI, HF) + LLM (OpenAI, HF)
├── retrieval/        # ChromaVectorStore + RAGRetriever + filters
├── evaluation/       # BenchmarkRunner + metrics (Hit Rate, MRR, nDCG)
├── pipelines/        # ingest, build_index, chat, runtime
├── config/           # settings (YAML 로딩) + prompts (시스템 프롬프트)
├── storage/          # MetadataStore (parquet 기반)
├── training/         # PEFT 준비 (SFT 포맷)
└── schema.py         # 공통 모델: Document, Chunk, RetrievedChunk, GenerationResult
```

## 핵심 규칙

### 메타데이터 키 — 공백 주의
ChromaDB에 저장된 키는 원본 CSV 컬럼명을 따름:
- `발주 기관` (O) / `발주기관` (X)
- `사업 금액` (O) / `사업금액` (X)

filters.py에서 where 필터를 생성할 때 반드시 공백 포함 키를 사용.

### gpt-5 계열 특성
- `temperature` 미지원 → 파라미터 넣지 말 것
- `max_tokens` 미지원 → `max_completion_tokens` 사용
- reasoning 토큰이 completion에 포함됨 → 최소 16000 이상 설정
- 빈 응답이 나오면 토큰 한도 부족 가능성

### 배치 처리 필수
- 임베딩 API: 100개씩 배치 (300K 토큰 한도)
- ChromaDB upsert: 5000개씩 배치 (5461 한도)

### 청킹 전략
- `ChunkingConfig`로 중앙 관리 (preset: small/medium/large)
- kordoc은 세부 항목에도 `#` 헤딩을 붙이므로 h1만 분리 + 500자 병합
- 표는 통째로 보존, 초과 시 헤더 반복 분할

### 공통 인터페이스
- LLM: `BaseLLMProvider.generate(question, context_chunks, history, generation_config, system_prompt) → GenerationResult`
- Embedding: `BaseEmbeddingProvider.embed_documents(texts) → list[list[float]]`
- Retriever: `RAGRetriever.retrieve(query, chat_history, top_k) → list[RetrievedChunk]`

## 테스트
```bash
uv run pytest tests/
```
