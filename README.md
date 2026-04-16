# BidMate RAG

공공입찰 RFP(제안요청서) 100건을 대상으로 한 RAG 질의응답 시스템.
입찰메이트 컨설팅 사내 도구로, 컨설턴트가 RFP 핵심 정보를 빠르게 파악하도록 지원합니다.

## 한 줄 요약

> **YAML 한 장으로 실험을 정의하고, CLI 한 줄로 실행하면 자동으로 비용/메트릭/메타데이터가 기록되며, 노션 친화 마크다운 리포트가 생성됩니다.** 평가셋 무결성, chunking 격리, stale chunk 방지 등 침묵 실패를 차단하는 검증이 곳곳에 들어 있습니다.

## 핵심 기능

- **공통 파이프라인**: 파싱 → 정제 → 청킹 → 인덱싱 → 검색 → 생성 → 평가 → 리포트
- **자동 트래킹**: 비용(생성/임베딩/judge 분리), 토큰, latency, git, config 스냅샷, 노션 친화 마크다운
- **평가 메트릭**: Hit Rate@5 / MRR / nDCG@5 (검색) + Faithfulness / Answer Relevance / Context Precision / Recall (LLM judge)
- **Grid search**: yaml 한 장에 `matrix:`로 chunk_size × top_k × provider 조합 자동 expand
- **Multi-run 비교**: `bidmate-compare`로 여러 run 메트릭 비교 + best/worst 자동 분석
- **평가셋 검증**: 작성 실수 (`agency: "다중"` 같은 매칭 실패 값) 사전 차단
- **시나리오**:
  - **B (default)**: OpenAI `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
  - **A**: local HF/vLLM + PEFT 준비
- **UI**: Streamlit 3탭 (라이브 데모 + 문서 목록 + 평가)

## Quick Start

```bash
# 1. 환경 셋업
uv sync --group dev
cp .env.example .env  # OPENAI_API_KEY 입력

# 2. 환경 검증
uv run pytest tests/

# 3. 단일 RAG 쿼리 (디버깅)
uv run python scripts/run_rag.py \
    --question "국민연금공단 이러닝 사업 요구사항" \
    --provider-config configs/providers/openai_gpt5mini.yaml

# 4. 평가 1회 (작은 샘플 + judge 끔)
uv run bidmate-eval \
    --evaluation-path data/eval/eval_v1/eval_batch_01.csv \
    --provider-config configs/providers/openai_gpt5mini.yaml \
    --experiment-config configs/experiments/generation_compare.yaml \
    --limit 3 --skip-judge
```

선택 설치:
```bash
uv sync --group dev --group ui   # Streamlit 추가 패키지
uv sync --group dev --group ml   # PEFT/transformers 등 ML 학습용
```

## 주요 명령

| 목적 | 명령 |
|---|---|
| 단일 쿼리 (디버깅) | `uv run python scripts/run_rag.py --question "..." --provider-config ...` |
| 단일 평가 + 자동 리포트 | `uv run bidmate-eval --evaluation-path ... --provider-config ... --experiment-config ...` |
| 풀 사이클 (ingest → index → eval → report) | `uv run python scripts/run_experiment.py --experiment-config configs/experiments/<name>.yaml` |
| 마크다운 리포트 재생성 | `uv run bidmate-report --run-id bench-XXXXXXXX` |
| 여러 run 비교 | `uv run bidmate-compare --experiment <name>` 또는 `--run-ids bench-A bench-B ...` |
| 인덱스만 빌드 | `uv run python scripts/build_index.py --provider-config ... --chunks-path ...` |
| Streamlit UI (팀 디버깅) | `PYTHONPATH=. uv run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0` |
| 사용자용 웹 UI (Next.js) | `./scripts/run_web.sh` → http://localhost:3000 |

### 사용자용 웹 UI (Next.js + FastAPI)

실제 컨설턴트가 사용하는 채팅 인터페이스. 팀 디버깅용 Streamlit과 별도로 실행되며 같은 `bidmate_rag` 파이프라인을 그대로 재사용합니다.

**기능**: `@` 문서 멘션 + `/` 슬래시 커맨드 12개(`/요약` `/요구사항` `/일정` `/예산` `/비교` `/자격요건` `/평가기준` `/리스크` `/기본정보` `/제출서류` `/도움말` `/초기화`) · 좌측 답변 + 우측 sticky 근거 패널 · 접히는 사이드바(`⌘B`) · 문서 검색(`⌘K`) · **문서 미리보기 모달**(보관 문서 카드 클릭 시 좌측 메타데이터·요약 패널 + 우측 PDF 뷰어 2단 레이아웃, "질문 시작" 버튼 한 번으로 채팅 질의로 전환) · **전체 카탈로그 모달**(`⌘D`, sortable table + Shift+Click 멀티 정렬, 검색/필터 사이드바와 실시간 공유, 체크박스 다중 선택 후 `/비교` 자동 진입) · sessionStorage 기반 세션(F5는 유지, 탭 닫으면 사라짐)

```bash
# 최초 1회 — 프론트엔드 의존성 설치
cd web && npm install && cd ..

# 실행 (FastAPI 8100 + Next.js 3000 동시)
./scripts/run_web.sh
# → http://localhost:3000 (사용자 UI)
# → http://localhost:8100/docs (FastAPI 자동 문서)
```

포트를 바꾸려면:
```bash
API_PORT=8200 WEB_PORT=3100 ./scripts/run_web.sh
```

프로덕션 빌드:
```bash
./scripts/run_web.sh prod
```


### `bidmate-eval` 자주 쓰는 옵션

```bash
uv run bidmate-eval \
    --evaluation-path data/eval/eval_v1/eval_batch_01.csv \
    --provider-config configs/providers/openai_gpt5mini.yaml \
    --experiment-config configs/experiments/generation_compare.yaml \
    --limit 5                  # 평가셋 처음 N개만
    --filter-type A,B          # type 컬럼 필터
    --filter-difficulty 하,중  # 난이도 필터
    --skip-judge               # LLM judge 끄기 (시간/비용 절감)
    --judge-model gpt-4o-mini  # judge 모델 변경
    --strict                   # 평가셋 검증 경고도 fail (CI용)
    --no-validate              # 평가셋 검증 자체 skip
```

## YAML 한 장으로 Grid Search

```yaml
# configs/experiments/chunk_topk_grid.yaml
name: chunk-topk-grid
mode: full_rag
provider_configs:
  - configs/providers/openai_gpt5mini.yaml
matrix:
  chunk_size: [500, 1000, 1500]
  retrieval_top_k: [3, 5, 8]
```

```bash
uv run python scripts/run_experiment.py --experiment-config configs/experiments/chunk_topk_grid.yaml
```

→ **9개 sub-experiment 자동 expand + 순차 실행 + 9개 마크다운 리포트 자동 생성**

매트릭스 규칙:
- `mode == full_rag` → chunking 격리 (collection 자동 분리)
- `mode == generation_only` → collection 공유 (LLM만 비교, 인덱스 재사용)
- 키 정렬은 결정론적 (`sorted()`)
- 청킹 변경 없는 sub-experiment는 base의 청크 디렉토리 재사용

## 출력 파일 구조

```
artifacts/
├── logs/
│   ├── runs/
│   │   ├── {run_id}.jsonl                  # 질문별 상세 (cost_usd, judge_scores 포함)
│   │   └── {run_id}.meta.json              # timestamp, git, configs, judge cost
│   ├── benchmarks/
│   │   └── {experiment_name}.parquet       # 요약 (run마다 append, 이전 row 보존)
│   └── embeddings/
│       └── {collection_name}.json          # 임베딩 토큰/비용/built_at
└── reports/
    └── {YYYY-MM-DD}_{HHMM}_{exp}_{model}.md  # 노션 복사용 (이름순=시간순 정렬)
```

## 평가셋 형식

`data/eval/eval_v1/eval_batch_*.csv` (버전 디렉토리, `find_latest_eval_dir`이 자동 탐지):

```csv
id,type,difficulty,question,ground_truth_answer,ground_truth_docs,metadata_filter,history
Q001,A,하,"한국가스공사 ERP 사업 예산은?",...,"[""파일명.hwp""]","{""agency"":""한국가스공사""}","[]"
```

| 필드 | 형식 | 자동 처리 |
|---|---|---|
| `ground_truth_docs` | JSON list (파일명) | Hit Rate 채점 시 파일명/사업명/doc_id 셋 모두 매칭 |
| `metadata_filter` | JSON dict (영문 키) | `EVAL_FILTER_KEY_MAP`으로 한국어 키 자동 정규화 (`agency` → `발주 기관`, `year` → `공개연도` int) |
| `history` | JSON list of `{role, content}` (OpenAI 표준) | multi-turn 평가 시 chat_history 전달 |

새 평가셋 버전 추가:
```bash
mkdir data/eval/eval_v2
cp 새 평가셋 *.csv data/eval/eval_v2/
# → 코드 수정 0줄. find_latest_eval_dir()이 자동으로 eval_v2/ 사용
```

## 시나리오 설정

| 시나리오 | Provider config | 모델 |
|---|---|---|
| B (OpenAI) | `configs/providers/openai_gpt5.yaml` | gpt-5 |
| B | `configs/providers/openai_gpt5mini.yaml` | gpt-5-mini |
| B | `configs/providers/openai_gpt5nano.yaml` | gpt-5-nano |
| A (Local) | `configs/providers/local_hf.yaml` | HuggingFace |
| A | `configs/providers/local_vllm.yaml` | vLLM |

실험 config:
- `configs/experiments/generation_compare.yaml` — 같은 인덱스에 LLM만 비교 (mode: generation_only)
- `configs/experiments/full_rag_compare.yaml` — chunking까지 비교 (mode: full_rag)

## 비용 가이드 (`configs/pricing.yaml`)

| 모델 | input | cached input | output |
|---|---|---|---|
| gpt-5-mini | $0.25 / 1M | $0.025 (10%) | $2.00 / 1M |
| gpt-4o | $2.50 / 1M | $1.25 (50%) | $10.00 / 1M |
| gpt-4o-mini | $0.15 / 1M | $0.075 (50%) | $0.60 / 1M |
| text-embedding-3-small | $0.02 / 1M | — | — |
| text-embedding-3-large | $0.13 / 1M | — | — |

실측 참고 (gpt-5-mini):
- 인덱스 빌드 1회 (14k 청크): ~$0.29
- 평가 2 샘플 + judge: ~$0.003
- 평가 580 샘플 + judge (전체 평가셋): ~$1 추정

## 안전장치 — 침묵 실패 차단

| 영역 | 보호 | 효과 |
|---|---|---|
| 평가셋 무결성 | `validate_eval_samples()` (load 직후 자동) | 작성 실수(`{"agency":"다중"}` 같은 매칭 실패 값) 사전 발견 |
| Chunking 격리 | `mode=full_rag`면 collection_name에 experiment.name prefix | chunking 비교 실험이 섞이지 않음 |
| Stale chunk | `replace_documents()` (build_index 자동) | 청크 수 줄어도 이전 빌드 잔존 0 |
| Parquet 보존 | `persist_benchmark_summary` append-or-replace | provider 비교 시 이전 run 안 사라짐 |
| Metadata 격리 | `_resolve_metadata_path()` (실험별 sub-dir 우선) | MetadataStore가 stale 파일 안 봄 |
| Cost 정확성 | `cached_tokens` 자동 반영 | 캐시된 prompt가 일반 단가로 청구되는 오류 방지 |
| Top-k 적용 | `runtime.experiment.retrieval_top_k`가 retriever까지 전달 | top_k 설정이 실제로 동작 |
| History 형식 | OpenAI 표준 + legacy 둘 다 지원 | multi-turn 평가가 KeyError로 죽는 것 방지 |

## 프로젝트 구조

```
src/bidmate_rag/
├── loaders/         # 문서 파싱 (kordoc → pdfplumber 폴백)
├── preprocessing/   # cleaner + chunker
├── providers/       # 임베딩 (OpenAI, HF) + LLM (OpenAI, HF)
├── retrieval/       # ChromaVectorStore + RAGRetriever + filters
├── evaluation/      # BenchmarkRunner + metrics + judge + dataset + schema_validator + pipeline
├── tracking/        # pricing, git_info, markdown_report, comparison
├── experiments/     # matrix grid expand
├── cli/             # bidmate-eval / bidmate-report / bidmate-compare 본체
├── pipelines/       # ingest, build_index, chat, runtime
├── config/          # settings (Pydantic) + prompts
├── storage/         # MetadataStore (parquet 기반)
└── schema.py        # 공통 모델: Document, Chunk, RetrievedChunk, GenerationResult

scripts/             # CLI shim (bidmate-eval 등)
app/                 # Streamlit UI
configs/             # base, providers, experiments, chunking, pricing.yaml
data/eval/eval_v1/   # 평가셋 (버전 디렉토리)
artifacts/           # logs/, reports/ (gitignore 예외)
```

## 테스트

```bash
uv run pytest tests/        # 112건 (unit + integration)
uv run ruff check .         # All checks passed
```

## 주의사항

- **API 키는 `.env`에서 관리**: 절대 커밋 금지
- **원본 RFP 문서는 비공개**: git에 올리지 말 것
- **gpt-5 계열은 reasoning tokens 사용**: completion에 포함되어 청구됨 → cost가 예상보다 클 수 있음
- **항상 프로젝트 루트에서 실행**: 일부 경로가 cwd 의존
- **ChromaDB 메타 키에 공백**: `발주 기관`, `사업 금액` (공백 포함, CLAUDE.md 참고)

## 프로젝트 문서

- [`docs/project-structure.md`](docs/project-structure.md): 저장소 구조 가이드
- [`docs/architecture.md`](docs/architecture.md): 파이프라인 구조 (구버전 가능성)
- [`docs/decision-log.md`](docs/decision-log.md): 실험/기술 의사결정 기록
- [`docs/collaboration/branch-strategy.md`](docs/collaboration/branch-strategy.md): 팀 브랜치 전략
- [`docs/collaboration/git-worktree-workflow.md`](docs/collaboration/git-worktree-workflow.md): worktree 워크플로우
- [`CLAUDE.md`](CLAUDE.md): 프로젝트 가이드 (Claude Code 협업 시 필독)

## 협업 규칙

- 기능 개발/실험은 `feat/<initial>/<topic>` 브랜치에서 (CLAUDE.md 참고)
- `develop`에 직접 커밋 금지 — 반드시 브랜치에서 작업 후 머지
- 원본 RFP 데이터와 실험 산출물(`artifacts/`)은 git에 올리지 않음
