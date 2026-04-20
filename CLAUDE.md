# CLAUDE.md — BidMate RAG 프로젝트 가이드

## 프로젝트 개요

공공입찰 RFP(제안요청서) 문서 100건을 대상으로 한 RAG 질의응답 시스템.
입찰메이트 컨설팅 스타트업의 사내 도구로, 컨설턴트가 RFP 핵심 정보를 빠르게 파악하도록 지원.

## Git 브랜치 전략

- **`main`**: 배포/제출 기준선. 직접 작업 금지.
- **`develop`**: 팀 공통 기준선. 모든 기능 브랜치는 여기서 분기.
- **`feat/<initial>/<topic>`**: 개인 기능 브랜치.
- **`fix/<initial>/<topic>`**: 버그 수정 브랜치.
- **`docs/<initial>/<topic>`**: 문서 작업 브랜치.

### 작업 흐름

```bash
# 1. develop에서 새 브랜치 생성
git checkout develop && git pull origin develop
git checkout -b feat/dm/my-feature

# 2. 작업 후 커밋
git add <files>
git commit -m "feat: 설명"

# 3. 푸시 후 develop에 머지
git push -u origin feat/dm/my-feature
git checkout develop && git merge feat/dm/my-feature --no-edit && git push origin develop
```

### 규칙

- `develop`에 직접 커밋하지 말 것 — 반드시 브랜치에서 작업 후 머지
- `main`에는 안정화된 시점에만 develop에서 반영
- 커밋 메시지 형식: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`

## 기술 스택

| 구분 | 기술 |
|---|---|
| 파싱 | kordoc (Node.js CLI) |
| 청킹 | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter |
| 임베딩 | text-embedding-3-small (OpenAI) |
| 벡터 DB | ChromaDB (PersistentClient, cosine) |
| LLM | gpt-5-mini (시나리오 B) |
| UI | Streamlit |
| 패키지 | Python 3.12 + uv |

## 프로젝트 구조

```
src/bidmate_rag/          # 메인 패키지 → src/CLAUDE.md 참조
app/                      # Streamlit UI → app/CLAUDE.md 참조
scripts/                  # CLI 스크립트
configs/                  # 설정 파일 (base, providers, chunking)
data/                     # 원본/처리된 데이터 + 평가셋
experiments/notebooks/    # 실험 노트북 01~08
docs/                     # 문서 → docs/CLAUDE.md 참조
```

## 주요 명령어

```bash
# 환경 세팅
uv sync --group dev
cp .env.example .env    # OPENAI_API_KEY 입력

# 파이프라인 실행
uv run python scripts/ingest_data.py
uv run python scripts/build_index.py --provider-config configs/providers/openai_gpt5mini.yaml

# Streamlit UI
PYTHONPATH=. uv run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

## 주의사항

- **ChromaDB 메타데이터 키에 공백 있음**: `발주 기관`, `사업 금액` (공백 포함)
- **gpt-5-mini는 temperature 미지원**: 기본값(1)으로 동작, 프롬프트로 제어
- **gpt-5 계열은 reasoning 토큰 사용**: `max_completion_tokens`을 넉넉하게 (16000+)
- **원본 RFP 문서는 비공개**: git에 올리지 말 것
- **API 키는 .env에서 관리**: 절대 커밋하지 말 것
