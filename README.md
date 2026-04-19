<div align="center">

# 📘 삼삼오오 RAG

**공공입찰 RFP를 빠르게 탐색하고 비교할 수 있도록 만든 Evidence-grounded RAG 시스템**

공공입찰 컨설턴트가 수백 페이지짜리 RFP에서 핵심 요구사항, 예산, 일정, 평가기준을 더 빠르게 파악할 수 있도록 설계한 질의응답 플랫폼입니다.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-10A37F?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-F97316)
![Next.js](https://img.shields.io/badge/Next.js-Web_UI-000000?logo=nextdotjs&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Debug_UI-FF4B4B?logo=streamlit&logoColor=white)

</div>

> **YAML 1장으로 실험을 정의하고, CLI 1줄로 실행하면 비용·메트릭·메타데이터가 자동 기록되며, 노션 친화 마크다운 리포트가 생성됩니다.**

## 🔗 Project Links

| 항목 | 링크 |
|---|---|
| Team Hub | [Notion Team Hub](https://www.notion.so/sprint-ai07-bidsupportconsulting/Team-Hub-334b6ef436c78097a4d2cf1637bedb18?source=copy_link) |
| 아키텍처 문서 | [`docs/architecture.md`](docs/architecture.md) |
| 프로젝트 구조 | [`docs/project-structure.md`](docs/project-structure.md) |
| 의사결정 기록 | [`docs/decision-log.md`](docs/decision-log.md) |
| 협업 규칙 | [`docs/collaboration/branch-strategy.md`](docs/collaboration/branch-strategy.md) |
| PPT | 추가 예정 |
| 보고서 | 추가 예정 |
| 협업 일지 | [Notion Logs](https://www.notion.so/sprint-ai07-bidsupportconsulting/Logs-334b6ef436c780c7976fedcdeae5b672?source=copy_link) |
| Test Record | [Notion Test Record](https://www.notion.so/sprint-ai07-bidsupportconsulting/Test-Record-335b6ef436c780cb93aac9852858cd2c?source=copy_link) |

## 🧭 Project Overview

### 1. Problem

공공입찰 RFP는 길고 복잡합니다. 컨설턴트는 수백 페이지 문서에서 요구사항, 자격요건, 평가기준, 예산, 일정 같은 핵심 정보를 빠르게 찾아야 하지만, 기존 업무는 대부분 수작업에 의존했습니다.

### 2. Goal

BidMate RAG는 공공입찰 RFP를 질문형으로 탐색하고, 문서 근거를 함께 제시하며, 여러 문서를 비교할 수 있는 실무형 RAG 도구를 만드는 것을 목표로 합니다.

### 3. Approach

- `멀티턴 재작성 + 하이브리드 검색 + 규칙/섹션 부스트`를 기본 운영 검색 경로로 사용합니다.
- `CLI 평가 도구 + Streamlit 디버깅 UI + Next.js 사용자 채팅 UI`를 함께 제공해 연구와 사용 경험을 모두 다룹니다.
- 검색 성능뿐 아니라 `Faithfulness`, `Answer Relevance`, `Context Precision`, `Context Recall`까지 함께 평가합니다.
- 평가셋 검증, 비용 추적, chunking 격리, stale chunk 방지 같은 신뢰성 장치를 기본으로 포함합니다.

## 📌 Project Summary

| 항목 | 내용 |
|---|---|
| 프로젝트명 | BidMate RAG |
| 목표 | 공공입찰 RFP를 근거 기반으로 탐색·비교·질의응답하는 시스템 구축 |
| 주요 사용자 | 입찰메이트 컨설팅 실무 컨설턴트 |
| 대상 문서 | 공공입찰 RFP 원본 문서 101건 |
| 문서 구성 | HWP 96건 / PDF 4건 / DOCX 1건 |
| 핵심 파이프라인 | 파싱 → 정제 → 청킹 → 인덱싱 → 검색 → 생성 → 평가 → 리포트 |
| 운영 시나리오 | 시나리오 A: 로컬 LLM / 시나리오 B: 상용 API LLM |
| 기본 검색 전략 | 멀티턴 재작성 + 하이브리드 검색 + 규칙/섹션 부스트 |
| 평가 방식 | 80문항 평가셋 + LLM-as-a-Judge 기반 다면 평가 |
| 산출물 | CLI 평가 도구 + Streamlit 디버깅 UI + Next.js 사용자 채팅 UI |
| 사용자 기능 | `@` 문서 멘션 + `/` 슬래시 커맨드 12종 + 근거 패널 |
| 협업 방식 | 시스템/데이터/검색·생성/평가/프론트엔드 역할 기반 분업 |

## 📊 Results

### KPI Snapshot

> ⚠️ `KPI Snapshot`은 현재 재정리 중이며, 대표 지표 구성과 수치는 운영 경로 기준으로 업데이트할 예정입니다.

| 지표 | 결과 |
|---|---|
| 대상 문서 처리 | 101건 |
| 평가셋 규모 | 80문항 |
| MRR | 0.988 |
| Latency | 5.3초 (`Top-K=5`) |
| Cost per Query | `$0.001` (`Top-K=5`) |
| 기본 사용자 인터페이스 | Next.js + FastAPI |
| 팀 디버깅 인터페이스 | Streamlit 3탭 UI |

### What We Improved

| Baseline 한계 | 개선 방식 |
|---|---|
| 후속 질문에서 대명사 맥락이 끊김 | 멀티턴 재작성으로 독립 검색 쿼리 복원 |
| 숫자·고유명사·정확 키워드 검색이 약함 | Dense + Sparse를 결합한 하이브리드 검색 적용 |
| 질문과 무관한 청크가 상위에 노출됨 | 규칙 기반 부스트와 섹션 단서 활용 |
| 평가/실험 결과 추적이 어려움 | 비용·토큰·Latency·Git·설정 스냅샷 자동 기록 |
| 데이터 품질 문제로 침묵 실패 발생 가능 | 평가셋 검증, metadata 정합성 관리, stale chunk 방지 |

### Why This Matters

- 제한된 자원 환경 때문에 `Cross-Encoder`를 기본 운영 경로에 넣지 않았습니다.
- 대신 `하이브리드 검색 + 규칙/섹션 부스트`를 고도화해 더 가벼운 비용으로 검색 품질을 끌어올렸습니다.
- 결과적으로 `실사용 가능한 속도`, `낮은 질의 비용`, `재현 가능한 평가 흐름`을 함께 확보했습니다.

## 🧪 Experiment Demo

### 1. User Demo

<p align="center">
  <img src="./reports/figures/nextjs-user-demo.png" alt="Next.js user demo" width="92%" />
</p>

> Next.js 사용자 UI에서 문서 검색, 필터링, 보관 문서 탐색, 채팅 입력, 슬래시 커맨드 시작점까지 한 화면에서 확인할 수 있는 데모입니다. 실제 응답이 생성되면 우측 근거 카드 영역에 출처 설명이 함께 표시됩니다.

- 실제 컨설턴트가 사용하는 채팅 인터페이스입니다.
- `@` 문서 멘션, `/` 슬래시 커맨드 12종, 근거 패널, 문서 미리보기, 전체 카탈로그 탐색을 지원합니다.

```bash
./scripts/run_web.sh
```

> 사용자용 웹 UI는 `http://localhost:3000`, FastAPI 문서는 `http://localhost:8100/docs`에서 확인할 수 있습니다.

### 2. Evaluation Demo

<p align="center">
  <img src="./reports/figures/streamlit-evaluation-demo.png" alt="Streamlit evaluation demo" width="88%" />
</p>

> Streamlit 평가 탭에서 평가셋, Provider, 청킹 전략, Judge 모델을 선택해 일괄 평가를 실행하는 흐름입니다. 이후 검색 메트릭, LLM Judge 결과, 비용·지연, 질문별 결과 리포트로 이어집니다.

```bash
uv sync --group dev --group ui
PYTHONPATH=. uv run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

> 팀 내부 디버깅 및 평가용 UI는 `http://localhost:8501`에서 확인할 수 있습니다.

### 3. Pipeline Demo

```bash
uv run python scripts/run_experiment.py \
    --experiment-config configs/experiments/generation_compare.yaml \
    --eval-path data/eval/eval_v1/eval_batch_01.csv
```

> YAML 실험 설정 파일 하나로 `ingest → build_index → eval → report`를 순차 실행하는 전체 파이프라인 데모입니다. 실행 결과는 `artifacts/logs/`와 `artifacts/reports/`에 자동 저장됩니다.


## 🚀 Quick Start

```bash
# 1. 기본 환경 셋업
uv sync --group dev
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
```

> 실행 명령은 각 `Demo` 섹션에서 바로 확인할 수 있습니다.

## 🏗 Architecture

```mermaid
flowchart LR
    A[RFP 101건] --> B[문서 파싱]
    B --> C[정제와 청킹]
    C --> D[임베딩과 인덱싱]
    H[사용자 질문 + 대화 이력] --> E[멀티턴 재작성]
    E --> F[하이브리드 검색]
    D --> F
    F --> G[규칙/섹션 부스트]
    G --> I[답변 생성]
    I --> J[평가와 리포트]
```

> 자세한 구조와 설계 배경은 [`docs/architecture.md`](docs/architecture.md)에서 확인할 수 있습니다.

## ✅ Evaluation & Reliability

| 영역 | 내용 |
|---|---|
| 검색 평가 | `Hit Rate@5`, `MRR`, `nDCG@5` |
| 생성 평가 | `Faithfulness`, `Answer Relevance`, `Context Precision`, `Context Recall` |
| 운영 추적 | 비용, 토큰, Latency, Git 해시, 설정 스냅샷, 마크다운 리포트 자동 기록 |
| 평가셋 검증 | 샘플 로딩 직후 형식 및 metadata_filter 정합성 검사 |
| Chunking 격리 | `mode=full_rag` 실험에서 collection을 분리해 실험 간 오염 방지 |
| Stale chunk 방지 | 인덱스 재구축 시 이전 청크 잔존 데이터 제거 |
| 비교 분석 | `bidmate-compare`로 여러 run의 메트릭과 best/worst 결과 비교 |

## 📚 Project Docs

- [`docs/project-structure.md`](docs/project-structure.md): 저장소 구조 가이드
- [`docs/architecture.md`](docs/architecture.md): 파이프라인 구조와 설계 배경
- [`docs/decision-log.md`](docs/decision-log.md): 실험 및 기술 의사결정 기록
- [`docs/collaboration/branch-strategy.md`](docs/collaboration/branch-strategy.md): 브랜치 전략
- [`docs/collaboration/git-worktree-workflow.md`](docs/collaboration/git-worktree-workflow.md): worktree 협업 워크플로우
- [`CLAUDE.md`](CLAUDE.md): 프로젝트 협업 가이드

## 🗂 Project Structure

```text
src/bidmate_rag/
├── loaders/         # 문서 파싱
├── preprocessing/   # cleaner + chunker
├── providers/       # 임베딩 + LLM
├── retrieval/       # retriever + filters
├── evaluation/      # metrics + judge + benchmark
├── tracking/        # pricing + report + comparison
├── pipelines/       # ingest + build_index + chat
└── config/          # settings + prompts

app/                 # Streamlit UI
web/                 # Next.js 사용자 UI
configs/             # provider / experiment / pricing 설정
data/eval/eval_v1/   # 평가셋
artifacts/           # logs / reports
```

## ⚠ Notes

- API 키는 `.env`에서 관리하며 절대 커밋하지 않습니다.
- 원본 RFP 문서는 비공개 데이터이므로 저장소에 포함하지 않습니다.
- 일부 스크립트는 프로젝트 루트 기준 경로를 사용하므로 루트에서 실행하는 것을 권장합니다.

## 🔭 Future Work

- README용 데모 이미지와 GIF 추가
- 공개 가능한 샘플 데이터 기반의 외부 시연 버전 정리
- 멀티턴 슬롯 메모리와 숫자/단위 검색 정밀도 강화
- 시나리오 A 로컬 LLM 품질 개선과 배포 전략 구체화

## 🤝 Team & Contributions

> 아래 표는 각 팀원의 주도 역할을 중심으로 정리한 내용이며, 실제 프로젝트 진행 과정에서는 역할 경계를 넘나들며 함께 기획·실험·문서화·검증을 협업했습니다.

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>▶ Lead Role<br />▷ Sub Role</th>
      <th>Role Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img src="./reports/figures/team-rabbit-lead.png" alt="Rabbit mascot for 김도민" width="96" /><br />
        <strong>김도민</strong>
      </td>
      <td>
        <strong>▶ Project Manager</strong><br />
        ▷ Evaluation / Documentation
      </td>
      <td>
        <ul>
          <li>프로젝트 일정 관리와 역할 분배, 우선순위 조정</li>
          <li>회의 주도, 노션/이슈보드 운영, 협업 체계 관리</li>
          <li>마일스톤 및 제출 일정 관리, A/B 시나리오 진행 상황 통합, 초기 파이프라인 구축</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="./reports/figures/team-fox-retrieval.png" alt="Fox mascot for 김예주" width="96" /><br />
        <strong>김예주</strong>
      </td>
      <td>
        <strong>▶ Retrieval / Embedding</strong><br />
        ▷ Chunking / Metadata
      </td>
      <td>
        <ul>
          <li>임베딩 모델과 Vector DB 선정</li>
          <li>문서 임베딩 생성과 similarity search, Top-k, MMR, re-ranking 기반 검색 성능 고도화</li>
          <li>검색 성능 비교 실험과 retrieval 파이프라인 개선</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="./reports/figures/team-raccoon-chunking.png" alt="Raccoon mascot for 김정우" width="96" /><br />
        <strong>김정우</strong>
      </td>
      <td>
        <strong>▶ Chunking / Metadata</strong><br />
        ▷ Document / Data
      </td>
      <td>
        <ul>
          <li>청킹 전략 설계와 chunk size / overlap 실험</li>
          <li>섹션·문단 기반 분할 검토와 문서 구조 기반 chunking 개선</li>
          <li>메타데이터 스키마 설계, 기관명·사업명 정규화, 메타데이터 필터링 기준 정리</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="./reports/figures/team-squirrel-data.png" alt="Squirrel mascot for 김준혁" width="96" /><br />
        <strong>김준혁</strong>
      </td>
      <td>
        <strong>▶ Document / Data</strong><br />
        ▷ Generation / Prompt
      </td>
      <td>
        <ul>
          <li>PDF/HWP 문서 로딩, 텍스트 추출 및 정제</li>
          <li>data_list.csv 메타데이터 확인 및 매핑, 문서 구조 파악</li>
          <li>문서 전처리 흐름 정리와 입력 데이터 정합성 관리</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="./reports/figures/team-owl-review.png" alt="Owl mascot for 손승만" width="96" /><br />
        <strong>손승만</strong>
      </td>
      <td>
        <strong>▶ Generation / Prompt</strong><br />
        ▷ Evaluation / Documentation
      </td>
      <td>
        <ul>
          <li>LLM 선정과 프롬프트 설계, 답변 포맷·톤·길이 제어</li>
          <li>검색된 컨텍스트 기반 응답 생성과 후속 질문 대응</li>
          <li>hallucination 억제와 응답 품질 개선</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
