"""Markdown templates for experiment reports.

실험 리포트 마크다운 템플릿.
str.format(**ctx)으로 렌더링되며, 모든 placeholder는 context dict에 존재해야 한다.
없는 값은 "N/A"로 채운다. markdown_report.render_markdown에서 사용.
"""

from __future__ import annotations

REPORT_TEMPLATE = """\
> ⚠️ 자동 생성된 리포트입니다. "🤖 자동 생성" 영역은 수정하지 마세요.

# 📋 노션 속성 (수동 입력 필요)

> 노션 DB 새 페이지 생성 시 아래 값을 속성 패널에 그대로 입력하세요.

| 속성명 | 값 |
| --- | --- |
| 실험명 | {experiment_name} |
| run_id | {run_id} |
| 날짜 | {timestamp_kst} |
| 시나리오 | {scenario} |
| 평가셋 버전 | {eval_basename} |
| 임베딩 모델 | {embedding_model} |
| 생성 모델 | {llm_model} |
| Chunk Size | {chunk_size} |
| Top-k | {top_k} |
| Git Commit | {git_commit_short} |
| Hit Rate@{top_k} | {hit_rate} |
| MRR | {mrr} |
| MAP@{top_k} | {map} |
| Faithfulness | {faithfulness} |
| Latency Avg (s) | {latency_avg_s} |
| Total Tokens | {total_tokens} |
| Cost (USD) | {grand_total_cost} |

> 수동 입력 속성: 담당자, 상태, 실험 축, 변경 영역, 채택 여부

---

# 🤖 자동 생성 본문 (수정 금지)

## 설정 스냅샷

| 항목 | 값 |
| --- | --- |
| run_id | {run_id} |
| 실험명 | {experiment_name} |
| 실행 시각 | {timestamp_kst} |
| Git Branch | {git_branch} |
| Git Commit | {git_commit} {dirty_marker} |
| 평가셋 | {eval_path} ({num_samples} samples) |
| 임베딩 모델 | {embedding_model} |
| 생성 모델 | {llm_model} |
| Vector DB | ChromaDB ({collection_name}) |
| Chunk Size / Overlap | {chunk_size} / {chunk_overlap} |
| Top-k | {top_k} |
| Provider Label | {provider_label} |

## 핵심 결과 표

| 지표 | 값 |
| --- | --- |
| Hit Rate@{top_k} | {hit_rate} |
| MRR | {mrr} |
| nDCG@{top_k} | {ndcg} |
| MAP@{top_k} | {map} |
| Faithfulness | {faithfulness} |
| Answer Relevance | {answer_relevance} |
| Context Precision | {context_precision} |
| Context Recall | {context_recall} |
| Latency Avg (s) | {latency_avg_s} |
| Latency P95 (s) | {latency_p95_s} |
| Prompt Tokens (sum) | {prompt_tokens_sum} |
| Completion Tokens (sum) | {completion_tokens_sum} |
| Total Tokens | {total_tokens} |
| 생성 비용 (USD) | {generation_cost} |
| 임베딩 비용 (USD) | {embedding_cost} |
| Judge 비용 (USD) | {judge_cost} |
| **Cost (USD)** | **{grand_total_cost}** |
{cost_warning}{gpt5_warning}
## 리소스 링크

- 상세 결과 (JSONL): `{run_jsonl_path}`
- 벤치마크 (Parquet): `{benchmark_parquet_path}`
- 메타 (JSON): `{meta_json_path}`
- 사용 Config:
{config_links}

---

# ✍️ 사람이 작성하는 영역

## 실험 개요
- **실험 제목**: {experiment_title}
- **실험 개요**: {experiment_overview}
- **최종 판단**: `채택 / 보류 / 미채택`

## 1. 가설
{hypothesis_bullets}

## 2. 결과 해석
### 변경점
{changes_bullets}

### 기대 결과
{expected_outcome_bullets}

## 3. 대표 실패 사례
{failure_case_blocks}

## 4. 핵심 인사이트
-

## 5. 다음 액션
{next_actions_bullets}

## 구현 메모
- 수정한 파일:
- 구현 중 막힌 점:
- 디버깅 포인트:
"""
