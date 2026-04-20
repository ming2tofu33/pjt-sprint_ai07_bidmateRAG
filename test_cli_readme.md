# 시나리오 A 실험 가이드 (Scenario A Evaluation Guide)

이 파일은 **BGE-M3(임베딩)**와 **Gemma-4(LLM)**를 고정한 상태에서 **P-C-F-S-T 기반 6가지 프롬프트**를 비교 실험하고 분석하는 절차를 안내합니다.

## 1. 실험 환경 (고정)
- **임베딩**: BAAI/bge-m3
- **LLM**: google/gemma-4-E4B-it (4-bit)
- **청킹**: 1000자 / 150 오버랩
- **평가셋**: `eval_batch_41_rank_1-20.csv` (20문제)

## 2. 사전 준비
시나리오 A 통합 설정을 생성합니다 (최초 1회).

```bash
cat <<EOF > configs/providers/scenario_a_tmp.yaml
provider: huggingface
scenario: scenario_a
model: google/gemma-4-E4B-it
embedding_model: BAAI/bge-m3
EOF
```

## 3. 실험 수행 및 분석 프로세스

각 프롬프트별로 다음 3단계를 수행합니다.

### Step 1: 실험 실행
`run-id`는 프롬프트 명칭과 일치시켜 실행합니다.

```bash
# 예: P3 프롬프트 실험
uv run python scripts/run_eval.py \
  --evaluation-path data/eval/eval_v1/eval_batch_41_rank_1-20.csv \
  --provider-config configs/providers/scenario_a_tmp.yaml \
  --prompt-config configs/prompts/prompt_P3_C2_F3_S2_T3.yaml \
  --run-id p3 \
  --progress
```

### Step 2: 답변 품질 및 지표 확인
실행 결과 로그(`artifacts/logs/runs/{run_id}.jsonl`)를 분석하여 지표가 낮게 나온 원인을 파악합니다.

```bash
# 지표가 낮은(faithfulness < 1.0) 사례 추출 및 분석
uv run python -c "
import json
with open('artifacts/logs/runs/p3.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]
low_cases = [r for r in results if r.get('judge_scores', {}).get('faithfulness', 1.0) < 1.0]
for d in low_cases[:2]:
    print(f'Q: {d[\"question\"]}\nA: {d[\"answer\"]}\nScores: {d[\"judge_scores\"]}\n' + '-'*40)
"
```

### Step 3: 결과 보고서 작성
실험 결과를 `P{N}_C{N}_F{N}_S{N}_T{N}.md` 파일로 정리하여 저장합니다.

## 4. 종합 비교 (Compare)

모든 실험(P1~P6)이 완료되면 종합 비교 리포트를 생성합니다.

```bash
# 6개 실험 한눈에 비교
uv run python scripts/compare_runs.py \
  --run-ids p1 p2 p3 p4 p5 p6 \
  --output artifacts/reports/scenario_a_prompt_comparison.md

# 개별 노션용 상세 리포트 생성 (필요 시)
uv run python scripts/generate_report.py --run-id p3
```

## 5. 지표 해석 가이드 (인사이트 도출 팁)
- **Faithfulness 하락**: 모델이 거짓 정보를 생성(환각)했거나, 너무 솔직하게 "없음"이라고 답하여 판정 모델이 오해했을 수 있음.
- **Answer Relevance 하락**: 답변이 너무 짧거나(P2 사례), 질문의 핵심 의도보다 구조(섹션 분리)에 집중했을 때 발생.
- **Latency 단축**: 답변이 간결해질수록 속도는 빨라지지만 품질과의 트레이드오프가 발생함.
