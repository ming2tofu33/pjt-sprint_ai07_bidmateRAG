# 시나리오 A-2 실험 가이드 (Scenario A-2 Evaluation Guide)

이 파일은 **BGE-M3(임베딩)**와 **Gemma-4(LLM)**를 고정한 상태에서, 새롭게 선별된 **8종의 핵심 프롬프트(configs/prompts_2)**를 비교 실험하고 분석하는 절차를 안내합니다.

## 1. 실험 환경 (고정)
- **임베딩**: BAAI/bge-m3
- **LLM**: google/gemma-4-E4B-it (4-bit)
- **청킹**: 1000자 / 150 오버랩
- **평가셋**: `eval_batch_45_rank_11-30.csv` (20문제)

## 2. 실험 수행 방법 (자동화)

8개의 실험을 연속으로 실행하고 개별 리포트까지 자동으로 생성하려면 제공된 쉘 스크립트를 사용합니다.

```bash
# 1. 실행 권한 부여 (최초 1회)
chmod +x run_all_a2_experiments.sh

# 2. 전회 실험 실행 (약 15분 소요)
./run_all_a2_experiments.sh
```

## 3. 결과물 저장 위치

실험 결과는 다음 경로들에 자동으로 정리되어 저장됩니다.

| 결과물 유형 | 저장 위치 (Path) | 파일명 형식 |
| :--- | :--- | :--- |
| **개별 실험 로그** | `artifacts/logs/runs/` | `{run_id}.jsonl`, `{run_id}.meta.json` |
| **개별 분석 리포트** | `artifacts/reports/scenario_a2/` | `P{N}_C{N}_F{N}_S{N}_T{N}.md` |
| **종합 비교 리포트** | `artifacts/reports/` | `scenario_a2_prompt_comparison.md` |
| **시각화 분석 보고서**| 프로젝트 루트 (`./`) | `scenario_a2_comparison_report.ipynb` |

---

## 4. 실험 상세 프로세스 (수동 실행 시)

특정 프롬프트만 개별적으로 다시 테스트하고 싶을 때 사용합니다.

### Step 1: 실험 실행
```bash
uv run python scripts/run_eval.py \
  --evaluation-path data/eval/eval_v1/eval_batch_45_rank_11-30.csv \
  --provider-config configs/providers/scenario_a_tmp.yaml \
  --prompt-config configs/prompts_2/prompt_P3_C3_F3_S3_T3.yaml \
  --run-id p3_c3_f3_s3_t3 \
  --progress
```

### Step 2: 개별 리포트 생성
```bash
uv run python scripts/generate_report.py --run-id p3_c3_f3_s3_t3
```

---

## 5. 최종 종합 분석 (Analysis)

모든 실험이 완료되면 아래 두 가지 리포트를 통해 분석을 마무리합니다.

### 1️⃣ CLI 종합 비교 (지표 중심)
8개 실험의 주요 수치(Faithfulness, Latency 등)를 한눈에 비교합니다.
```bash
uv run python scripts/compare_runs.py \
  --run-ids p0_c4_f4_s4_t4 p1_c4_f4_s3_t4 p2_c3_f3_s1_t5 p4_c2_f5_s3_t5 p3_c3_f3_s3_t3 p3_c3_f3_s4_t4 p2_c0_f2_s5_t1 p2_c0_f4_s1_t2 \
  --output artifacts/reports/scenario_a2_prompt_comparison.md
```

### 2️⃣ 시각화 분석 보고서 (인사이트 중심)
`scenario_a2_comparison_report.ipynb` 파일을 열어 다음 4개 그룹에 대한 가설 검증을 진행합니다.
- **Group 1 (복잡도)**: 850자 이상 프롬프트의 성능 임계점 분석
- **Group 2 (English CoT)**: T5 전략의 실효성 최종 검증
- **Group 3 (High Score)**: Control Score와 실제 품질의 상관관계
- **Group 4 (Low Score)**: 최소 구성 프롬프트의 가성비 검증
