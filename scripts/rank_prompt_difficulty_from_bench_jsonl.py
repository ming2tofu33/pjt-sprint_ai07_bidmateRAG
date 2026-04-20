"""bench-*.jsonl 한 파일만으로 '프롬프트 실험용' 문항 난이도 순위를 낸다.

설계 요지 (시나리오 A: 프롬프트만 바꿈)
--------------------------------------
- 생성 품질(프롬프트 영향권) 비중을 가장 크게 둔다.
- 전이(컨텍스트 정밀도/리콜)는 '가져온 근거를 질문에 맞게 썼는가'에 가깝고 프롬프트와 상호작용한다.
- 검색 단계는 원래 Hit@k/MRR이 맞지만, jsonl 단독으로는 정답 문서 ID가 없어
  **상위 검색 청크의 dense score 평균**만으로 약한 프록시를 쓴다 → 가중치는 낮게 둔다.

점수 (높을수록 '프롬프트 관점에서 잘 함')
    S = w_r * R* + w_t * T + w_g * G

  R* : 한 질문에 대해 retrieved_chunks[].score 평균을, 이 run 전체 80문항에서 min-max 정규화.
  T  : (context_precision + context_recall) / 2  (judge v2 산출, 이미 0~1)
  G  : (faithfulness + answer_relevance + answer_correctness) / 3

기본 가중치 (합 1.0)
  w_r = 0.15 , w_t = 0.25 , w_g = 0.60

난이도 (높을수록 어려움, 프롬프트를 가르기 어려운 문항)
    D = 1 - S

사용 예::

    uv run python scripts/rank_prompt_difficulty_from_bench_jsonl.py \\
        --jsonl artifacts/logs/runs/bench-e6af1c80.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _min_max(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("artifacts/logs/runs/bench-e6af1c80.jsonl"),
        help="벤치마크 run jsonl 경로",
    )
    parser.add_argument(
        "--w-r",
        type=float,
        default=0.15,
        help="검색 프록시(청크 score 평균) 가중치",
    )
    parser.add_argument(
        "--w-t",
        type=float,
        default=0.25,
        help="전이(context precision/recall 평균) 가중치",
    )
    parser.add_argument(
        "--w-g",
        type=float,
        default=0.60,
        help="생성(3지표 평균) 가중치",
    )
    parser.add_argument(
        "--tsv-out",
        type=Path,
        default=None,
        help="TSV 저장 경로 (미지정 시 stdout만)",
    )
    args = parser.parse_args()

    w = args.w_r + args.w_t + args.w_g
    if abs(w - 1.0) > 1e-6:
        print("가중치 합이 1.0이 아닙니다:", args.w_r, args.w_t, args.w_g, file=sys.stderr)
        sys.exit(1)

    text = args.jsonl.read_text(encoding="utf-8")
    rows_raw: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        qid = str(o.get("question_id", ""))
        rc = o.get("retrieved_chunks") or []
        scores = [float(c.get("score") or 0.0) for c in rc if isinstance(c, dict)]
        r_mean = _mean(scores) if scores else 0.0

        js = o.get("judge_scores") or {}
        cp = float(js.get("context_precision") or 0.0)
        cr = float(js.get("context_recall") or 0.0)
        ff = float(js.get("faithfulness") or 0.0)
        ar = float(js.get("answer_relevance") or 0.0)
        ac = float(js.get("answer_correctness") or 0.0)

        t = (cp + cr) / 2.0
        g = (ff + ar + ac) / 3.0
        rows_raw.append(
            {
                "question_id": qid,
                "r_mean_chunk_score": r_mean,
                "context_precision": cp,
                "context_recall": cr,
                "T": t,
                "faithfulness": ff,
                "answer_relevance": ar,
                "answer_correctness": ac,
                "G": g,
            }
        )

    r_means = [r["r_mean_chunk_score"] for r in rows_raw]
    r_star = _min_max(r_means)

    out_rows: list[dict] = []
    for r, rs in zip(rows_raw, r_star, strict=True):
        s = args.w_r * rs + args.w_t * r["T"] + args.w_g * r["G"]
        d = 1.0 - s
        out_rows.append(
            {
                **r,
                "R_star": rs,
                "S_prompt": round(s, 6),
                "D_prompt": round(d, 6),
            }
        )

    out_rows.sort(key=lambda x: (-x["D_prompt"], x["question_id"]))

    fieldnames = [
        "rank",
        "question_id",
        "D_prompt",
        "S_prompt",
        "R_star",
        "r_mean_chunk_score",
        "T",
        "G",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
    ]

    def emit(handle) -> None:
        wcsv = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        wcsv.writeheader()
        for i, row in enumerate(out_rows, 1):
            wcsv.writerow(
                {
                    "rank": i,
                    "question_id": row["question_id"],
                    "D_prompt": row["D_prompt"],
                    "S_prompt": row["S_prompt"],
                    "R_star": round(row["R_star"], 6),
                    "r_mean_chunk_score": round(row["r_mean_chunk_score"], 6),
                    "T": round(row["T"], 6),
                    "G": round(row["G"], 6),
                    "context_precision": round(row["context_precision"], 6),
                    "context_recall": round(row["context_recall"], 6),
                    "faithfulness": round(row["faithfulness"], 6),
                    "answer_relevance": round(row["answer_relevance"], 6),
                    "answer_correctness": round(row["answer_correctness"], 6),
                }
            )

    if args.tsv_out:
        args.tsv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.tsv_out.open("w", encoding="utf-8", newline="") as f:
            emit(f)
        print(f"wrote {args.tsv_out}", file=sys.stderr)

    emit(sys.stdout)


if __name__ == "__main__":
    main()
