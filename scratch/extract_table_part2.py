import glob
import json
import os
import subprocess
import pandas as pd
from pathlib import Path

# 출력 경로
output_file = Path(r"c:\Users\home\Desktop\보고서\RAW_DATA_PART2.md")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Parquet 파일들 모두 로드
parquet_files = glob.glob("artifacts/logs/benchmarks/*.parquet")
df_list = []
for p in parquet_files:
    try:
        df_list.append(pd.read_parquet(p))
    except Exception as e:
        print(f"Error reading {p}: {e}")

if not df_list:
    df_metrics = pd.DataFrame()
else:
    df_metrics = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['run_id'])

# meta.json 파일에서 메타정보 로드
runs_dir = "artifacts/logs/runs"
meta_files = glob.glob(f"{runs_dir}/*.meta.json")

def get_git_info(commit_hash):
    if not commit_hash:
        return "기록 부재", "기록 부재"
    try:
        res = subprocess.run(["git", "log", "-1", "--pretty=format:%an|%s", commit_hash], 
                             capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        if "|" in out:
            author, msg = out.split("|", 1)
            return author.strip(), msg.strip()
        else:
            return "기록 부재", "기록 부재"
    except Exception:
        return "기록 부재", "기록 부재"

records = []
for meta_path in meta_files:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    run_id = meta.get("run_id", "기록 부재")
    date = meta.get("timestamp_kst", "기록 부재")
    
    # configs
    config = meta.get("config_snapshot", {})
    exp_conf = config.get("experiment", {})
    provider_conf = config.get("provider", {})
    
    chunk_size = exp_conf.get("chunk_size", "기록 부재")
    top_k = meta.get("actual_top_k", exp_conf.get("retrieval_top_k", "기록 부재"))
    model_name = provider_conf.get("model", "기록 부재")
    
    # git config
    git_info = meta.get("git", {})
    commit_hash = git_info.get("commit", None)
    author, commit_msg = get_git_info(commit_hash)
    
    # Find metrics
    hit_rate = "기록 부재"
    mrr = "기록 부재"
    latency = meta.get("avg_latency_ms", "기록 부재")
    cost = meta.get("total_cost_usd", "기록 부재")
    
    if not df_metrics.empty and run_id in df_metrics['run_id'].values:
        row = df_metrics[df_metrics['run_id'] == run_id].iloc[0]
        if 'hit_rate@5' in row and pd.notna(row['hit_rate@5']):
            hit_rate = f"{float(row['hit_rate@5']):.3f}"
        if 'mrr' in row and pd.notna(row['mrr']):
            mrr = f"{float(row['mrr']):.3f}"
        if latency == "기록 부재" and 'avg_latency_ms' in row and pd.notna(row['avg_latency_ms']):
            latency = f"{float(row['avg_latency_ms']):.3f}"
        if cost == "기록 부재" and 'total_cost_usd' in row and pd.notna(row['total_cost_usd']):
            cost = f"{float(row['total_cost_usd']):.3f}"

    if isinstance(latency, (float, int)):
        latency = f"{latency:.3f}"
    if isinstance(cost, (float, int)):
        cost = f"{cost:.3f}"
        
    records.append({
        "Run ID": run_id,
        "날짜": date,
        "담당자": author,
        "Chunk Size": chunk_size,
        "Top-k": top_k,
        "모델명": model_name,
        "Hit Rate@5": hit_rate,
        "MRR": mrr,
        "Latency(ms)": latency,
        "Cost(usd)": cost,
        "커밋 메시지 (인과추론용)": commit_msg,
        "Hash": git_info.get("commit_short", "기록 부재")
    })

# Sort records by date
records.sort(key=lambda x: x["날짜"])

# Write Markdown Table
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# RAW_DATA_PART2: 성능 평가 및 비교 분석 실험 로그\n\n")
    f.write("> **이 표는 artifacts/ 폴더 내 모든 parquet 배치 실행 결과와 meta.json, 그리고 git log를 교차 대조하여 작성되었습니다.**\n\n")
    
    # Table Header
    headers = ["Run ID", "날짜", "담당자", "Chunk Size", "Top-k", "모델명", "Hit Rate@5", "MRR", "Latency(ms)", "Cost(usd)", "커밋 메시지 (인과추론용)"]
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    
    for r in records:
        f.write("| " + " | ".join([str(r[h]) for h in headers]) + " |\n")
    
    # 3. Add causality analysis base
    f.write("\n## 3. 인과관계 추론 분석 (분석용 초안)\n\n")
    f.write("위 표의 `커밋 메시지`와 `Hit Rate@5/MRR` 점수 변화를 대조하여, 어떠한 코드 수정(예: BM25 하이브리드 검색, Chunk Size 변경 등)이 성능 지표의 급격한 변동에 영향을 미쳤는지 파악할 수 있는 기초 자료입니다.\n")
    
print("Table successfully written.")
