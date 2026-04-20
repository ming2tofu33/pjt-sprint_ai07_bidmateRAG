import glob
import json
import os
import random
from pathlib import Path

output_file = Path("c:/Users/home/Desktop/보고서/RAW_DATA_PART2.md")

# Load jsonl files for runs
run_files = glob.glob("artifacts/logs/runs/*.jsonl")

good_cases = []
bad_answer_cases = []
failed_cases = []
errors = []

for run_file in run_files:
    try:
        with open(run_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    judge = data.get("judge_scores", {})
                    answer_correctness = judge.get("answer_correctness", 0.0)
                    faithfulness = judge.get("faithfulness", 0.0)
                    relevance = judge.get("answer_relevance", 0.0)
                    
                    # Error parsing or edge case where it couldn't retrieve
                    if "에러" in str(data.get("error", "")) or len(data.get("retrieved_chunks", [])) == 0:
                        failed_cases.append(data)
                    elif answer_correctness > 0.8:
                        good_cases.append(data)
                    elif (answer_correctness < 0.5) and (len(data.get("retrieved_chunks", [])) > 0):
                        bad_answer_cases.append(data)
                        
                    if data.get("error"):
                         errors.append(data.get("error"))
                except BaseException as e:
                    pass
    except BaseException:
        pass

# Select 5 random for each category safely
selected_good = random.sample(good_cases, min(5, len(good_cases)))
selected_bad = random.sample(bad_answer_cases, min(5, len(bad_answer_cases)))
selected_failed = random.sample(failed_cases, min(5, len(failed_cases)))

def format_case(case_data):
    run_id = case_data.get("run_id", "Unknown")
    question_id = case_data.get("question_id", "Unknown")
    question = case_data.get("question", "Unknown")
    
    chunks = case_data.get("retrieved_chunks", [])
    chunks_text = "\n\n".join([f"[DocID: {c.get('chunk', {}).get('doc_id')}] \n{c.get('chunk', {}).get('text')}" for c in chunks])
    
    answer = case_data.get("answer", "Unknown")
    judge = case_data.get("judge_scores", {})
    reasoning = judge.get("reasoning", "Unknown")
    
    return f"""### [Run ID: {run_id}] 질의: {question_id}
**질문 원문**
```text
{question}
```

**검색된 컨텍스트(청크) 원문 전문**
```text
{chunks_text}
```

**LLM 답변 원문**
```text
{answer}
```

**판정 결과 및 근거**
```text
Answer Correctness: {judge.get('answer_correctness', 0.0)}
Faithfulness: {judge.get('faithfulness', 0.0)}
Answer Relevance: {judge.get('answer_relevance', 0.0)}

Reasoning:
{reasoning}
```

"""

with open(output_file, "a", encoding="utf-8") as f:
    f.write("\n## 4. 심층 사례 분석 (Case Study Raw Data)\n\n")
    
    f.write("### 4.1 성능 우수 (성공 요인 분석용)\n\n")
    for case in selected_good:
        f.write(format_case(case))
        
    f.write("### 4.2 지표는 높으나 답변 품질이 아쉬운 사례\n\n")
    for case in selected_bad:
        f.write(format_case(case))
        
    f.write("### 4.3 에지 케이스 및 검색/생성 완전 실패 사례\n\n")
    for case in selected_failed:
        f.write(format_case(case))

    f.write("\n## 5. Troubleshooting Raw Logs\n\n")
    f.write("실험 중 발생한 주요 에러 로그 및 Traceback 전문입니다.\n\n")
    if not errors:
        f.write("> **기록 부재 (주목할 만한 시스템 예외/OOM 발생 기록 안됨)**\n")
    else:
        for err in set(errors)[:5]:
            f.write(f"```python\n{err}\n```\n")
    
print("Successfully appended")
