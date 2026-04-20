import json
import glob
import os

files = glob.glob('artifacts/logs/runs/*.jsonl')
files = [f for f in files if os.path.getsize(f) > 5000000]
latest_file = max(files, key=os.path.getmtime)

cases = []
with open(latest_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        scores = data.get('judge_scores', {})
        if not scores:
            continue
            
        f_score = scores.get('faithfulness', 0)
        ar_score = scores.get('answer_relevance', 0)
        cp_score = scores.get('context_precision', 0)
        cr_score = scores.get('context_recall', 0)
        ac_score = scores.get('answer_correctness', 0)
        
        total_score = f_score + ar_score + cp_score + cr_score + ac_score
        
        chunks = data.get('retrieved_chunks', [])
        if isinstance(chunks, list):
            chunks_str = "\n\n".join([str(c) for c in chunks])
        else:
            chunks_str = str(chunks)
            
        context_str = data.get('context', chunks_str)
            
        case = {
            'id': data.get('question_id', 'N/A'),
            'question': data.get('question', 'N/A'),
            'context': context_str,
            'answer': data.get('answer', 'N/A'),
            'reasoning': scores.get('reasoning', 'N/A'),
            'total_score': total_score,
            'scores': f"F:{f_score:.2f}, AR:{ar_score:.2f}, CP:{cp_score:.2f}, CR:{cr_score:.2f}, AC:{ac_score:.2f}"
        }
        cases.append(case)

cases.sort(key=lambda x: x['total_score'], reverse=True)
top_5 = cases[:5]
bottom_5 = cases[-5:]

out_file = r"c:\Users\home\Desktop\보고서\RAW_DATA_PART2.md"
with open(out_file, 'a', encoding='utf-8') as f:
    f.write("\n\n## 핵심 사례 10선 추출 (Case Study)\n\n")
    
    f.write("### [가장 완벽한 성공 사례 5개]\n\n")
    for i, c in enumerate(top_5, 1):
        f.write(f"#### 성공 사례 {i}\n")
        f.write(f"- **질문 ID**: {c['id']}\n")
        f.write(f"- **질문 원문**: {c['question']}\n")
        f.write(f"- **검색된 컨텍스트(청크) 전문**:\n```text\n{c['context']}\n```\n")
        f.write(f"- **LLM 답변 전문**:\n```text\n{c['answer']}\n```\n")
        f.write(f"- **판정 근거**: {c['reasoning']} (세부점수: {c['scores']})\n\n")
        
    f.write("### [처참하게 실패한 사례 5개]\n\n")
    for i, c in enumerate(bottom_5, 1):
        f.write(f"#### 실패 사례 {i}\n")
        f.write(f"- **질문 ID**: {c['id']}\n")
        f.write(f"- **질문 원문**: {c['question']}\n")
        f.write(f"- **검색된 컨텍스트(청크) 전문**:\n```text\n{c['context']}\n```\n")
        f.write(f"- **LLM 답변 전문**:\n```text\n{c['answer']}\n```\n")
        f.write(f"- **판정 근거**: {c['reasoning']} (세부점수: {c['scores']})\n\n")

print("Finished writing to", out_file)
