import itertools
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import components  # noqa: E402  # components.py (동일 디렉터리)

_REPO = _HERE.parent.parent.parent.parent
output_dir = str(_REPO / "configs" / "experiments" / "generated2")
os.makedirs(output_dir, exist_ok=True)

# 부품 딕셔너리 매핑
p_dict = components.PERSONA
c_dict = components.CONSTRAINTS
f_dict = components.FORMAT
s_dict = components.FEW_SHOT
t_dict = components.COT_STRATEGY

combinations = list(itertools.product(p_dict.keys(), c_dict.keys(), f_dict.keys(), s_dict.keys(), t_dict.keys()))
print(f"총 {len(combinations)}개의 프롬프트 조합 생성을 시작합니다...")

def format_block_scalar(text, indent=2):
    """문자열을 YAML의 깔끔한 블록(|) 형태로 강제 변환 (따옴표 제거)"""
    if not text:
        return ' ""\n'
    lines = str(text).strip().split('\n')
    indented_lines = [' ' * indent + line if line.strip() else '' for line in lines]
    return " |\n" + '\n'.join(indented_lines) + "\n"

def format_list_item(text):
    """문자열을 YAML 리스트(-) 내부의 블록(|) 형태로 강제 변환"""
    if not text:
        return ""
    lines = str(text).strip().split('\n')
    indented_lines = [' ' * 4 + line if line.strip() else '' for line in lines]
    return "  - |\n" + '\n'.join(indented_lines) + "\n"

for comb in combinations:
    p_key, c_key, f_key, s_key, t_key = comb
    
    part_p = p_dict.get(p_key, {})
    part_c = c_dict.get(c_key, {})
    part_f = f_dict.get(f_key, {})
    part_s = s_dict.get(s_key, {})
    part_t = t_dict.get(t_key, {})
    
    experiment_id = f"prompt_{p_key}_{c_key}_{f_key}_{s_key}_{t_key}"
    description = f"{p_key} + {c_key} + {f_key} + {s_key} + {t_key} 조합"
    
    control_score = (int(part_p.get('control_score', 0)) + 
                     int(part_c.get('control_score', 0)) + 
                     int(part_f.get('control_score', 0)) + 
                     int(part_s.get('control_score', 0)) + 
                     int(part_t.get('control_score', 0)))
    
    is_english_cot = any([
        bool(part_p.get('is_english_cot', False)),
        bool(part_c.get('is_english_cot', False)),
        bool(part_f.get('is_english_cot', False)),
        bool(part_s.get('is_english_cot', False)),
        bool(part_t.get('is_english_cot', False))
    ])
    is_eng_str = "true" if is_english_cot else "false"
    
    char_count = sum([
        len(str(part_p.get('text', ''))),
        len(str(part_c.get('text', ''))),
        len(str(part_f.get('text', ''))),
        len(str(part_s.get('text', ''))),
        len(str(part_t.get('text', '')))
    ])

    # 어떤 기호가 들어가도 절대 깨지지 않는 완벽한 구조
    yaml_content = f"""00_metadata:
  experiment_id: "{experiment_id}"
  name: "{experiment_id}"
  description: "{description}"
  is_english_cot: {is_eng_str}
  char_count: {char_count}
  control_score: {control_score}

persona:{format_block_scalar(part_p.get('text', ''), 2)}
constraints:
{format_list_item(part_c.get('text', ''))}output_format:
{format_list_item(part_f.get('text', ''))}few_shot_examples:
{format_list_item(part_s.get('text', ''))}thought_trigger:
{format_list_item(part_t.get('text', ''))}"""

    file_path = os.path.join(output_dir, f"{experiment_id}.yaml")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

print(f"완료: {len(combinations)}개의 YAML 파일이 생성되었습니다.")