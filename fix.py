import re
import os

files = [
    r'C:\Users\home\Desktop\보고서\REPORT_V09_Part1.md',
    r'C:\Users\home\Desktop\보고서\REPORT_V09_Part2.md',
    r'C:\Users\home\Desktop\보고서\REPORT_V09_Part3.md'
]

def format_deleted_line(line):
    if line.strip() == '': return line
    if re.match(r'^[\|\-\s:]+$', line): return line
    
    # Headers
    h_match = re.match(r'^(#+)\s+(.+)$', line)
    if h_match:
        return f'{h_match.group(1)} <del style="color:red;">{h_match.group(2)}</del>'
    
    # List items
    l_match = re.match(r'^(\s*[\-\*\+]\s+)(.+)$', line)
    if l_match:
        return f'{l_match.group(1)}<del style="color:red;">{l_match.group(2)}</del>'
        
    # Tables
    if line.strip().startswith('|') and line.strip().endswith('|'):
        parts = line.split('|')
        new_parts = []
        for p in parts:
            if p.strip() == '':
                new_parts.append(p)
            else:
                new_parts.append(f' <del style="color:red;">{p.strip()}</del> ')
        return '|'.join(new_parts)
    
    return f'<del style="color:red;">{line}</del>'

for path in files:
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove stars
    content = content.replace('★ ', '')
    
    # 2. Process <del> blocks line by line
    def replacer(match):
        block = match.group(1)
        lines = block.split('\n')
        return '\n'.join(format_deleted_line(l) for l in lines)
    
    content = re.sub(r'<del style="color:red;">\s*(.*?)\s*</del>', replacer, content, flags=re.DOTALL)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Also copy to 보고서 수정본
    copy_path = os.path.join(r'C:\Users\home\Desktop\보고서\보고서 수정본', os.path.basename(path))
    if os.path.exists(copy_path):
        with open(copy_path, 'w', encoding='utf-8') as f:
            f.write(content)
print("Done")
