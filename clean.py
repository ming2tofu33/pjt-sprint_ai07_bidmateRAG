import os
import re

files_to_clean = [
    r'REPORT_V09_Part1.md',
    r'REPORT_V09_Part2.md',
    r'REPORT_V09_Part3.md',
    r'목차1.md',
    r'보고서 수정본\목차1.md'
]

base_dir = r'C:\Users\home\Desktop\보고서'

for fname in files_to_clean:
    filepath = os.path.join(base_dir, fname)
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    in_deleted_table = False
    
    for line in lines:
        stripped = line.strip()
        
        # Drop lines that contain the red strike-through tag
        if '<del style="color:red;">' in line:
            # We are dropping a table row, so mark that we are in a deleted table
            if stripped.startswith('|'):
                in_deleted_table = True
            continue
            
        # Drop stray closing block tags
        if stripped == '</del>':
            continue
            
        # Drop floating table separators linked to a deleted table
        if re.match(r'^[\|\-\s:]+$', stripped) and len(stripped) > 0 and in_deleted_table:
            continue
            
        # If we encounter a line that isn't part of a table, reset the flag
        if not stripped.startswith('|'):
            in_deleted_table = False
            
        # Keep the content of blue spans, but remove the span wrappers
        clean_line = line.replace('<span style="color:blue;">', '').replace('</span>', '')
        
        # In case there were any inline <del> tags without color:red for some reason
        # clean_line = re.sub(r'<del.*?>', '', clean_line).replace('</del>', '')
        
        new_lines.append(clean_line)
        
    # Re-assemble text
    content = "".join(new_lines)
    
    # Clean up multiple consecutive blank lines caused by dropping blocks
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"Cleanup complete for {len(files_to_clean)} files!")
