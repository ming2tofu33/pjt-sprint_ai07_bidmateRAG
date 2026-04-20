import os
import shlex
import subprocess

import pdfplumber
from markitdown import MarkItDown

# 1. 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_dir = os.path.join(base_dir, "data", "raw")
output_base = os.path.join(base_dir, "data", "test")

# 출력 폴더 생성 (파서별 / 파일타입별)
for parser in ["plumber", "markitdown", "kordoc"]:
    for ext in ["pdf", "hwp"]:
        os.makedirs(os.path.join(output_base, parser, ext), exist_ok=True)

# 2. 파일 리스트 확보 (PDF + HWP)
if not os.path.exists(input_dir):
    print(f"❌ 경로를 찾을 수 없습니다: {input_dir}")
else:
    target_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".pdf", ".hwp"))][:4]
    print(f"🚀 총 {len(target_files)}개의 파일 파싱을 시작합니다.")


def parse_with_pdfplumber(file_path):
    """pdfplumber: 표(Table) 추출에 강점 (PDF 전용)"""
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            text_content.append(f"--- Page {i + 1} ---\n{page_text}")

            # 표 데이터 추출
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                text_content.append(f"\n[Table {table_idx + 1}]\n")
                for row in table:
                    text_content.append(str(row))
    return "\n".join(text_content)


def parse_with_markitdown(file_path):
    """MarkItDown: 다양한 포맷을 LLM용 마크다운으로 변환"""
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


def parse_with_kordoc(file_path, output_path):
    """kordoc: Node.js 기반 CLI 파서, PDF/HWP → 마크다운 변환"""
    input_path = os.path.abspath(file_path)
    kordoc_cmd = (
        'export NVM_DIR="$HOME/.nvm" && '
        '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
        "nvm use 20 >/dev/null && "
        f"npx kordoc {shlex.quote(input_path)} -o {shlex.quote(os.path.abspath(output_path))}"
    )
    result = subprocess.run(
        ["bash", "-lc", kordoc_cmd], capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "kordoc 실행 실패")


# 3. 실행 및 저장
for fname in target_files:
    file_path = os.path.join(input_dir, fname)
    file_name = os.path.splitext(fname)[0]
    ext = os.path.splitext(fname)[1].lstrip(".").lower()  # 'pdf' or 'hwp'

    print(f"📦 처리 중: {fname}...")

    # A. pdfplumber 실행 (PDF만)
    if ext == "pdf":
        try:
            plumber_result = parse_with_pdfplumber(file_path)
            out_path = os.path.join(output_base, "plumber", ext, f"{file_name}_plumber.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(plumber_result)
        except Exception as e:
            print(f"❌ pdfplumber 에러: {e}")

    # B. MarkItDown 실행
    try:
        mid_result = parse_with_markitdown(file_path)
        out_path = os.path.join(output_base, "markitdown", ext, f"{file_name}_mid.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(mid_result)
    except Exception as e:
        print(f"❌ MarkItDown 에러: {e}")

    # C. kordoc 실행
    try:
        out_path = os.path.join(output_base, "kordoc", ext, f"{file_name}_kordoc.md")
        parse_with_kordoc(file_path, out_path)
    except Exception as e:
        print(f"❌ kordoc 에러: {e}")

print(f"\n✅ 모든 작업이 완료되었습니다. '{output_base}' 폴더를 확인하세요.")
