import os
import random
import shlex
import subprocess
import zlib

import olefile
from hwp_hwpx_parser import Reader

# 1. 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hwp_dir = os.path.join(base_dir, "data", "raw")
output_base = os.path.join(base_dir, "data", "test")
os.makedirs(os.path.join(output_base, "hwp-hwpx-parser", "hwp"), exist_ok=True)
os.makedirs(os.path.join(output_base, "kordoc", "hwp"), exist_ok=True)
os.makedirs(os.path.join(output_base, "olefile", "hwp"), exist_ok=True)


# 2. 랜덤 20개 추출
all_files = [f for f in os.listdir(hwp_dir) if f.lower().endswith(".hwp")]
sample_files = random.sample(all_files, min(20, len(all_files)))

print(f"🚀 {len(sample_files)}개 파일 분석 시작 (리스트 형변환 픽스 적용)")

for idx, file_name in enumerate(sample_files, 1):
    path = os.path.join(hwp_dir, file_name)
    print(f"[{idx}/20] 처리 중: {file_name}")

    # A. hwp-hwpx-parser (데이터 타입 강제 변환 적용)
    try:
        with Reader(path) as r:
            # r.text가 리스트일 경우 join으로 묶고, 아니면 문자열로 캐스팅
            text_part = "\n".join(r.text) if isinstance(r.text, list) else str(r.text)

            # 마크다운 표 데이터 추출 및 형변환
            table_raw = r.get_tables_as_markdown()
            table_part = "\n\n".join(table_raw) if isinstance(table_raw, list) else str(table_raw)

            content = text_part + "\n\n--- [표 데이터] ---\n" + table_part

            with open(
                os.path.join(output_base, "hwp-hwpx-parser", "hwp", f"{file_name}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(content)
    except Exception as e:
        with open(
            os.path.join(output_base, "hwp-hwpx-parser", "hwp", f"{file_name}_ERROR.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"에러: {str(e)}")

    # B. kordoc 실행
    try:
        input_path = os.path.abspath(path)
        out_path = os.path.abspath(
            os.path.join(output_base, "kordoc", "hwp", f"{file_name}_kordoc.md")
        )
        kordoc_cmd = (
            'export NVM_DIR="$HOME/.nvm" && '
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
            "nvm use 20 >/dev/null && "
            f"npx kordoc {shlex.quote(input_path)} -o {shlex.quote(out_path)}"
        )
        result = subprocess.run(
            ["bash", "-lc", kordoc_cmd], capture_output=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "kordoc 실행 실패")
    except Exception as e:
        print(f"❌ kordoc 에러: {e}")

    # C. olefile (원천 텍스트 추출)
    try:
        f_ole = olefile.OleFileIO(path)
        encoded_text = f_ole.openstream("BodyText/Section0").read()
        raw_text = zlib.decompress(encoded_text, -15).decode("utf-16", errors="ignore")
        with open(
            os.path.join(output_base, "olefile", "hwp", f"{file_name}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(raw_text)
    except Exception as e:
        with open(
            os.path.join(output_base, "olefile", "hwp", f"{file_name}_ERROR.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"에러: {str(e)}")

print(
    f"\n파싱 완료. '{output_base}/hwp-hwpx-parser/hwp' 폴더에서 파일 용량이 정상(1KB 이상)인지 확인하십시오."
)
