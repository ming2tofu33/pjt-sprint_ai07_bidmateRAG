"""kordoc subprocess helper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DEFAULT_KORDOC_BIN = shutil.which("kordoc") or Path.home() / ".npm-global" / "bin" / "kordoc"


def parse_with_kordoc(
    file_path: str | Path,
    kordoc_bin: str | Path | None = None,
    timeout: int = 120,
) -> dict:
    """kordoc CLI로 문서 파일을 파싱하여 텍스트를 추출

    Args:
        file_path: 파싱할 문서 파일 경로.
        kordoc_bin: kordoc 바이너리 경로.
        timeout: 프로세스 타임아웃(초).

    Returns:
        파싱 결과 딕셔너리 (파일명, 파서, 텍스트, 글자수, 성공, 에러).
    """
    path = Path(file_path)
    binary = Path(kordoc_bin) if kordoc_bin else DEFAULT_KORDOC_BIN
    try:
        result = subprocess.run(
            [str(binary), str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        text = result.stdout.strip()
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": text,
            "글자수": len(text),
            "성공": bool(text),
            "에러": None if text else result.stderr.strip() or "empty-output",
        }
    except subprocess.TimeoutExpired:
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": "timeout",
        }
    except Exception as exc:  # pragma: no cover - system dependent
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": str(exc),
        }
