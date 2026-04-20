"""PDF loading logic."""

from __future__ import annotations

from pathlib import Path

from bidmate_rag.loaders.kordoc_loader import parse_with_kordoc


def parse_pdf(file_path: str | Path, kordoc_bin: str | Path | None = None) -> dict:
    """PDF 파일을 파싱하여 텍스트를 추출

    Args:
        file_path: PDF 파일 경로.
        kordoc_bin: kordoc 바이너리 경로.

    Returns:
        파싱 결과 딕셔너리 (파일명, 파서, 텍스트, 글자수, 성공, 에러).
    """
    result = parse_with_kordoc(file_path, kordoc_bin=kordoc_bin)
    if result["성공"]:
        return result
    path = Path(file_path)
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return {
            "파일명": path.name,
            "파서": "pdfplumber",
            "텍스트": text,
            "글자수": len(text),
            "성공": bool(text),
            "에러": None,
        }
    except Exception as exc:  # pragma: no cover - fallback path
        return {
            "파일명": path.name,
            "파서": "pdfplumber",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": str(exc),
        }
