"""Shared helpers for rendering retrieved chunks into LLM context blocks."""

from __future__ import annotations

import math
import re
from numbers import Real

from bidmate_rag.schema import RetrievedChunk

_MISSING_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "undefined",
    "<na>",
    "n/a",
    "na",
    "-",
}

_SOURCE_KEYS = ("사업명", "발주 기관", "파일명")
_DETAIL_KEYS = ("사업 금액", "공개연도", "기관유형", "사업도메인")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRINGS
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False
    value_str = str(value).strip().lower()
    return value_str in _MISSING_STRINGS


def _clean_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _format_won(value: object) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value:,}원"
    if isinstance(value, float):
        number = int(value)
        return f"{number:,}원"
    raw_text = str(value).strip()
    if not raw_text:
        return None
    normalized = raw_text.replace(",", "").replace("원", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized):
        return raw_text
    number = int(float(normalized))
    return f"{number:,}원"


def _format_metadata_line(key: str, value: object) -> str | None:
    if key == "사업 금액":
        formatted = _format_won(value)
    else:
        formatted = _clean_text(value)
    if formatted is None:
        return None
    return f"{key}={formatted}"


def _build_chunk_header(metadata: dict[str, object]) -> str:
    source_values: list[str] = []
    for key in _SOURCE_KEYS:
        value = _clean_text(metadata.get(key))
        if value is not None:
            source_values.append(value)

    lines: list[str] = []
    if source_values:
        lines.append(f"[출처: {' | '.join(source_values)}]")

    for key in _DETAIL_KEYS:
        line = _format_metadata_line(key, metadata.get(key))
        if line is not None:
            lines.append(line)

    return "\n".join(lines)


def _build_chunk_block(chunk: RetrievedChunk) -> str:
    header = _build_chunk_header(chunk.chunk.metadata)
    text = chunk.chunk.text
    if header and text:
        return f"{header}\n{text}"
    if header:
        return header
    return text


def _get_group_key(chunk: RetrievedChunk) -> str:
    metadata = chunk.chunk.metadata
    for key in ("파일명", "사업명", "doc_id"):
        value = metadata.get(key) if key != "doc_id" else chunk.chunk.doc_id
        cleaned = _clean_text(value)
        if cleaned is not None:
            return cleaned
    return chunk.chunk.chunk_id


def _build_doc_label(chunk: RetrievedChunk) -> str:
    metadata = chunk.chunk.metadata
    parts: list[str] = []
    for key in _SOURCE_KEYS:
        value = _clean_text(metadata.get(key))
        if value is not None:
            parts.append(value)
    return " | ".join(parts) if parts else chunk.chunk.doc_id


def _build_grouped_chunk_body(chunk: RetrievedChunk) -> str:
    lines: list[str] = []

    section = _clean_text(chunk.chunk.section)
    if section:
        lines.append(f"섹션={section}")

    for key in _DETAIL_KEYS:
        line = _format_metadata_line(key, chunk.chunk.metadata.get(key))
        if line is not None:
            lines.append(line)

    text = chunk.chunk.text
    if text:
        lines.append(text)

    return "\n".join(lines)


def _render_grouped_context(
    chunks: list[RetrievedChunk], used_indices: list[int], *, with_citation_numbers: bool
) -> str:
    if not used_indices:
        return ""

    groups: list[tuple[str, RetrievedChunk, list[tuple[int, RetrievedChunk]]]] = []
    group_lookup: dict[str, int] = {}

    for citation_idx, chunk_idx in enumerate(used_indices, 1):
        chunk = chunks[chunk_idx]
        group_key = _get_group_key(chunk)
        if group_key not in group_lookup:
            group_lookup[group_key] = len(groups)
            groups.append((group_key, chunk, []))
        groups[group_lookup[group_key]][2].append((citation_idx, chunk))

    rendered_groups: list[str] = []
    for _group_key, first_chunk, grouped_items in groups:
        header = f"[문서: {_build_doc_label(first_chunk)}]"
        blocks = [header]
        for citation_idx, chunk in grouped_items:
            body = _build_grouped_chunk_body(chunk)
            if with_citation_numbers:
                blocks.append(f"[{citation_idx}] {body}")
            else:
                blocks.append(body)
        rendered_groups.append("\n\n".join(blocks))

    return "\n\n---\n\n".join(rendered_groups)


def build_context_block(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Render retrieved chunks into a metadata-aware context block.

    백워드 호환용. LLM 경로에서는 `build_numbered_context_block`을 사용해
    `[n]` 인용 번호와 LLM이 실제로 본 청크 인덱스를 함께 받는 것을 권장.
    """
    if max_chars <= 0:
        return ""

    parts: list[str] = []
    total_chars = 0
    separator = "\n\n---\n\n"

    for chunk in chunks:
        block = _build_chunk_block(chunk)
        if not block:
            continue
        candidate = block if not parts else f"{separator}{block}"
        if total_chars + len(candidate) > max_chars:
            break
        parts.append(block)
        total_chars += len(candidate)

    return separator.join(parts)


def build_numbered_context_block(
    chunks: list[RetrievedChunk],
    max_chars: int = 8000,
    *,
    with_citation_numbers: bool = True,
) -> tuple[str, list[int]]:
    """번호가 붙은 컨텍스트 블록과 LLM이 실제로 본 청크 인덱스를 반환한다.

    Args:
        chunks: 검색된 청크 (이미 score desc 정렬 가정).
        max_chars: 컨텍스트 총 글자 수 상한. 초과 시 뒤쪽 청크는 **통째로** 탈락한다.
        with_citation_numbers: True면 각 청크 앞에 `[1]`, `[2]`... prefix를 붙인다.
            프론트/LLM 인용 번호와 1:1 매칭시키기 위한 것. 평가/로그 경로에서는
            번호가 불필요하면 False로 끈다.

    Returns:
        (context_str, used_indices):
          - context_str: 최종 프롬프트에 들어갈 문자열
          - used_indices: 실제로 컨텍스트에 포함된 청크의 원본 인덱스 (0-based)
            예) 5개 청크 중 첫 3개만 포함 → [0, 1, 2]. Citation 생성 시 이
            인덱스를 기준으로 필터해야 본문 `[n]`과 카드가 일치한다.
    """
    if max_chars <= 0:
        return "", []

    used_indices: list[int] = []
    total_chars = 0
    separator = "\n\n---\n\n"

    for idx, chunk in enumerate(chunks):
        block = _build_chunk_block(chunk)
        if not block:
            continue
        if with_citation_numbers:
            # 1-indexed 인용 번호를 청크 앞에 박는다. LLM이 답변에서 동일 번호로 참조.
            citation_idx = len(used_indices) + 1
            block = f"[{citation_idx}] {block}"
        candidate = block if not used_indices else f"{separator}{block}"
        if total_chars + len(candidate) > max_chars:
            break
        used_indices.append(idx)
        total_chars += len(candidate)

    return _render_grouped_context(
        chunks, used_indices, with_citation_numbers=with_citation_numbers
    ), used_indices
