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
_STOPWORDS = {
    "사업",
    "시스템",
    "구축",
    "용역",
    "문서",
    "질문",
    "관련",
    "해당",
    "기준",
    "무엇",
    "얼마",
    "언제",
    "어떻게",
    "대한",
    "대해",
    "각각",
    "모두",
    "비교",
    "정보",
    "정리",
    "알려줘",
    "알려",
    "있나",
    "있나요",
    "입니까",
    "무엇입니까",
    "가능합니까",
}
_FOCUS_TERM_ALIASES = (
    (
        ("예산", "금액", "가격", "비용"),
        ("예산", "사업예산", "사업 금액", "사업금액", "소요예산", "추정가격", "금액"),
    ),
    (
        ("지식재산권", "저작권", "소유", "귀속"),
        ("지식재산권", "저작권", "귀속", "단독 소유", "공동 소유", "발주처 단독", "공동활용"),
    ),
    (
        ("심의", "점검표", "법제도"),
        ("과업심의위원회", "과업변경심의위원회", "과업내용 확정 심의", "법제도 준수 여부 점검표", "법제도"),
    ),
    (
        ("기간", "기한", "종료일", "착수일", "마감"),
        ("사업기간", "과업기간", "기간", "종료일", "착수일", "기한", "마감"),
    ),
)


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


def _tokenize_text(value: str) -> list[str]:
    return re.findall(r"[0-9a-z가-힣]+", value.lower())


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", value.lower())


def _extract_focus_terms(question: str | None) -> list[str]:
    if not question:
        return []

    normalized_question = " ".join(str(question).split()).lower()
    terms: list[str] = []

    for triggers, aliases in _FOCUS_TERM_ALIASES:
        if any(trigger in normalized_question for trigger in triggers):
            terms.extend(aliases)

    for token in _tokenize_text(normalized_question):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        terms.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = term.strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _question_mentions_source(question: str | None, metadata: dict[str, object]) -> bool:
    if not question:
        return False

    normalized_question = _normalize_for_match(question)
    question_tokens = {
        token for token in _tokenize_text(question) if len(token) >= 2 and token not in _STOPWORDS
    }

    for key in _SOURCE_KEYS:
        value = _clean_text(metadata.get(key))
        if value is None:
            continue

        normalized_value = _normalize_for_match(value)
        if normalized_value and len(normalized_value) >= 4 and normalized_value in normalized_question:
            return True

        value_tokens = {
            token for token in _tokenize_text(value) if len(token) >= 2 and token not in _STOPWORDS
        }
        if len(question_tokens & value_tokens) >= 2:
            return True

    return False


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

    text = chunk.chunk.text
    if text:
        lines.append(text)

    return "\n".join(lines)


def _chunk_focus_score(chunk: RetrievedChunk, focus_terms: list[str]) -> int:
    if not focus_terms:
        return 0

    metadata = chunk.chunk.metadata or {}
    section_text = (_clean_text(chunk.chunk.section) or "").lower()
    header_text = " ".join(
        filter(
            None,
            [
                *(_clean_text(metadata.get(key)) for key in _SOURCE_KEYS),
                *(_clean_text(metadata.get(key)) for key in _DETAIL_KEYS),
            ],
        )
    ).lower()
    body_text = (chunk.chunk.text or "").lower()

    score = 0
    for term in focus_terms:
        if term in section_text:
            score += 4
        if term in header_text:
            score += 3
        if term in body_text:
            score += 1
    return score


def _order_context_indices(chunks: list[RetrievedChunk], question: str | None) -> list[int]:
    focus_terms = _extract_focus_terms(question)
    if not focus_terms:
        return list(range(len(chunks)))

    group_order: list[str] = []
    grouped_indices: dict[str, list[int]] = {}

    for idx, chunk in enumerate(chunks):
        group_key = _get_group_key(chunk)
        if group_key not in grouped_indices:
            grouped_indices[group_key] = []
            group_order.append(group_key)
        grouped_indices[group_key].append(idx)

    sorted_groups: list[list[int]] = []
    for group_key in group_order:
        indices = grouped_indices[group_key]
        sorted_groups.append(
            sorted(
                indices,
                key=lambda idx: (
                    -_chunk_focus_score(chunks[idx], focus_terms),
                    -float(chunks[idx].score),
                    idx,
                ),
            )
        )

    ordered_indices: list[int] = []
    max_group_len = max((len(indices) for indices in sorted_groups), default=0)
    for offset in range(max_group_len):
        for indices in sorted_groups:
            if offset < len(indices):
                ordered_indices.append(indices[offset])
    return ordered_indices


def _render_grouped_context(
    chunks: list[RetrievedChunk],
    used_indices: list[int],
    *,
    with_citation_numbers: bool,
    question: str | None = None,
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
        header_lines = [f"[문서: {_build_doc_label(first_chunk)}]"]
        if _question_mentions_source(question, first_chunk.chunk.metadata):
            header_lines.append("질문대상=예")
        for key in _DETAIL_KEYS:
            line = _format_metadata_line(key, first_chunk.chunk.metadata.get(key))
            if line is not None:
                header_lines.append(line)
        blocks = ["\n".join(header_lines)]
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
    question: str | None = None,
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

    ordered_indices = _order_context_indices(chunks, question)

    for idx in ordered_indices:
        chunk = chunks[idx]
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
        chunks, used_indices, with_citation_numbers=with_citation_numbers, question=question
    ), used_indices
