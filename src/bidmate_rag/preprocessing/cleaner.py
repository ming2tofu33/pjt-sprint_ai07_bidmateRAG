"""Text cleaning helpers reproduced from the notebook baseline."""

from __future__ import annotations

import re

BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TABLE_RE = re.compile(r"<table>.*?</table>", re.IGNORECASE | re.DOTALL)
TABLE_DELIMITER_RE = re.compile(r"^\|[\s\-:|]+\|$")
ROMAN_MARKER_RE = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?$")
NUMERIC_MARKER_RE = re.compile(r"^\d+(?:-\d+)?\.?$")
TOC_ENTRY_RE = re.compile(
    r"^(?:"
    r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*.+?\d+|"
    r"\d+(?:-\d+)*\.?\s+.+?\d+|"
    r"별지.*\d+"
    r")$"
)


def clean_br_tags(text: str) -> str:
    """Replace HTML line breaks with real newlines."""

    return BR_TAG_RE.sub("\n", text)


def _table_plain_text(table: str) -> str:
    plain = re.sub(r"</?(?:table|tr|td|th)[^>]*>", " ", table, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", plain).strip()


def remove_empty_html_tables(text: str) -> str:
    """Drop HTML tables that do not contain any meaningful text."""

    def _replace(match: re.Match[str]) -> str:
        plain = _table_plain_text(match.group(0))
        return "" if not plain else match.group(0)

    return HTML_TABLE_RE.sub(_replace, text)


def remove_toc_html_tables(text: str) -> str:
    """Drop HTML tables that are clearly acting as a table of contents."""

    def _replace(match: re.Match[str]) -> str:
        plain = _table_plain_text(match.group(0))
        if ("목 차" in plain or "목차" in plain) and len(re.findall(r"\d+", plain)) >= 3:
            return ""
        return match.group(0)

    return HTML_TABLE_RE.sub(_replace, text)


def remove_text_toc_block(text: str) -> str:
    """Remove plain-text table-of-contents blocks near the document head."""

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.lstrip("#").strip() in {"목 차", "목차"}:
            block_end = i + 1
            toc_entries = 0
            while block_end < len(lines):
                candidate = lines[block_end].strip()
                if not candidate:
                    block_end += 1
                    continue
                if TOC_ENTRY_RE.match(candidate):
                    toc_entries += 1
                    block_end += 1
                    continue
                break
            if toc_entries >= 3:
                i = block_end
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _extract_markdown_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    parts = stripped.split("|")
    if len(parts) < 3:
        return None
    return [cell.strip() for cell in parts[1:-1]]


def _format_section_heading(marker: str, title: str) -> str | None:
    marker = marker.strip()
    title = title.strip()
    if not marker or not title:
        return None
    if ROMAN_MARKER_RE.fullmatch(marker):
        return f"## {marker.rstrip('.')}." + f" {title}"
    if NUMERIC_MARKER_RE.fullmatch(marker):
        return f"### {marker.rstrip('.')}." + f" {title}"
    return None


def normalize_section_box_tables(text: str) -> str:
    """Convert section-box style markdown tables into normal headings."""

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        current_cells = _extract_markdown_cells(lines[i])
        next_cells = _extract_markdown_cells(lines[i + 1]) if i + 1 < len(lines) else None
        if current_cells and next_cells and TABLE_DELIMITER_RE.match(lines[i + 1].strip()):
            nonempty = [cell for cell in current_cells if cell]
            if len(nonempty) == 2:
                heading = _format_section_heading(nonempty[0], nonempty[1])
                if heading:
                    if result and result[-1].strip():
                        result.append("")
                    result.append(heading)
                    if i + 2 < len(lines) and lines[i + 2].strip() == "":
                        i += 3
                    else:
                        i += 2
                    continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def clean_duplicate_table_cells(text: str) -> str:
    """Collapse table rows whose non-empty cells repeat the same value."""

    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and not TABLE_DELIMITER_RE.match(stripped):
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            if len(cells) >= 2 and len(set(cells)) == 1 and cells[0]:
                result.append(f"| {cells[0]} |")
                continue
        result.append(line)
    return "\n".join(result)


def clean_whitespace(text: str) -> str:
    """Normalize repeated blank lines and multiple spaces."""

    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    normalized: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            normalized.append(line)
        else:
            normalized.append(re.sub(r" {2,}", " ", line))
    return "\n".join(normalized)


def clean_broken_chars(text: str) -> str:
    """Remove broken unicode characters and non-breaking spaces."""

    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    return text.replace("\u00a0", " ")


def clean_kordoc_warnings(text: str) -> str:
    """Remove warning lines emitted by kordoc from the text body."""

    return "\n".join(line for line in text.split("\n") if not line.startswith("Warning: "))


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline in a stable order."""

    if not text or not isinstance(text, str):
        return ""
    text = clean_kordoc_warnings(text)
    text = clean_br_tags(text)
    text = remove_empty_html_tables(text)
    text = remove_toc_html_tables(text)
    text = remove_text_toc_block(text)
    text = normalize_section_box_tables(text)
    text = clean_duplicate_table_cells(text)
    text = clean_broken_chars(text)
    text = clean_whitespace(text)
    return text.strip()
