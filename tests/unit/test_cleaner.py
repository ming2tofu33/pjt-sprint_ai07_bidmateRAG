from bidmate_rag.preprocessing.cleaner import clean_text


def test_clean_text_removes_known_noise_and_preserves_real_table_rows() -> None:
    raw = """Warning: Cannot access the `require` function
Warning: Cannot polyfill `DOMMatrix`
Sentence  with<br>line break
| dup | dup | dup |
| keep  two | cell |
\ue000broken\u00a0space



tail"""

    cleaned = clean_text(raw)

    assert "Warning:" not in cleaned
    assert "<br>" not in cleaned
    assert "Sentence with\nline break" in cleaned
    assert "| dup |" in cleaned
    assert "| keep  two | cell |" in cleaned
    assert "\ue000" not in cleaned
    assert "\u00a0" not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_removes_toc_html_table_and_normalizes_section_boxes() -> None:
    raw = """# 2024. 10.

<table>
<tr><th></th><th></th><th>목 차</th></tr>
<tr><td colspan="3">Ⅰ. 사업 안내 1<br>1. 사업개요 1<br>2. 추진배경 2<br>Ⅲ. 기타사항 9</td></tr>
</table>

| Ⅰ |  | 사업 안내 |
| --- | --- | --- |

| 1 |  | 사업개요 |  |
| --- | --- | --- | --- |

## □ 사업예산 : 130,000,000원 범위 내 (VAT 포함)
"""

    cleaned = clean_text(raw)

    assert "목 차" not in cleaned
    assert "<table>" not in cleaned
    assert "## Ⅰ. 사업 안내" in cleaned
    assert "### 1. 사업개요" in cleaned
    assert "130,000,000원" in cleaned


def test_clean_text_removes_plain_text_toc_block_but_keeps_body_content() -> None:
    raw = """# 2024년 이러닝시스템 운영 용역 제안요청서

# 목 차

Ⅰ. 사업 안내\t1

1. 사업개요\t1

2. 사업목적\t1

Ⅲ. 입찰 관련사항 및 제안서 평가\t20

| Ⅰ |  | 사업 안내 |
| --- | --- | --- |

# 1. 사업개요

## □ 사업기간: 계약체결일로부터 2025. 2월까지
"""

    cleaned = clean_text(raw)

    assert "# 목 차" not in cleaned
    assert "Ⅲ. 입찰 관련사항 및 제안서 평가\t20" not in cleaned
    assert "## Ⅰ. 사업 안내" in cleaned
    assert "계약체결일로부터 2025. 2월까지" in cleaned


def test_clean_text_keeps_information_table() -> None:
    raw = """| 담 당 | 부서명 | 성 명 | 전 화 |
| --- | --- | --- | --- |
| 사업관련 | 학술데이터분석팀 | 한민아 | 042-869-6672 |
| 계약관련 | 운영지원팀 | 이현우 | 042-869-6232 |
"""

    cleaned = clean_text(raw)

    assert "| 담 당 | 부서명 | 성 명 | 전 화 |" in cleaned
    assert "한민아" in cleaned
    assert "이현우" in cleaned
