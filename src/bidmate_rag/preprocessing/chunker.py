"""Chunking helpers based on the notebook baseline."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

import pandas as pd
from bidmate_rag.schema import Chunk

HEADERS_TO_SPLIT = [("#", "h1")]
MIN_SECTION_SIZE = 500

# ── 청킹 전략 설정 ──

CHUNKING_PRESETS = {
    "small": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "min_section_size": 300,
        "max_table_size": 1000,
        "description": "작은 청크 — 팩토이드/단답형 질문에 유리",
    },
    "medium": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "min_section_size": 500,
        "max_table_size": 1500,
        "description": "기본 설정 — baseline",
    },
    "large": {
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "min_section_size": 700,
        "max_table_size": 2000,
        "description": "큰 청크 — 분석형/비교 질문에 유리",
    },
}


@dataclass
class ChunkingConfig:
    """청킹 전략 설정. preset 또는 커스텀 값으로 생성."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_section_size: int = 500
    max_table_size: int = 1500
    headers_to_split: list[tuple[str, str]] = field(default_factory=lambda: [("#", "h1")])
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    description: str = "기본 설정"
    chunking_strategy: str = "recursive" # "recursive" 또는 "semantic"
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95.0

    @classmethod
    def from_preset(cls, name: str) -> "ChunkingConfig":
        """프리셋 이름으로 ChunkingConfig를 생성

        Args:
            name: 프리셋 이름 (small / medium / large).

        Returns:
            해당 프리셋의 ChunkingConfig.
        """
        if name not in CHUNKING_PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(CHUNKING_PRESETS.keys())}")
        return cls(
            **{k: v for k, v in CHUNKING_PRESETS[name].items() if k != "description"},
            description=CHUNKING_PRESETS[name]["description"],
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChunkingConfig":
        """YAML 파일에서 ChunkingConfig를 로드

        Args:
            path: YAML 설정 파일 경로.

        Returns:
            YAML 값으로 생성된 ChunkingConfig.
        """
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            chunk_size=cfg.get("chunk_size", 1000),
            chunk_overlap=cfg.get("chunk_overlap", 150),
            min_section_size=cfg.get("min_section_size", 500),
            max_table_size=cfg.get("max_table_size", 1500),
            description=cfg.get("description", cfg.get("name", "")),
            chunking_strategy=cfg.get("chunking_strategy", "recursive"),
            breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold_amount=cfg.get("breakpoint_threshold_amount", 95.0),
        )

    def to_dict(self) -> dict:
        """설정값을 딕셔너리로 변환

        Returns:
            청킹 설정 딕셔너리.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_section_size": self.min_section_size,
            "max_table_size": self.max_table_size,
            "description": self.description,
        }

    @staticmethod
    def list_presets() -> dict[str, dict]:
        """사용 가능한 프리셋 목록을 반환

        Returns:
            프리셋 이름-설정 딕셔너리.
        """
        return CHUNKING_PRESETS


TECH_STACK_KEYWORDS = {
    "AI": ["AI", "인공지능", "머신러닝", "딥러닝"],
    "클라우드": ["클라우드", "cloud", "SaaS"],
    "IoT": ["IoT", "센서", "SCADA"],
    "GIS": ["GIS", "지도", "공간정보"],
    "모바일": ["모바일앱", "모바일 앱"],
}


def classify_agency(name: str) -> str:
    """발주기관명으로 기관유형을 분류

    Args:
        name: 발주기관 이름.

    Returns:
        기관유형 문자열.
    """
    name = str(name)
    # 키워드 매칭으로 기관유형 판별
    if any(keyword in name for keyword in ["부 ", "처 ", "청 ", "위원회", "대검찰청", "선거관리"]):
        return "중앙행정기관"
    if any(keyword in name for keyword in ["대학", "학교"]):
        return "대학교"
    if any(keyword in name for keyword in ["연구원", "연구소", "과학", "나노종합기술원"]):
        return "연구기관"
    if any(
        keyword in name
        for keyword in ["공사", "공단", "진흥원", "진흥회", "평가원", "정보원", "테크노파크"]
    ):
        return "공기업/준정부기관"
    if any(
        keyword in name for keyword in ["특별시", "광역시", "특별자치", "도 ", "시 ", "군 ", "구 "]
    ):
        return "지방자치단체"
    if any(
        keyword in name
        for keyword in [
            "협회",
            "협의회",
            "재단",
            "센터",
            "사단법인",
            "(사)",
            "체육회",
            "상공회의소",
        ]
    ):
        return "협회/재단"
    if any(keyword in name for keyword in ["국립", "박물관", "의료원", "BioIN"]):
        return "공공기관"
    if any(keyword in name for keyword in ["(주)", "주식회사", "㈜"]):
        return "민간기업"
    return "기타"


def classify_domain(name: str, text: str = "") -> str:
    """사업명과 본문으로 사업 도메인을 분류

    Args:
        name: 사업명.
        text: 본문 텍스트.

    Returns:
        도메인 문자열.
    """
    # 사업명 + 본문 앞부분을 결합하여 키워드 매칭
    combined = f"{name} {text[:5000]}"
    rules = [
        ("교육/학습", ["교육", "이러닝", "학습", "학사", "LMS", "LRS", "연수", "아카데미"]),
        ("안전/재난", ["안전", "재난", "방재", "관제", "선량"]),
        ("웹/포털", ["홈페이지", "포털", "웹", "온라인서비스", "플랫폼"]),
        ("경영/행정", ["ERP", "그룹웨어", "경영", "인사", "회계", "전자결재", "오피스"]),
        ("공간정보/GIS", ["GIS", "지도", "공간", "측량", "수문", "관개", "수자원"]),
        ("의료/바이오", ["의료", "건강", "바이오", "병원", "보험"]),
        ("ISP/컨설팅", ["ISP", "전략", "컨설팅", "타당성", "ISMP"]),
        ("AI/데이터", ["AI", "인공지능", "빅데이터", "데이터분석"]),
        ("교통/물류", ["버스정보", "교통", "BIS", "ITS"]),
        ("농축수산", ["축산", "농업", "수산", "어촌", "품질평가"]),
        ("문화/콘텐츠", ["문화", "예술", "박물관", "아카이브", "영화"]),
        ("복지/사회서비스", ["복지", "돌봄", "사회보험", "사회보장", "서민금융"]),
        ("조달/계약", ["조달", "입찰", "계약관리"]),
    ]
    for domain, keywords in rules:
        # 일부 도메인은 사업명만으로 판별, 나머지는 본문까지 포함하여 판별
        target = (
            name
            if domain
            in {
                "교육/학습",
                "안전/재난",
                "웹/포털",
                "경영/행정",
                "공간정보/GIS",
                "의료/바이오",
                "ISP/컨설팅",
                "AI/데이터",
            }
            else combined
        )
        if any(keyword in target for keyword in keywords):
            return domain
    return "기타"


def extract_tech_stack(text: str) -> str:
    """본문에서 기술스택 키워드를 추출

    Args:
        text: 본문 텍스트.

    Returns:
        쉼표로 구분된 기술스택 문자열.
    """
    matches = [
        label
        for label, keywords in TECH_STACK_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    return ", ".join(matches)


def split_by_headers(text: str, min_size: int = MIN_SECTION_SIZE) -> list[dict]:
    """마크다운 헤더 기준으로 텍스트를 섹션 분할

    Args:
        text: 마크다운 텍스트.
        min_size: 섹션 최소 글자수 (미만이면 다음 섹션과 병합).

    Returns:
        섹션 딕셔너리(text, section, char_count) 리스트.
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT, strip_headers=False)
    docs = splitter.split_text(text)
    merged: list[dict] = []
    buffer_text = ""
    buffer_headers: list[str] = []
    # 최소 크기 미만인 섹션은 다음 섹션과 병합
    for doc in docs:
        header = doc.metadata.get("h1", "")
        buffer_text = (
            f"{buffer_text}\n\n{doc.page_content}".strip() if buffer_text else doc.page_content
        )
        if header and header not in buffer_headers:
            buffer_headers.append(header)
        if len(buffer_text) >= min_size:
            merged.append(
                {
                    "text": buffer_text,
                    "section": buffer_headers[-1] if buffer_headers else "",
                    "char_count": len(buffer_text),
                }
            )
            buffer_text = ""
            buffer_headers = []
    if buffer_text:
        merged.append(
            {
                "text": buffer_text,
                "section": buffer_headers[-1] if buffer_headers else "",
                "char_count": len(buffer_text),
            }
        )
    return merged


def is_table_block(text: str) -> bool:
    """텍스트가 마크다운 테이블 블록인지 판별

    Args:
        text: 판별할 텍스트.

    Returns:
        파이프(|)로 시작하는 줄이 50% 초과이면 True.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    table_lines = sum(1 for line in lines if line.strip().startswith("|"))
    return table_lines / len(lines) > 0.5


def split_table_with_headers(text: str, max_size: int) -> list[str]:
    """큰 테이블을 헤더를 보존하며 여러 청크로 분할

    Args:
        text: 테이블을 포함한 텍스트.
        max_size: 청크 최대 글자수.

    Returns:
        분할된 테이블 청크 문자열 리스트.
    """
    lines = text.split("\n")
    # 테이블 전후 텍스트 분리
    pre_table: list[str] = []
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table:
            table_lines.append(line)
        else:
            pre_table.append(line)
    if not table_lines or len(text) <= max_size:
        return [text]
    # 테이블 헤더(컬럼명 + 구분선) 추출
    header_lines: list[str] = []
    for table_line in table_lines:
        header_lines.append(table_line)
        if re.match(r"^\|[\s\-:|]+\|$", table_line.strip()):
            break
    header_text = "\n".join(header_lines)
    preamble = "\n".join(pre_table).strip()
    prefix = f"{preamble}\n{header_text}".strip()
    data_rows = table_lines[len(header_lines) :]
    # 각 청크에 헤더를 붙여서 max_size 이내로 분할
    chunks: list[str] = []
    current_rows: list[str] = []
    for row in data_rows:
        candidate = f"{prefix}\n{'\n'.join(current_rows + [row])}".strip()
        if len(candidate) > max_size and current_rows:
            chunks.append(f"{prefix}\n{'\n'.join(current_rows)}".strip())
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        chunks.append(f"{prefix}\n{'\n'.join(current_rows)}".strip())
    return chunks or [text]


def _chunk_doc_id(doc_metadata: dict) -> str:
    """메타데이터에서 문서 ID를 추출
    Args:
        doc_metadata: 문서 메타데이터.
    Returns:
        문서 ID 문자열.
    """
    for key in ("doc_id", "공고 번호", "파일명"):
        val = doc_metadata.get(key)
        if val is not None and pd.notna(val) and str(val).strip():
            return str(val).strip()
    return "unknown-doc"

def _meta_prefix(doc_metadata: dict) -> str:
    """청크에 붙일 메타데이터 접두어를 생성

    Args:
        doc_metadata: 문서 메타데이터.

    Returns:
        "[발주기관: ... | 사업명: ...]" 형식 문자열.
    """
    agency = doc_metadata.get("발주 기관", "")
    project = doc_metadata.get("사업명", "")
    return f"[발주기관: {agency} | 사업명: {project}]"


def chunk_document(
    text: str,
    doc_metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_table_size: int = 1500,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """문서 텍스트를 섹션 분할 후 청크 리스트로 변환

    Args:
        text: 문서 전체 텍스트.
        doc_metadata: 문서 메타데이터.
        chunk_size: 청크 최대 글자수.
        chunk_overlap: 청크 간 겹침 글자수.
        max_table_size: 테이블 청크 최대 글자수.
        config: ChunkingConfig (지정 시 개별 파라미터 무시).

    Returns:
        Chunk 리스트.
    """
    # config가 있으면 개별 파라미터를 덮어씀
    if config is not None:
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        max_table_size = config.max_table_size
    # 헤더 기준 섹션 분할 → 각 섹션을 텍스트/테이블 유형별로 청킹
    sections = split_by_headers(
        text, min_size=config.min_section_size if config else MIN_SECTION_SIZE
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.separators if config else ["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    chunk_index = 0
    doc_id = _chunk_doc_id(doc_metadata)
    prefix = _meta_prefix(doc_metadata)

    def append_chunk(chunk_text: str, section: str, content_type: str) -> None:
        nonlocal chunk_index
        chunks.append(
            Chunk(
                chunk_id=f"{doc_metadata.get('파일명', doc_id)}_{chunk_index}",
                doc_id=doc_id,
                text=chunk_text,
                text_with_meta=f"{prefix}\n{chunk_text}",
                char_count=len(chunk_text),
                section=section,
                content_type=content_type,
                chunk_index=chunk_index,
                metadata=dict(doc_metadata),
            )
        )
        chunk_index += 1

    for section in sections:
        section_text = section["text"]
        section_name = section["section"]
        if len(section_text) <= chunk_size:
            append_chunk(
                section_text, section_name, "table" if is_table_block(section_text) else "text"
            )
            continue
        if is_table_block(section_text):
            for table_chunk in split_table_with_headers(section_text, max_table_size):
                append_chunk(table_chunk, section_name, "table")
            continue
        if config and config.chunking_strategy == "semantic":
            semantic_chunker = SemanticChunker(
                OpenAIEmbeddings(model="text-embedding-3-small"),
                breakpoint_threshold_type=config.breakpoint_threshold_type,
                breakpoint_threshold_amount=config.breakpoint_threshold_amount,
                )
            semantic_docs = semantic_chunker.create_documents([section_text])
            for sem_doc in semantic_docs:
                sub_text = sem_doc.page_content
                # 의미기반 청크가 chunk_size 초과하면 recursive로 후처리
                if len(sub_text) > chunk_size:
                    for subdoc in splitter.create_documents([sub_text]):
                        append_chunk(subdoc.page_content, section_name, "text")
                else:
                    append_chunk(sub_text, section_name, "text")
        else:
            for subdoc in splitter.create_documents([section_text]):
                append_chunk(subdoc.page_content, section_name, "text")
    return chunks


def chunk_dataframe(
    rows: Iterable[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """DataFrame 행들을 순회하며 각 문서를 청킹

    Args:
        rows: 문서 메타데이터 + 본문_정제 키를 가진 딕셔너리 이터러블.
        chunk_size: 청크 최대 글자수.
        chunk_overlap: 청크 간 겹침 글자수.
        config: ChunkingConfig (지정 시 개별 파라미터 무시).

    Returns:
        전체 문서의 Chunk 리스트.
    """
    all_chunks: list[Chunk] = []
    for row in rows:
        all_chunks.extend(
            chunk_document(
                row["본문_정제"],
                row,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                config=config,
            )
        )
    return all_chunks
