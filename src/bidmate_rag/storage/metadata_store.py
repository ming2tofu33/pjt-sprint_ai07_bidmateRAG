"""Structured metadata helpers used by retrieval and UI.

파싱된 문서의 메타데이터(사업명, 발주기관 등)를 parquet에서 읽어
검색 및 UI에서 활용할 수 있도록 구조화한다.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

_SEARCH_TEXT_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]+")
_TOKEN_SUFFIXES = (
    "으로부터",
    "들에게",
    "에서도",
    "으로는",
    "으로",
    "에서",
    "에는",
    "에게",
    "까지",
    "부터",
    "처럼",
    "보다",
    "이며",
    "하고",
    "와의",
    "과의",
    "이다",
    "하는",
    "한",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "의",
    "에",
    "로",
)


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = _SEARCH_TEXT_PATTERN.sub(" ", normalized).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _strip_token_suffix(token: str) -> str:
    for suffix in _TOKEN_SUFFIXES:
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _extract_query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _normalize_search_text(query).split():
        if len(raw) < 2:
            continue
        for candidate in (raw, _strip_token_suffix(raw)):
            if len(candidate) < 2 or candidate in seen:
                continue
            tokens.append(candidate)
            seen.add(candidate)
    return tokens


class MetadataStore:
    """문서 메타데이터 저장소. parquet 기반으로 발주기관 목록 조회, 키워드 검색을 제공한다."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """MetadataStore를 초기화한다.

        Args:
            frame: 문서 메타데이터가 담긴 DataFrame.
        """
        # NaN을 빈 문자열로 치환
        self.frame = frame.fillna("")
        # 발주 기관 목록 추출 (UI 필터 드롭다운 등에 사용)
        self.agency_list = (
            sorted(self.frame["발주 기관"].astype(str).unique().tolist())
            if "발주 기관" in self.frame.columns
            else []
        )
        self._search_rows: list[dict[str, str]] = []
        for _, row in self.frame.iterrows():
            self._search_rows.append(
                {
                    "file_name": str(row.get("파일명", "") or ""),
                    "title": _normalize_search_text(row.get("사업명", "")),
                    "summary": _normalize_search_text(row.get("사업 요약", "")),
                    "body": _normalize_search_text(
                        f"{row.get('텍스트', '')} {row.get('본문_마크다운', '')}"
                    ),
                }
            )

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MetadataStore":
        """parquet 파일에서 MetadataStore를 생성한다.

        Args:
            path: parquet 파일 경로.

        Returns:
            MetadataStore 인스턴스.
        """
        return cls(pd.read_parquet(path))

    def find_relevant_docs(self, query: str, top_n: int = 3) -> list[str]:
        """질문과 관련된 문서 파일명을 키워드 매칭으로 찾는다.

        Args:
            query: 사용자 질문 문자열.
            top_n: 반환할 최대 문서 수.

        Returns:
            관련도 높은 순으로 정렬된 파일명 리스트.
        """
        tokens = _extract_query_tokens(query)
        if not tokens:
            return []

        normalized_query = _normalize_search_text(query)
        scored: list[tuple[int, str]] = []

        for row in self._search_rows:
            title = row["title"]
            summary = row["summary"]
            body = row["body"]
            score = 0

            if normalized_query and len(normalized_query) >= 4:
                if normalized_query in title:
                    score += 12
                if normalized_query in summary:
                    score += 8
                if normalized_query in body:
                    score += 5

            for token in tokens:
                if token in title:
                    score += 4
                if token in summary:
                    score += 2
                if token in body:
                    score += 1

            if score > 0 and row["file_name"]:
                scored.append((score, row["file_name"]))

        scored.sort(key=lambda item: item[0], reverse=True)

        ordered_docs: list[str] = []
        seen: set[str] = set()
        for _, file_name in scored:
            if file_name in seen:
                continue
            ordered_docs.append(file_name)
            seen.add(file_name)
            if len(ordered_docs) >= top_n:
                break
        return ordered_docs
