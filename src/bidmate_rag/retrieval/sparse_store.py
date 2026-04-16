"""Lightweight in-memory BM25 sparse retrieval over chunk parquet files."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.schema import Chunk, RetrievedChunk

# 한글·영문·숫자 2글자 이상 토큰만 추출
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokenize(text: object) -> list[str]:
    """텍스트를 소문자 토큰 리스트로 분리한다 (2글자 이상만)."""
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(str(text or ""))
        if len(token) >= 2
    ]


def _coerce_numeric(value: Any) -> float | None:
    """숫자로 변환 가능하면 float 반환, 아니면 None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_operator(field_value: Any, operator: str, operand: Any) -> bool:
    """단일 비교 연산자($in, $gte, $gt, $lte, $lt) 매칭 여부를 판단한다."""
    if operator == "$in":
        return field_value in operand

    left = _coerce_numeric(field_value)
    right = _coerce_numeric(operand)
    if left is None or right is None:
        return False

    if operator == "$gte":
        return left >= right
    if operator == "$gt":
        return left > right
    if operator == "$lte":
        return left <= right
    if operator == "$lt":
        return left < right
    return False


def _matches_where(record: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """ChromaDB 스타일 where 필터 조건과 레코드를 매칭한다."""
    if not where:
        return True

    # $and / $or 복합 조건 처리
    if "$and" in where:
        return all(_matches_where(record, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches_where(record, clause) for clause in where["$or"])

    for field, expected in where.items():
        value = record.get(field)
        if isinstance(expected, dict):
            for operator, operand in expected.items():
                if not _matches_operator(value, operator, operand):
                    return False
            continue
        if value == expected:
            continue
        if str(value) != str(expected):
            return False
    return True


@dataclass(slots=True)
class _SparseEntry:
    chunk: Chunk
    record: dict[str, Any]
    doc_len: int


class BM25SparseStore:
    """청크 parquet 데이터 기반 인메모리 BM25 검색 스토어."""

    def __init__(
        self,
        entries: list[_SparseEntry],
        inverted_index: dict[str, list[tuple[int, int]]],
        doc_freq: dict[str, int],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """BM25SparseStore를 초기화한다.

        Args:
            entries: 청크별 토큰·메타 정보 리스트.
            inverted_index: 역색인 {토큰: [(문서인덱스, 빈도), ...]}.
            doc_freq: 문서 빈도 {토큰: 등장 문서 수}.
            k1: BM25 term frequency 포화 파라미터.
            b: BM25 문서 길이 정규화 파라미터.
        """
        self.entries = entries
        self.inverted_index = inverted_index
        self.doc_freq = doc_freq
        self.num_docs = len(entries)
        self.avg_doc_len = (
            sum(entry.doc_len for entry in entries) / len(entries) if entries else 1.0
        )
        self.k1 = k1
        self.b = b

    @classmethod
    def from_parquet(cls, path: str | Path) -> "BM25SparseStore":
        """청크 parquet 파일에서 BM25 인덱스를 빌드한다.

        Args:
            path: chunks.parquet 경로.

        Returns:
            인덱싱 완료된 BM25SparseStore 인스턴스.
        """
        frame = pd.read_parquet(path).fillna("")
        entries: list[_SparseEntry] = []
        inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)

        for row in frame.to_dict(orient="records"):
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"text", "text_with_meta", "char_count"}
            }
            chunk = Chunk(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                text=str(row.get("text", "")),
                text_with_meta=str(row.get("text_with_meta") or row.get("text", "")),
                char_count=int(row.get("char_count") or len(str(row.get("text", "")))),
                section=str(row.get("section", "")),
                content_type=str(row.get("content_type", "text")),
                chunk_index=int(row.get("chunk_index", 0)),
                metadata=metadata,
            )
            tokens = _tokenize(chunk.text_with_meta)
            term_counts = Counter(tokens)
            doc_len = sum(term_counts.values()) or 1
            entry = _SparseEntry(
                chunk=chunk,
                doc_len=doc_len,
                record={
                    **metadata,
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "content_type": chunk.content_type,
                    "chunk_index": chunk.chunk_index,
                },
            )
            doc_index = len(entries)
            entries.append(entry)
            for term, tf in term_counts.items():
                inverted_index[term].append((doc_index, tf))
                doc_freq[term] += 1

        return cls(entries, dict(inverted_index), dict(doc_freq))

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 점수 기반으로 관련 청크를 검색한다.

        Args:
            query: 사용자 질의 문자열.
            top_k: 반환할 최대 청크 수.
            where: ChromaDB 스타일 메타데이터 필터.

        Returns:
            BM25 점수 기준 상위 top_k개 RetrievedChunk 리스트.
        """
        # 질의 토큰 추출 (중복 제거)
        query_terms = list(dict.fromkeys(_tokenize(query)))
        if not query_terms or not self.entries:
            return []

        scores: dict[int, float] = defaultdict(float)
        matches_cache: dict[int, bool] = {}  # where 필터 매칭 결과 캐시

        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            if not postings:
                continue

            # IDF 계산: 해당 토큰이 얼마나 희귀한지
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + ((self.num_docs - df + 0.5) / (df + 0.5)))

            for doc_index, tf in postings:
                # where 필터 매칭 여부 캐싱
                if doc_index not in matches_cache:
                    matches_cache[doc_index] = _matches_where(self.entries[doc_index].record, where)
                if not matches_cache[doc_index]:
                    continue

                # BM25 점수 누적: IDF * TF 정규화
                doc_len = self.entries[doc_index].doc_len
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[doc_index] += idf * (numerator / denominator)

        if not scores:
            return []

        # 상위 top_k 선택 후 min-max 정규화
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        raw_scores = [score for _, score in ordered]
        max_score = max(raw_scores)
        min_score = min(raw_scores)

        retrieved: list[RetrievedChunk] = []
        for rank, (doc_index, raw_score) in enumerate(ordered, start=1):
            normalized = 1.0 if max_score == min_score else (raw_score - min_score) / (
                max_score - min_score
            )
            entry = self.entries[doc_index]
            chunk = entry.chunk.model_copy(deep=True)
            chunk.metadata["sparse_bm25_score"] = round(raw_score, 6)
            chunk.metadata["retrieval_source"] = "sparse"
            retrieved.append(
                RetrievedChunk(
                    rank=rank,
                    score=round(normalized, 4),
                    chunk=chunk,
                )
            )
        return retrieved
