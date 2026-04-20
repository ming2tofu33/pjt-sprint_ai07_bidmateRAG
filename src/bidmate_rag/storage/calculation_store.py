"""SQLite-backed structured fact store for calculation-oriented queries."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

COL_NOTICE_ID = "공고 번호"
COL_TITLE = "사업명"
COL_BUDGET = "사업 금액"
COL_AGENCY = "발주 기관"
COL_PUBLIC_DATE = "공개 일자"
COL_BID_START = "입찰 참여 시작일"
COL_BID_END = "입찰 참여 마감일"
COL_FILE_TYPE = "파일형식"
COL_FILE_NAME = "파일명"
COL_CANONICAL_FILE = "canonical_file"
COL_IS_DUPLICATE = "is_duplicate"
COL_INGEST_ENABLED = "ingest_enabled"
COL_INGEST_FILE = "ingest_file"
COL_RESOLVED_AGENCY = "resolved_agency"
COL_ORIGINAL_AGENCY = "original_agency"
COL_BODY_CHARS = "본문_글자수"
COL_BODY_MARKDOWN = "본문_마크다운"
COL_CLEANED_TEXT = "본문_정제"
COL_CLEANED_CHARS = "정제_글자수"
COL_AGENCY_TYPE = "기관유형"
COL_DOMAIN = "사업도메인"
COL_PUBLIC_YEAR = "공개연도"

_AMOUNT_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{5,})(?:\.\d+)?")
_WINDOW_SIZE = 120
_MIN_AMOUNT_DIGITS = 5
_EXTRACT_KIND_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("allocated_budget", ("배정예산", "배정 예산")),
    (
        "estimated_price",
        ("추정가격", "추정 가격", "추정금액", "추정 금액", "추정계약금액", "추정 계약금액"),
    ),
    ("planned_price", ("예정가격", "예정 가격")),
    ("base_amount", ("기초금액", "기초 금액", "기초가격", "기초 가격")),
    ("contract_amount", ("계약금액", "계약 금액", "계약예산", "계약 예산")),
    (
        "project_budget_labeled",
        (
            "사업예산",
            "사업 예산",
            "사업금액",
            "사업 금액",
            "사업비",
            "총사업비",
            "총 사업비",
            "소요예산",
            "소요 예산",
            "예산액",
        ),
    ),
)
_ALL_LABEL_VARIANTS: tuple[str, ...] = tuple(
    variant
    for _, variants in _EXTRACT_KIND_SPECS
    for variant in variants
)


@dataclass(slots=True)
class CalculationFact:
    """Structured document-level facts used for deterministic calculations."""

    document_key: str
    file_name: str
    ingest_file: str
    canonical_file: str
    title: str
    agency: str
    resolved_agency: str
    original_agency: str
    budget_amount: float | None
    allocated_budget: float | None
    estimated_price: float | None
    planned_price: float | None
    base_amount: float | None
    contract_amount: float | None
    project_budget_labeled: float | None
    public_year: int | None
    published_at: str | None
    bid_start_at: str | None
    bid_end_at: str | None
    bid_window_days: float | None
    notice_id: str | None
    file_type: str
    agency_type: str
    domain: str
    body_chars: int
    cleaned_chars: int
    is_duplicate: bool
    ingest_enabled: bool

    def amount_for_kind(self, kind: str | None) -> float | None:
        if kind is None:
            return self.budget_amount
        if kind == "allocated_budget":
            return self.allocated_budget
        if kind == "estimated_price":
            return self.estimated_price
        if kind == "planned_price":
            return self.planned_price
        if kind == "base_amount":
            return self.base_amount
        if kind == "contract_amount":
            return self.contract_amount
        if kind == "project_budget":
            return self.project_budget_labeled or self.budget_amount
        return self.budget_amount


@dataclass(slots=True)
class BudgetComparison:
    """Comparison result between two document budgets."""

    left: CalculationFact
    right: CalculationFact
    difference: float
    ratio: float | None


def _safe_strip(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_lookup_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_strip(value))
    return " ".join(text.lower().split())


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", _safe_strip(value))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    return _safe_strip(value).lower() in {"1", "true", "t", "yes", "y"}


def _safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = _safe_strip(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = _safe_strip(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_iso_datetime(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().isoformat(sep=" ", timespec="seconds")
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")

    text = _safe_strip(value)
    if not text:
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.to_pydatetime().isoformat(sep=" ", timespec="seconds")


def _days_between(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    start = pd.to_datetime(start_iso, errors="coerce")
    end = pd.to_datetime(end_iso, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return round((end - start).total_seconds() / 86400, 3)


def _find_labeled_amount(text: str, variants: tuple[str, ...]) -> float | None:
    normalized = _normalize_text(text)
    candidates: list[float] = []

    def scan_window(source: str) -> None:
        for variant in variants:
            for match in re.finditer(re.escape(variant), source):
                window_start = max(0, match.start() - (_WINDOW_SIZE // 2))
                window_end = min(len(source), match.start() + _WINDOW_SIZE)
                window = source[window_start:window_end]
                for amount_match in _AMOUNT_PATTERN.finditer(window):
                    digits_only = amount_match.group(0).replace(",", "")
                    if len(digits_only) < _MIN_AMOUNT_DIGITS:
                        continue
                    candidates.append(float(digits_only))

    def has_other_label(source: str) -> bool:
        return any(variant in source for variant in _ALL_LABEL_VARIANTS)

    lines = normalized.splitlines()
    for idx, line in enumerate(lines):
        if any(variant in line for variant in variants):
            before = len(candidates)
            scan_window(line)
            if len(candidates) > before:
                continue
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if not has_other_label(next_line):
                    scan_window(f"{line}\n{next_line}")

    if not candidates:
        return None
    return max(candidates)


def _extract_budget_kinds(row: pd.Series) -> dict[str, float | None]:
    texts = [
        _safe_strip(row.get(COL_CLEANED_TEXT)),
        _safe_strip(row.get(COL_BODY_MARKDOWN)),
    ]
    extracted: dict[str, float | None] = {}
    for kind, variants in _EXTRACT_KIND_SPECS:
        value = None
        for text in texts:
            if not text:
                continue
            value = _find_labeled_amount(text, variants)
            if value is not None:
                break
        extracted[kind] = value
    return extracted


def _document_key(row: pd.Series) -> str:
    for column in (COL_INGEST_FILE, COL_CANONICAL_FILE, COL_FILE_NAME, COL_NOTICE_ID):
        value = _safe_strip(row.get(column))
        if value:
            return value
    raise ValueError("document_key를 만들 수 있는 식별자가 없습니다.")


def _row_to_record(row: pd.Series) -> dict[str, Any]:
    published_at = _to_iso_datetime(row.get(COL_PUBLIC_DATE))
    bid_start_at = _to_iso_datetime(row.get(COL_BID_START))
    bid_end_at = _to_iso_datetime(row.get(COL_BID_END))
    extracted_amounts = _extract_budget_kinds(row)

    return {
        "document_key": _document_key(row),
        "file_name": _safe_strip(row.get(COL_FILE_NAME)),
        "ingest_file": _safe_strip(row.get(COL_INGEST_FILE)) or _safe_strip(row.get(COL_FILE_NAME)),
        "canonical_file": _safe_strip(row.get(COL_CANONICAL_FILE)) or _safe_strip(row.get(COL_FILE_NAME)),
        "title": _safe_strip(row.get(COL_TITLE)),
        "agency": _safe_strip(row.get(COL_AGENCY)),
        "resolved_agency": _safe_strip(row.get(COL_RESOLVED_AGENCY)) or _safe_strip(row.get(COL_AGENCY)),
        "original_agency": _safe_strip(row.get(COL_ORIGINAL_AGENCY)) or _safe_strip(row.get(COL_AGENCY)),
        "budget_amount": _safe_float(row.get(COL_BUDGET)),
        "allocated_budget": extracted_amounts["allocated_budget"],
        "estimated_price": extracted_amounts["estimated_price"],
        "planned_price": extracted_amounts["planned_price"],
        "base_amount": extracted_amounts["base_amount"],
        "contract_amount": extracted_amounts["contract_amount"],
        "project_budget_labeled": extracted_amounts["project_budget_labeled"],
        "public_year": _safe_int(row.get(COL_PUBLIC_YEAR)),
        "published_at": published_at,
        "bid_start_at": bid_start_at,
        "bid_end_at": bid_end_at,
        "bid_window_days": _days_between(bid_start_at, bid_end_at),
        "notice_id": _safe_strip(row.get(COL_NOTICE_ID)) or None,
        "file_type": _safe_strip(row.get(COL_FILE_TYPE)),
        "agency_type": _safe_strip(row.get(COL_AGENCY_TYPE)),
        "domain": _safe_strip(row.get(COL_DOMAIN)),
        "body_chars": _safe_int(row.get(COL_BODY_CHARS)) or 0,
        "cleaned_chars": _safe_int(row.get(COL_CLEANED_CHARS)) or 0,
        "is_duplicate": int(_safe_bool(row.get(COL_IS_DUPLICATE), False)),
        "ingest_enabled": int(_safe_bool(row.get(COL_INGEST_ENABLED), True)),
    }


class CalculationStore:
    """SQLite store for structured numeric/date facts."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def create(cls, db_path: str | Path = ":memory:") -> "CalculationStore":
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path))

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        db_path: str | Path = ":memory:",
    ) -> "CalculationStore":
        store = cls.create(db_path)
        frame = pd.read_parquet(parquet_path)
        store.rebuild_from_frame(frame)
        return store

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_facts (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_key TEXT NOT NULL,
                file_name TEXT NOT NULL,
                ingest_file TEXT NOT NULL,
                canonical_file TEXT NOT NULL,
                title TEXT NOT NULL,
                agency TEXT NOT NULL,
                resolved_agency TEXT NOT NULL,
                original_agency TEXT NOT NULL,
                budget_amount REAL,
                allocated_budget REAL,
                estimated_price REAL,
                planned_price REAL,
                base_amount REAL,
                contract_amount REAL,
                project_budget_labeled REAL,
                public_year INTEGER,
                published_at TEXT,
                bid_start_at TEXT,
                bid_end_at TEXT,
                bid_window_days REAL,
                notice_id TEXT,
                file_type TEXT NOT NULL,
                agency_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                body_chars INTEGER NOT NULL DEFAULT 0,
                cleaned_chars INTEGER NOT NULL DEFAULT 0,
                is_duplicate INTEGER NOT NULL DEFAULT 0,
                ingest_enabled INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_document_facts_agency
                ON document_facts(resolved_agency);
            CREATE INDEX IF NOT EXISTS idx_document_facts_year
                ON document_facts(public_year);
            CREATE INDEX IF NOT EXISTS idx_document_facts_budget
                ON document_facts(budget_amount);
            CREATE INDEX IF NOT EXISTS idx_document_facts_document_key
                ON document_facts(document_key);
            """
        )
        self.connection.commit()

    def rebuild_from_frame(self, frame: pd.DataFrame) -> None:
        records = [_row_to_record(row) for _, row in frame.fillna("").iterrows()]
        with self.connection:
            self.connection.execute("DROP TABLE IF EXISTS document_facts")
            self._ensure_schema()
            self.connection.executemany(
                """
                INSERT INTO document_facts (
                    document_key, file_name, ingest_file, canonical_file, title, agency,
                    resolved_agency, original_agency, budget_amount, allocated_budget,
                    estimated_price, planned_price, base_amount, contract_amount, project_budget_labeled, public_year,
                    published_at, bid_start_at, bid_end_at, bid_window_days, notice_id,
                    file_type, agency_type, domain, body_chars, cleaned_chars,
                    is_duplicate, ingest_enabled
                ) VALUES (
                    :document_key, :file_name, :ingest_file, :canonical_file, :title, :agency,
                    :resolved_agency, :original_agency, :budget_amount, :allocated_budget,
                    :estimated_price, :planned_price, :base_amount, :contract_amount, :project_budget_labeled, :public_year,
                    :published_at, :bid_start_at, :bid_end_at, :bid_window_days, :notice_id,
                    :file_type, :agency_type, :domain, :body_chars, :cleaned_chars,
                    :is_duplicate, :ingest_enabled
                )
                """,
                records,
            )

    def count(self, canonical_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM document_facts"
        if canonical_only:
            sql += " WHERE ingest_enabled = 1"
        return int(self.connection.execute(sql).fetchone()[0])

    def get_fact(self, doc_id: str, canonical_only: bool = True) -> CalculationFact | None:
        where = [
            "(document_key = ? OR file_name = ? OR ingest_file = ? OR canonical_file = ? OR notice_id = ?)"
        ]
        params: list[Any] = [doc_id, doc_id, doc_id, doc_id, doc_id]
        if canonical_only:
            where.append("ingest_enabled = 1")
        row = self.connection.execute(
            f"""
            SELECT * FROM document_facts
            WHERE {' AND '.join(where)}
            ORDER BY ingest_enabled DESC, cleaned_chars DESC, file_name ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row:
            return self._row_to_fact(row)

        normalized = _normalize_lookup_text(doc_id)
        if not normalized:
            return None

        fallback_rows = self.connection.execute(
            """
            SELECT * FROM document_facts
            ORDER BY ingest_enabled DESC, cleaned_chars DESC, file_name ASC
            """
        ).fetchall()
        for candidate in fallback_rows:
            if canonical_only and not bool(candidate["ingest_enabled"]):
                continue
            keys = (
                candidate["document_key"],
                candidate["file_name"],
                candidate["ingest_file"],
                candidate["canonical_file"],
                candidate["notice_id"],
            )
            if any(_normalize_lookup_text(key) == normalized for key in keys if key):
                return self._row_to_fact(candidate)
        return None

    def list_facts(
        self,
        doc_ids: list[str] | None = None,
        agency: str | None = None,
        year: int | None = None,
        canonical_only: bool = True,
    ) -> list[CalculationFact]:
        sql = "SELECT * FROM document_facts"
        clauses: list[str] = []
        params: list[Any] = []

        if canonical_only:
            clauses.append("ingest_enabled = 1")
        if agency:
            clauses.append("(resolved_agency = ? OR agency = ? OR original_agency = ?)")
            params.extend([agency, agency, agency])
        if year is not None:
            clauses.append("public_year = ?")
            params.append(year)
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            clauses.append(
                "("
                f"document_key IN ({placeholders}) OR "
                f"file_name IN ({placeholders}) OR "
                f"ingest_file IN ({placeholders}) OR "
                f"canonical_file IN ({placeholders}) OR "
                f"notice_id IN ({placeholders})"
                ")"
            )
            params.extend(doc_ids * 5)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY budget_amount IS NULL ASC, budget_amount DESC, cleaned_chars DESC, file_name ASC"

        rows = self.connection.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def budget_by_kind(
        self,
        doc_id: str,
        kind: str | None,
        canonical_only: bool = True,
    ) -> float | None:
        fact = self.get_fact(doc_id, canonical_only=canonical_only)
        if fact is None:
            raise KeyError(f"계산할 문서를 찾을 수 없습니다: {doc_id}")
        return fact.amount_for_kind(kind)

    def sum_budget(
        self,
        doc_ids: list[str] | None = None,
        agency: str | None = None,
        year: int | None = None,
        canonical_only: bool = True,
        budget_kind: str | None = None,
    ) -> float:
        rows = self.list_facts(doc_ids=doc_ids, agency=agency, year=year, canonical_only=canonical_only)
        return round(sum(row.amount_for_kind(budget_kind) or 0.0 for row in rows), 3)

    def average_budget(
        self,
        doc_ids: list[str] | None = None,
        agency: str | None = None,
        year: int | None = None,
        canonical_only: bool = True,
        budget_kind: str | None = None,
    ) -> float | None:
        rows = [
            row
            for row in self.list_facts(doc_ids, agency, year, canonical_only)
            if row.amount_for_kind(budget_kind) is not None
        ]
        if not rows:
            return None
        return round(sum(row.amount_for_kind(budget_kind) or 0.0 for row in rows) / len(rows), 3)

    def compare_budget(
        self,
        left_doc_id: str,
        right_doc_id: str,
        canonical_only: bool = True,
        budget_kind: str | None = None,
    ) -> BudgetComparison:
        left = self.get_fact(left_doc_id, canonical_only=canonical_only)
        right = self.get_fact(right_doc_id, canonical_only=canonical_only)
        if left is None or right is None:
            missing = left_doc_id if left is None else right_doc_id
            raise KeyError(f"계산할 문서를 찾을 수 없습니다: {missing}")

        left_amount = left.amount_for_kind(budget_kind)
        right_amount = right.amount_for_kind(budget_kind)
        if left_amount is None or right_amount is None:
            raise ValueError("선택한 금액 종류가 없는 문서는 비교할 수 없습니다.")

        difference = round(left_amount - right_amount, 3)
        ratio = None
        if right_amount != 0:
            ratio = round(left_amount / right_amount, 6)
        return BudgetComparison(left=left, right=right, difference=difference, ratio=ratio)

    def apply_budget_ratio(
        self,
        doc_id: str,
        ratio: float,
        canonical_only: bool = True,
        budget_kind: str | None = None,
    ) -> float:
        fact = self.get_fact(doc_id, canonical_only=canonical_only)
        if fact is None:
            raise ValueError("계산 대상 문서가 없습니다.")
        amount = fact.amount_for_kind(budget_kind)
        if amount is None:
            raise ValueError("선택한 금액 종류가 없는 문서는 비율 계산할 수 없습니다.")
        return round(amount * ratio, 3)

    def bid_window_days(self, doc_id: str, canonical_only: bool = True) -> float | None:
        fact = self.get_fact(doc_id, canonical_only=canonical_only)
        if fact is None:
            raise KeyError(f"계산할 문서를 찾을 수 없습니다: {doc_id}")
        return fact.bid_window_days

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> CalculationFact:
        return CalculationFact(
            document_key=row["document_key"],
            file_name=row["file_name"],
            ingest_file=row["ingest_file"],
            canonical_file=row["canonical_file"],
            title=row["title"],
            agency=row["agency"],
            resolved_agency=row["resolved_agency"],
            original_agency=row["original_agency"],
            budget_amount=row["budget_amount"],
            allocated_budget=row["allocated_budget"],
            estimated_price=row["estimated_price"],
            planned_price=row["planned_price"],
            base_amount=row["base_amount"],
            contract_amount=row["contract_amount"],
            project_budget_labeled=row["project_budget_labeled"],
            public_year=row["public_year"],
            published_at=row["published_at"],
            bid_start_at=row["bid_start_at"],
            bid_end_at=row["bid_end_at"],
            bid_window_days=row["bid_window_days"],
            notice_id=row["notice_id"],
            file_type=row["file_type"],
            agency_type=row["agency_type"],
            domain=row["domain"],
            body_chars=row["body_chars"],
            cleaned_chars=row["cleaned_chars"],
            is_duplicate=bool(row["is_duplicate"]),
            ingest_enabled=bool(row["ingest_enabled"]),
        )
