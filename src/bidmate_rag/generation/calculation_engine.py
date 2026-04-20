"""Deterministic calculator for numeric/date questions over structured facts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from uuid import uuid4

from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.storage.calculation_store import CalculationFact, CalculationStore

_BUDGET_KEYWORDS = ("예산", "사업비", "사업 금액", "사업금액", "소요예산")
_DIFF_KEYWORDS = ("차이", "차액", "얼마 차", "얼마나 차")
_SUM_KEYWORDS = ("합산", "총합", "합계", "더하면", "모두 합한")
_AVG_KEYWORDS = ("평균", "평균값")
_RATIO_KEYWORDS = ("몇 배", "비율", "배수")
_MAX_KEYWORDS = ("가장 큰", "제일 큰", "최대", "큰 사업", "더 큰 사업", "예산 규모가 큰")
_MIN_KEYWORDS = ("가장 작은", "제일 작은", "최소", "작은 사업", "더 작은 사업", "예산 규모가 작은")
_ORDER_ASC_KEYWORDS = ("작은 순", "낮은 순", "오름차순", "작은 것부터")
_ORDER_DESC_KEYWORDS = ("큰 순", "높은 순", "내림차순", "큰 것부터")
_ORDER_GENERIC_KEYWORDS = ("순서대로", "정렬", "나열", "비교하면")
_BID_WINDOW_KEYWORDS = (
    "입찰 기간",
    "참여 기간",
    "입찰 참여 기간",
    "참여 시작",
    "참여 마감",
    "시작일",
    "마감일",
    "며칠",
    "얼마 동안",
)
_BUDGET_SINGLE_BLOCKERS = (
    "비율",
    "몇프로",
    "몇 프로",
    "퍼센트",
    "백분율",
    "배수",
    "페이지",
    "계좌",
    "납부",
    "보증금",
    "기한",
    "제출",
    "하도급",
    "재하도급",
    "허용",
    "원칙",
    "지체상금",
    "감점",
    "서약서",
    "확약서",
)
_BID_WINDOW_BLOCKERS = (
    "보증금",
    "납부",
    "계좌",
    "페이지",
    "서류",
    "지체상금",
    "감점",
    "서약서",
    "확약서",
)
_GENERIC_ANCHOR_TOKENS = {
    "사업",
    "구축",
    "용역",
    "시스템",
    "기능",
    "기능개선",
    "고도화",
    "운영",
    "지원",
    "통합",
    "정보",
    "차세대",
    "재공고",
    "긴급",
    "사업의",
}
_QUOTED_ANCHOR_PATTERN = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,})[\"“”'‘’]")
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
_PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_AMOUNT_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{5,})(?:\.\d+)?")

_BUDGET_KIND_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("allocated_budget", "배정예산", ("배정예산", "배정 예산")),
    (
        "estimated_price",
        "추정가격",
        ("추정가격", "추정 가격", "추정금액", "추정 금액", "추정계약금액", "추정 계약금액"),
    ),
    ("planned_price", "예정가격", ("예정가격", "예정 가격")),
    ("base_amount", "기초금액", ("기초금액", "기초 금액", "기초가격", "기초 가격")),
    ("contract_amount", "계약금액", ("계약금액", "계약 금액", "계약예산", "계약 예산")),
    (
        "project_budget",
        "사업예산",
        (
            "사업예산",
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


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _compact_text(value: str) -> str:
    return "".join(_normalize_text(value).split())


def _strip_extension(value: str) -> str:
    text = _normalize_text(value).strip()
    if "." not in text:
        return text
    return text.rsplit(".", 1)[0]


def _extract_quoted_anchors(question: str) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()
    for raw in _QUOTED_ANCHOR_PATTERN.findall(question):
        anchor = _normalize_text(raw).strip()
        if not anchor or anchor in seen:
            continue
        anchors.append(anchor)
        seen.add(anchor)
    return anchors


def _meaningful_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_PATTERN.findall(_normalize_text(value).lower()):
        normalized = token.strip()
        if len(normalized) <= 1:
            continue
        if normalized in _GENERIC_ANCHOR_TOKENS:
            continue
        tokens.add(normalized)
    return tokens


def _bigram_overlap(anchor: str, candidate: str) -> float:
    if len(anchor) < 2 or len(candidate) < 2:
        return 0.0
    anchor_grams = {anchor[idx : idx + 2] for idx in range(len(anchor) - 1)}
    candidate_grams = {candidate[idx : idx + 2] for idx in range(len(candidate) - 1)}
    if not anchor_grams or not candidate_grams:
        return 0.0
    return len(anchor_grams & candidate_grams) / len(anchor_grams)


def _anchor_matches_text(anchor: str, candidate: str) -> bool:
    anchor_compact = _compact_text(anchor).lower()
    candidate_compact = _compact_text(_strip_extension(candidate)).lower()
    if not anchor_compact or not candidate_compact:
        return False
    if anchor_compact in candidate_compact or candidate_compact in anchor_compact:
        return True

    anchor_tokens = _meaningful_tokens(anchor)
    candidate_tokens = _meaningful_tokens(candidate)
    overlap_count = len(anchor_tokens & candidate_tokens)
    similarity = _bigram_overlap(anchor_compact, candidate_compact)

    if overlap_count >= 2:
        return True
    if overlap_count >= 1 and similarity >= 0.45:
        return True
    return similarity >= 0.72


@dataclass(slots=True)
class CalculationAnswer:
    """Deterministic answer generated from structured facts."""

    mode: str
    answer: str
    facts: list[CalculationFact]


@dataclass(slots=True)
class ResolvedBudgetFact:
    """Budget fact resolved either from structured metadata or explicit chunk labels."""

    fact: CalculationFact
    amount: float
    label: str | None = None


def _format_won(amount: float | None) -> str:
    if amount is None:
        return "-"
    return f"{int(round(amount)):,}원"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}배"


def _format_days(days: float | None) -> str:
    if days is None:
        return "-"
    if abs(days - round(days)) < 1e-9:
        return f"{int(round(days))}일"
    return f"{days:.1f}일"


def _citation_map(chunks: list[RetrievedChunk]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, item in enumerate(chunks, start=1):
        candidates = {
            str(item.chunk.doc_id),
            str(item.chunk.metadata.get("파일명", "")),
            str(item.chunk.metadata.get("ingest_file", "")),
            str(item.chunk.metadata.get("canonical_file", "")),
            str(item.chunk.metadata.get("공고 번호", "")),
        }
        for candidate in candidates:
            if candidate and candidate not in mapping:
                mapping[candidate] = idx
    return mapping


def _compose_calculation_answer(
    summary_lines: list[str],
    evidence_lines: list[str],
) -> str:
    answer = "핵심 답변:\n" + "\n".join(summary_lines)
    if evidence_lines:
        answer += "\n\n계산 근거:\n" + "\n".join(evidence_lines)
    return answer


def build_calculation_generation_result(
    *,
    question: str,
    calculation_answer: CalculationAnswer,
    context_chunks: list[RetrievedChunk],
    llm_provider: str,
    llm_model: str,
    generation_config: dict,
    latency_ms: float = 0.0,
) -> GenerationResult:
    return GenerationResult(
        question_id=generation_config.get("question_id", f"q-{uuid4().hex[:8]}"),
        question=question,
        scenario=generation_config.get("scenario", "ad-hoc"),
        run_id=generation_config.get("run_id", f"run-{uuid4().hex[:8]}"),
        embedding_provider=generation_config.get("embedding_provider", ""),
        embedding_model=generation_config.get("embedding_model", ""),
        llm_provider=llm_provider,
        llm_model=llm_model,
        answer=calculation_answer.answer,
        retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in context_chunks],
        retrieved_doc_ids=[chunk.chunk.doc_id for chunk in context_chunks],
        retrieved_chunks=context_chunks,
        latency_ms=round(latency_ms, 1),
        token_usage={"prompt": 0, "completion": 0, "cached": 0, "total": 0},
        cost_usd=0.0,
        debug={"calculation_mode": calculation_answer.mode},
        context="",
    )


class CalculationEngine:
    """Structured calculator backed by CalculationStore."""

    def __init__(self, store: CalculationStore) -> None:
        self.store = store

    def try_answer(
        self,
        *,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        metadata_filter: dict | None = None,
    ) -> CalculationAnswer | None:
        facts = self._resolve_facts(question, retrieved_chunks, metadata_filter)
        if not facts:
            return None

        normalized = question.replace(" ", "")
        budget_facts = self._resolve_budget_facts(question, facts, retrieved_chunks)

        if self._is_budget_difference(question, normalized) and len(budget_facts) >= 2:
            return self._budget_difference_answer(budget_facts[:2], retrieved_chunks)
        if self._is_budget_sum(question, normalized) and len(budget_facts) >= 2:
            return self._budget_sum_answer(budget_facts, retrieved_chunks)
        if self._is_budget_average(question, normalized) and len(budget_facts) >= 2:
            return self._budget_average_answer(budget_facts, retrieved_chunks)
        if self._is_budget_ratio(question, normalized) and len(budget_facts) >= 2:
            return self._budget_ratio_answer(budget_facts[:2], retrieved_chunks)
        if self._is_budget_max(question, normalized) and len(budget_facts) >= 2:
            return self._budget_extreme_answer(budget_facts, retrieved_chunks, largest=True)
        if self._is_budget_min(question, normalized) and len(budget_facts) >= 2:
            return self._budget_extreme_answer(budget_facts, retrieved_chunks, largest=False)
        if self._is_budget_ordering(question, normalized) and len(budget_facts) >= 2:
            descending = self._is_descending_order(question, normalized)
            return self._budget_ordering_answer(budget_facts, retrieved_chunks, descending=descending)

        if self._is_budget_single_lookup(question, normalized) and len(budget_facts) >= 1:
            return self._budget_single_answer(budget_facts[0], retrieved_chunks)

        percent_match = _PERCENT_PATTERN.search(question)
        if percent_match and self._is_budget_percent_apply(question, normalized) and len(budget_facts) >= 1:
            ratio = float(percent_match.group(1)) / 100.0
            return self._budget_percentage_answer(budget_facts[0], ratio, retrieved_chunks)

        if self._is_bid_window_days(question, normalized) and len(facts) >= 1:
            return self._bid_window_days_answer(facts[0], retrieved_chunks)

        return None

    def _resolve_facts(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        metadata_filter: dict | None,
    ) -> list[CalculationFact]:
        facts: list[CalculationFact] = []
        seen_keys: set[str] = set()

        candidate_ids: list[str] = []
        if metadata_filter:
            for value in metadata_filter.values():
                if isinstance(value, dict) and "$in" in value and isinstance(value["$in"], list):
                    candidate_ids.extend(str(item) for item in value["$in"] if item)
                elif isinstance(value, str):
                    candidate_ids.append(value)

        for item in retrieved_chunks:
            candidate_ids.extend(
                [
                    str(item.chunk.doc_id),
                    str(item.chunk.metadata.get("파일명", "")),
                    str(item.chunk.metadata.get("ingest_file", "")),
                    str(item.chunk.metadata.get("canonical_file", "")),
                    str(item.chunk.metadata.get("공고 번호", "")),
                ]
            )

        for candidate in candidate_ids:
            if not candidate:
                continue
            fact = self.store.get_fact(candidate)
            if fact is None or fact.document_key in seen_keys:
                continue
            facts.append(fact)
            seen_keys.add(fact.document_key)

        anchors = _extract_quoted_anchors(question)
        if anchors:
            anchored_facts = [
                fact for fact in facts if any(self._fact_matches_anchor(fact, anchor) for anchor in anchors)
            ]
            if anchored_facts:
                facts = anchored_facts

        return facts

    def _fact_matches_anchor(self, fact: CalculationFact, anchor: str) -> bool:
        candidates = (
            fact.title,
            fact.file_name,
            fact.ingest_file,
            fact.canonical_file,
            fact.agency,
            fact.resolved_agency,
            fact.original_agency,
        )
        return any(_anchor_matches_text(anchor, candidate) for candidate in candidates if candidate)

    def _resolve_budget_facts(
        self,
        question: str,
        facts: list[CalculationFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[ResolvedBudgetFact]:
        explicit_kind = self._detect_explicit_budget_kind(question)
        if explicit_kind is None:
            return [
                ResolvedBudgetFact(fact=fact, amount=fact.budget_amount, label=None)
                for fact in facts
                if fact.budget_amount is not None
            ]

        kind_key, label, variants = explicit_kind
        labeled_amounts = self._extract_labeled_amounts(retrieved_chunks, variants)
        resolved: list[ResolvedBudgetFact] = []

        for fact in facts:
            amount = fact.amount_for_kind(kind_key)
            if amount is not None:
                resolved.append(ResolvedBudgetFact(fact=fact, amount=amount, label=label))
                continue
            for candidate in self._fact_candidate_keys(fact):
                amount = labeled_amounts.get(candidate)
                if amount is not None:
                    break
            if amount is None:
                continue
            resolved.append(ResolvedBudgetFact(fact=fact, amount=amount, label=label))

        return resolved

    def _detect_explicit_budget_kind(self, question: str) -> tuple[str, str, tuple[str, ...]] | None:
        normalized_question = " ".join(_normalize_text(question).split())
        for kind_key, label, variants in _BUDGET_KIND_SPECS:
            if any(" ".join(_normalize_text(variant).split()) in normalized_question for variant in variants):
                return kind_key, label, variants
        return None

    def _extract_labeled_amounts(
        self,
        retrieved_chunks: list[RetrievedChunk],
        variants: tuple[str, ...],
    ) -> dict[str, float]:
        amount_by_key: dict[str, float] = {}
        for item in retrieved_chunks:
            amount = self._extract_amount_from_chunk(item, variants)
            if amount is None:
                continue
            for key in self._chunk_candidate_keys(item):
                amount_by_key.setdefault(key, amount)
        return amount_by_key

    def _extract_amount_from_chunk(
        self,
        item: RetrievedChunk,
        variants: tuple[str, ...],
    ) -> float | None:
        for text in (item.chunk.text_with_meta, item.chunk.text):
            amount = self._extract_labeled_amount(text, variants)
            if amount is not None:
                return amount
        return None

    def _extract_labeled_amount(self, text: str, variants: tuple[str, ...]) -> float | None:
        normalized = _normalize_text(text)
        for line in normalized.splitlines():
            amount = self._extract_amount_from_text_window(line, variants)
            if amount is not None:
                return amount
        return self._extract_amount_from_text_window(normalized, variants)

    def _extract_amount_from_text_window(self, text: str, variants: tuple[str, ...]) -> float | None:
        for variant in variants:
            for match in re.finditer(re.escape(variant), text):
                window = text[match.start() : match.start() + 120]
                for amount_match in _AMOUNT_PATTERN.finditer(window):
                    digits_only = amount_match.group(0).replace(",", "")
                    if len(digits_only) < 5:
                        continue
                    return float(digits_only)
        return None

    def _chunk_candidate_keys(self, item: RetrievedChunk) -> set[str]:
        return {
            _compact_text(str(item.chunk.doc_id)),
            _compact_text(str(item.chunk.metadata.get("파일명", ""))),
            _compact_text(str(item.chunk.metadata.get("ingest_file", ""))),
            _compact_text(str(item.chunk.metadata.get("canonical_file", ""))),
            _compact_text(str(item.chunk.metadata.get("공고 번호", ""))),
        } - {""}

    def _fact_candidate_keys(self, fact: CalculationFact) -> set[str]:
        return {
            _compact_text(fact.document_key),
            _compact_text(fact.file_name),
            _compact_text(fact.ingest_file),
            _compact_text(fact.canonical_file),
            _compact_text(fact.notice_id or ""),
        } - {""}

    def _contains_any(self, normalized: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.replace(" ", "") in normalized for keyword in keywords)

    def _has_budget_signal(self, question: str) -> bool:
        if any(keyword in question for keyword in _BUDGET_KEYWORDS):
            return True
        return self._detect_explicit_budget_kind(question) is not None

    def _budget_label(self, facts: list[ResolvedBudgetFact]) -> str:
        for fact in facts:
            if fact.label:
                return fact.label
        return "예산"

    def _is_budget_difference(self, question: str, normalized: str) -> bool:
        return self._has_budget_signal(question) and (
            "차이" in question or "차액" in question or self._contains_any(normalized, _DIFF_KEYWORDS)
        )

    def _is_budget_sum(self, question: str, normalized: str) -> bool:
        return self._has_budget_signal(question) and self._contains_any(
            normalized, _SUM_KEYWORDS
        )

    def _is_budget_average(self, question: str, normalized: str) -> bool:
        return self._has_budget_signal(question) and self._contains_any(
            normalized, _AVG_KEYWORDS
        )

    def _is_budget_ratio(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        return self._contains_any(normalized, _RATIO_KEYWORDS)

    def _is_budget_max(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        return self._contains_any(normalized, _MAX_KEYWORDS)

    def _is_budget_min(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        return self._contains_any(normalized, _MIN_KEYWORDS)

    def _is_budget_ordering(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        return self._contains_any(
            normalized,
            _ORDER_ASC_KEYWORDS + _ORDER_DESC_KEYWORDS + _ORDER_GENERIC_KEYWORDS,
        )

    def _is_descending_order(self, question: str, normalized: str) -> bool:
        if self._contains_any(normalized, _ORDER_ASC_KEYWORDS):
            return False
        if self._contains_any(normalized, _ORDER_DESC_KEYWORDS):
            return True
        return True

    def _is_budget_percent_apply(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        return any(token in normalized for token in ("가정", "사용", "책정", "해당금액", "몇원", "얼마"))

    def _is_budget_single_lookup(self, question: str, normalized: str) -> bool:
        if not self._has_budget_signal(question):
            return False
        if _PERCENT_PATTERN.search(question):
            return False
        if self._contains_any(normalized, _RATIO_KEYWORDS):
            return False
        if self._contains_any(normalized, _BUDGET_SINGLE_BLOCKERS):
            return False
        return any(token in normalized for token in ("얼마", "몇원", "가격"))

    def _is_bid_window_days(self, question: str, normalized: str) -> bool:
        has_bid_signal = "입찰" in question or "입찰" in normalized or "참여" in normalized
        if not has_bid_signal:
            return False
        if self._contains_any(normalized, _BID_WINDOW_BLOCKERS):
            return False
        return self._contains_any(normalized, _BID_WINDOW_KEYWORDS)

    def _budget_difference_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        left, right = facts
        citations = _citation_map(retrieved_chunks)
        left_cite = citations.get(left.fact.document_key) or citations.get(left.fact.file_name) or 1
        right_cite = citations.get(right.fact.document_key) or citations.get(right.fact.file_name) or 2
        difference = left.amount - right.amount
        label = self._budget_label(facts)
        answer = _compose_calculation_answer(
            [
                f"- 두 사업의 {label} 차액은 {_format_won(abs(difference))}이며, "
                f"{left.fact.title if difference >= 0 else right.fact.title} 쪽이 더 큽니다.",
            ],
            [
                f"- {left.fact.title}: {label} {_format_won(left.amount)} [{left_cite}]",
                f"- {right.fact.title}: {label} {_format_won(right.amount)} [{right_cite}]",
            ],
        )
        return CalculationAnswer(mode="budget_difference", answer=answer, facts=[left.fact, right.fact])

    def _budget_sum_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        if len(facts) < 2:
            return None

        citations = _citation_map(retrieved_chunks)
        label = self._budget_label(facts)
        total = sum(fact.amount for fact in facts)
        detail_lines = []
        for fact in facts:
            cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
            detail_lines.append(f"- {fact.fact.title}: {_format_won(fact.amount)} [{cite}]")

        answer = _compose_calculation_answer(
            [f"- 조회한 사업들의 {label} 합계는 {_format_won(total)}입니다."],
            detail_lines,
        )
        return CalculationAnswer(mode="budget_sum", answer=answer, facts=[fact.fact for fact in facts])

    def _budget_average_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        if len(facts) < 2:
            return None

        citations = _citation_map(retrieved_chunks)
        label = self._budget_label(facts)
        avg = sum(fact.amount for fact in facts) / len(facts)
        detail_lines = []
        for fact in facts:
            cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
            detail_lines.append(f"- {fact.fact.title}: {_format_won(fact.amount)} [{cite}]")

        answer = _compose_calculation_answer(
            [f"- 조회한 사업들의 평균 {label}은 {_format_won(avg)}입니다."],
            detail_lines,
        )
        return CalculationAnswer(mode="budget_average", answer=answer, facts=[fact.fact for fact in facts])

    def _budget_ratio_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        left, right = facts
        if right.amount == 0:
            return None
        citations = _citation_map(retrieved_chunks)
        left_cite = citations.get(left.fact.document_key) or citations.get(left.fact.file_name) or 1
        right_cite = citations.get(right.fact.document_key) or citations.get(right.fact.file_name) or 2
        ratio = left.amount / right.amount
        label = self._budget_label(facts)

        answer = _compose_calculation_answer(
            [f"- {left.fact.title}의 {label}은 {right.fact.title}의 {label} 대비 {_format_ratio(ratio)}입니다."],
            [
                f"- {left.fact.title}: {_format_won(left.amount)} [{left_cite}]",
                f"- {right.fact.title}: {_format_won(right.amount)} [{right_cite}]",
            ],
        )
        return CalculationAnswer(mode="budget_ratio", answer=answer, facts=[left.fact, right.fact])

    def _budget_extreme_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
        *,
        largest: bool,
    ) -> CalculationAnswer | None:
        if len(facts) < 2:
            return None

        selected = max(facts, key=lambda fact: fact.amount) if largest else min(facts, key=lambda fact: fact.amount)
        citations = _citation_map(retrieved_chunks)
        cite = citations.get(selected.fact.document_key) or citations.get(selected.fact.file_name) or 1
        label = self._budget_label(facts)
        title = "가장 큰 사업" if largest else "가장 작은 사업"

        detail_lines = []
        for fact in sorted(facts, key=lambda item: item.amount, reverse=True):
            fact_cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
            detail_lines.append(f"- {fact.fact.title}: {_format_won(fact.amount)} [{fact_cite}]")

        answer = _compose_calculation_answer(
            [f"- {title}은 {selected.fact.title}이고, {label}은 {_format_won(selected.amount)}입니다[{cite}]."],
            detail_lines,
        )
        return CalculationAnswer(
            mode="budget_max" if largest else "budget_min",
            answer=answer,
            facts=[fact.fact for fact in facts],
        )

    def _budget_ordering_answer(
        self,
        facts: list[ResolvedBudgetFact],
        retrieved_chunks: list[RetrievedChunk],
        *,
        descending: bool,
    ) -> CalculationAnswer | None:
        if len(facts) < 2:
            return None

        ordered = sorted(facts, key=lambda fact: fact.amount, reverse=descending)
        citations = _citation_map(retrieved_chunks)
        direction = "큰 순서" if descending else "작은 순서"
        label = self._budget_label(facts)
        lines: list[str] = []
        for idx, fact in enumerate(ordered, start=1):
            cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
            lines.append(f"- {idx}. {fact.fact.title}: {_format_won(fact.amount)} [{cite}]")

        answer = _compose_calculation_answer(
            [f"- {label} 기준 {direction}대로 정렬하면 다음과 같습니다."] + lines,
            lines,
        )
        return CalculationAnswer(
            mode="budget_order_desc" if descending else "budget_order_asc",
            answer=answer,
            facts=[fact.fact for fact in ordered],
        )

    def _budget_percentage_answer(
        self,
        fact: ResolvedBudgetFact,
        ratio: float,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        citations = _citation_map(retrieved_chunks)
        cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
        computed = fact.amount * ratio
        percent_text = f"{ratio * 100:g}%"
        label = fact.label or "예산"

        answer = _compose_calculation_answer(
            [
                f"- {fact.fact.title}의 전체 {label} {_format_won(fact.amount)} 중 "
                f"{percent_text}는 {_format_won(computed)}입니다[{cite}]."
            ],
            [f"- {fact.fact.title}: {label} {_format_won(fact.amount)} [{cite}]"],
        )
        return CalculationAnswer(mode="budget_percentage", answer=answer, facts=[fact.fact])

    def _budget_single_answer(
        self,
        fact: ResolvedBudgetFact,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        citations = _citation_map(retrieved_chunks)
        cite = citations.get(fact.fact.document_key) or citations.get(fact.fact.file_name) or 1
        label = fact.label or "예산"

        answer = _compose_calculation_answer(
            [f"- {fact.fact.title}의 {label}은 {_format_won(fact.amount)}입니다[{cite}]."],
            [f"- {fact.fact.title}: {label} {_format_won(fact.amount)} [{cite}]"],
        )
        return CalculationAnswer(mode="budget_single", answer=answer, facts=[fact.fact])

    def _bid_window_days_answer(
        self,
        fact: CalculationFact,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CalculationAnswer | None:
        if fact.bid_window_days is None:
            return None
        citations = _citation_map(retrieved_chunks)
        cite = citations.get(fact.document_key) or citations.get(fact.file_name) or 1

        answer = _compose_calculation_answer(
            [f"- 입찰 참여 기간은 {_format_days(fact.bid_window_days)}입니다[{cite}]."],
            [
                f"- 입찰 참여 시작: {fact.bid_start_at or '-'} [{cite}]",
                f"- 입찰 참여 마감: {fact.bid_end_at or '-'} [{cite}]",
            ],
        )
        return CalculationAnswer(mode="bid_window_days", answer=answer, facts=[fact])
