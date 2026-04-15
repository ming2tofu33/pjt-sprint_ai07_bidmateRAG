"""LLM-as-judge v2 for evidence-first RAG evaluation.

This module keeps v1 judge intact and introduces a separate v2 path.
v2 adds:
- answer_correctness (ground-truth based)
- evidence payload for traceability
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing

logger = logging.getLogger(__name__)


JUDGE_V2_SYSTEM_PROMPT = """당신은 RAG 시스템 평가 심사자입니다.
질문, 검색 컨텍스트, 생성 답변, (선택) 정답을 보고 아래 항목을 평가하세요.

중요:
1) 반드시 JSON만 출력합니다.
2) evidence 필드를 반드시 채웁니다.
3) 점수는 계산하지 말고, 라벨링 결과만 출력합니다.

출력 스키마(키 이름 고정):
{
  "reasoning": "한국어 1~3문장",
  "evidence": {
    "claims": [
      {
        "id": "c1",
        "text": "답변의 핵심 주장",
        "supported_context_ids": [1, 3],
        "is_supported": true
      }
    ],
    "required_items": [
      {"id": "r1", "item": "질문이 요구한 항목", "is_answered": true}
    ],
    "gt_facts": [
      {
        "id": "g1",
        "fact": "정답 핵심 사실",
        "supported_context_ids": [2],
        "is_covered": true,
        "is_matched": true
      }
    ],
    "contexts": [
      {"context_id": 1, "is_relevant": true, "relevance_reason_short": "질문 핵심 근거 포함"}
    ],
    "missing_facts": ["컨텍스트에 없는 정답 사실"]
  }
}
"""


@dataclass
class JudgeScoresV2:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    reasoning: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMJudgeV2:
    """Separate v2 judge implementation with evidence schema."""

    METRIC_KEYS = (
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "answer_correctness",
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )
        self.pricing = load_pricing()
        self.cumulative_cost_usd: float = 0.0
        self.cumulative_tokens: int = 0

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None = None,
    ) -> JudgeScoresV2:
        """Score one sample. Never raises; returns parse/call error in payload."""
        user_prompt = self._build_user_prompt(question, answer, contexts, expected_answer)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_V2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge v2 LLM call failed: %s", exc)
            return JudgeScoresV2(error=f"llm_call_failed: {exc}")

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        self.cumulative_tokens += prompt_tokens + completion_tokens
        self.cumulative_cost_usd += calc_llm_cost(
            self.model,
            prompt_tokens,
            completion_tokens,
            self.pricing,
            cached_tokens=cached_tokens,
        )

        raw_text = response.choices[0].message.content or ""
        return self._parse_scores(raw_text)

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None,
    ) -> str:
        context_block = "\n\n---\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        ) or "(검색된 context 없음)"
        expected_block = (
            f"\n\n[Expected Answer]\n{expected_answer}" if expected_answer else ""
        )
        return (
            f"[Question]\n{question}\n\n"
            f"[Generated Answer]\n{answer}\n\n"
            f"[Retrieved Contexts]\n{context_block}"
            f"{expected_block}"
        )

    def _parse_scores(self, raw_text: str) -> JudgeScoresV2:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Judge v2 response not valid JSON: %s", exc)
            return JudgeScoresV2(error=f"json_parse_failed: {raw_text[:200]}")

        evidence = self._normalize_evidence(data.get("evidence"))
        scores = JudgeScoresV2(
            reasoning=str(data.get("reasoning", ""))[:1000],
            evidence=evidence,
        )

        # v2 principle: LLM performs labeling only, final scores are computed in code.
        # Keep model-provided scores (if any) for debugging, but do not use as final values.
        model_scores = {}
        for key in self.METRIC_KEYS:
            try:
                model_scores[key] = self._clamp_01(float(data.get(key, 0.0)))
            except (TypeError, ValueError):
                model_scores[key] = 0.0

        computed = self._compute_scores_from_evidence(evidence)
        scores.faithfulness = computed["faithfulness"]
        scores.answer_relevance = computed["answer_relevance"]
        scores.context_precision = computed["context_precision"]
        scores.context_recall = computed["context_recall"]
        scores.answer_correctness = computed["answer_correctness"]
        scores.evidence["score_source"] = "code_from_evidence"
        scores.evidence["model_scores"] = model_scores
        return scores

    def _normalize_evidence(self, evidence: Any) -> dict[str, Any]:
        """Best-effort evidence normalization with safe defaults."""
        if not isinstance(evidence, dict):
            return {
                "claims": [],
                "required_items": [],
                "gt_facts": [],
                "contexts": [],
                "missing_facts": [],
            }

        claims = evidence.get("claims", [])
        required_items = evidence.get("required_items", [])
        gt_facts = evidence.get("gt_facts", [])
        contexts = evidence.get("contexts", [])
        missing_facts = evidence.get("missing_facts", [])

        return {
            "claims": claims if isinstance(claims, list) else [],
            "required_items": required_items if isinstance(required_items, list) else [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "contexts": contexts if isinstance(contexts, list) else [],
            "missing_facts": missing_facts if isinstance(missing_facts, list) else [],
        }

    def _compute_scores_from_evidence(self, evidence: dict[str, Any]) -> dict[str, float]:
        claims = evidence.get("claims", [])
        required_items = evidence.get("required_items", [])
        gt_facts = evidence.get("gt_facts", [])
        contexts = evidence.get("contexts", [])

        faithfulness = self._ratio_from_flag(claims, "is_supported")
        answer_relevance = self._ratio_from_flag(required_items, "is_answered")
        context_precision = self._ratio_from_flag(contexts, "is_relevant")
        context_recall = self._ratio_from_flag(gt_facts, "is_covered")
        answer_correctness = self._ratio_from_flag(gt_facts, "is_matched")

        return {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

    def _ratio_from_flag(self, items: Any, flag_key: str) -> float:
        if not isinstance(items, list) or not items:
            return 0.0
        valid_items = [item for item in items if isinstance(item, dict)]
        if not valid_items:
            return 0.0
        matched = sum(1 for item in valid_items if self._as_bool(item.get(flag_key)))
        return self._clamp_01(matched / len(valid_items))

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))
