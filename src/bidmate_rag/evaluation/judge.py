"""LLM-as-judge for RAG generation quality.

Computes 4 RAGAS-style metrics in a single LLM call per sample:
- faithfulness:      답변이 검색된 context에만 근거하는가? (hallucination 검출)
- answer_relevance:  답변이 질문에 직접적으로 답하는가?
- context_precision: 검색된 context가 답변에 실제로 사용된 비율
- context_recall:    expected_answer 정보가 검색된 context에 포함된 비율
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """당신은 RAG 시스템의 품질을 평가하는 전문 심사자입니다.
주어진 질문, 검색된 context, 생성된 답변을 보고 4가지 항목을 평가합니다.

각 항목은 다음 5단계 중 하나로만 평가하세요:
0.0 / 0.25 / 0.5 / 0.75 / 1.0

극단값(0.0 또는 1.0)은 명백히 그 정도일 때만 사용하고,
부분적으로 만족하면 0.25 / 0.5 / 0.75 같은 중간값을 적극 사용하세요.
대부분의 실제 답변은 중간값을 받습니다.

평가 항목과 척도:

1. faithfulness — 답변의 모든 주장이 context에 근거하는가?
   - 1.0: 모든 주장이 context로 검증 가능
   - 0.75: 대부분 근거 있음, 사소한 추론 1~2개만 외부 지식
   - 0.5: 절반 정도만 context 기반, 나머지는 일반 상식 또는 추론
   - 0.25: 일부만 근거 있고 대부분 외부 지식/추측
   - 0.0: context와 무관한 환각

2. answer_relevance — 답변이 질문에 직접 답하는가?
   - 1.0: 질문의 모든 부분에 정확히 답함
   - 0.75: 핵심에는 답하지만 일부 누락
   - 0.5: 질문의 절반 정도만 답함
   - 0.25: 주변적인 정보만 제공
   - 0.0: 질문과 무관한 답변

3. context_precision — 검색된 context 중 답변에 실제 사용된 비율
   - 1.0: 모든 검색 context가 답변 생성에 기여
   - 0.75: 대부분 사용됨, 1~2개만 무관
   - 0.5: 절반 정도가 답변에 사용
   - 0.25: 일부만 사용, 대부분 노이즈
   - 0.0: 검색 결과가 답변 생성에 전혀 도움 안 됨

4. context_recall — expected_answer가 주어졌다면, 그 정보가 context에 포함된 비율
   - 1.0: expected_answer의 모든 핵심 정보가 context에 있음
   - 0.75: 핵심 정보의 대부분이 있음
   - 0.5: 절반 정도만 있음
   - 0.25: 일부 단편만 있음
   - 0.0: expected_answer 정보가 context에 없음
   - expected_answer가 없으면 1.0으로 처리

먼저 짧게 reasoning을 적은 뒤 4개 점수를 매기세요.
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 금지:
{
  "reasoning": "각 항목의 점수 근거를 한국어로 2~3문장으로 설명",
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_precision": 0.0,
  "context_recall": 0.0
}
"""


@dataclass
class JudgeScores:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    reasoning: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMJudge:
    """Single-model judge that scores 4 RAG quality metrics in one call."""

    METRIC_KEYS = ("faithfulness", "answer_relevance", "context_precision", "context_recall")

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
    ) -> JudgeScores:
        """Score one (question, answer, contexts) sample. Never raises."""
        user_prompt = self._build_user_prompt(question, answer, contexts, expected_answer)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=800,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge LLM call failed: %s", exc)
            return JudgeScores(error=f"llm_call_failed: {exc}")

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

    def _parse_scores(self, raw_text: str) -> JudgeScores:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Judge response not valid JSON: %s", exc)
            return JudgeScores(error=f"json_parse_failed: {raw_text[:100]}")

        # reasoning 필드는 jsonl 파일 크기 폭주를 막기 위해 500자로 자름
        scores = JudgeScores(reasoning=str(data.get("reasoning", ""))[:500])
        for key in self.METRIC_KEYS:
            value = data.get(key, 0.0)
            try:
                setattr(scores, key, max(0.0, min(1.0, float(value))))
            except (TypeError, ValueError):
                setattr(scores, key, 0.0)
        return scores
