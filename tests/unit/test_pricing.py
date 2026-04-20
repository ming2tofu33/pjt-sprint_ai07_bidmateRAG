"""Tests for tracking/pricing.py."""

from __future__ import annotations

from bidmate_rag.tracking.pricing import (
    calc_embedding_cost,
    calc_llm_cost,
    is_model_priced,
    load_pricing,
    normalize_run_costs,
    resolve_llm_cost,
)

PRICING_FIXTURE = {
    "llm": {
        "test-model": {"input_per_1m": 1.00, "output_per_1m": 4.00},
    },
    "embedding": {
        "test-embed": {"per_1m": 0.10},
    },
}


def test_calc_llm_cost_uses_separate_input_output_rates():
    cost = calc_llm_cost("test-model", prompt_tokens=1_000_000, completion_tokens=0, pricing=PRICING_FIXTURE)
    assert cost == 1.0
    cost = calc_llm_cost("test-model", prompt_tokens=0, completion_tokens=1_000_000, pricing=PRICING_FIXTURE)
    assert cost == 4.0


def test_calc_llm_cost_partial_tokens():
    # 1k prompt + 500 completion → 1000*1 + 500*4 = 3000 micro-USD = 0.003
    cost = calc_llm_cost("test-model", 1_000, 500, PRICING_FIXTURE)
    assert cost == 0.003


def test_calc_llm_cost_unknown_model_returns_zero():
    cost = calc_llm_cost("unknown-model", 1_000, 500, PRICING_FIXTURE)
    assert cost == 0.0


def test_calc_embedding_cost():
    cost = calc_embedding_cost("test-embed", 1_000_000, PRICING_FIXTURE)
    assert cost == 0.1


def test_calc_embedding_cost_unknown_model_returns_zero():
    cost = calc_embedding_cost("nope", 1_000_000, PRICING_FIXTURE)
    assert cost == 0.0


def test_is_model_priced():
    assert is_model_priced("llm", "test-model", PRICING_FIXTURE) is True
    assert is_model_priced("llm", "missing", PRICING_FIXTURE) is False
    assert is_model_priced("embedding", "test-embed", PRICING_FIXTURE) is True


def test_load_pricing_returns_empty_when_missing(tmp_path):
    pricing = load_pricing(tmp_path / "missing.yaml")
    assert pricing == {"llm": {}, "embedding": {}}


def test_load_pricing_reads_yaml(tmp_path):
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(
        "llm:\n  foo:\n    input_per_1m: 0.5\n    output_per_1m: 1.5\n",
        encoding="utf-8",
    )
    pricing = load_pricing(pricing_file)
    assert pricing["llm"]["foo"]["input_per_1m"] == 0.5
    assert pricing["embedding"] == {}  # default added


# ---------------------------------------------------------------------------
# cached_tokens 처리 (OpenAI prompt_tokens_details.cached_tokens)
# ---------------------------------------------------------------------------


CACHED_PRICING = {
    "llm": {
        "x": {"input_per_1m": 1.0, "output_per_1m": 4.0, "cached_input_per_1m": 0.1},
        "no_cached": {"input_per_1m": 1.0, "output_per_1m": 4.0},
    }
}


def test_calc_llm_cost_cached_input_uses_discount():
    # 1000 prompt 중 800 cached → uncached 200 * 1.0 + cached 800 * 0.1 + completion 0
    # = 200 + 80 = 280 micro-USD
    cost = calc_llm_cost("x", 1000, 0, CACHED_PRICING, cached_tokens=800)
    assert cost == round(280 / 1_000_000, 6)


def test_calc_llm_cost_cached_zero_equals_no_cached_arg():
    a = calc_llm_cost("x", 1000, 500, CACHED_PRICING)
    b = calc_llm_cost("x", 1000, 500, CACHED_PRICING, cached_tokens=0)
    assert a == b


def test_calc_llm_cost_cached_falls_back_to_input_rate_when_unset():
    # cached_input_per_1m이 없는 모델은 cached_tokens도 일반 input 단가로 계산
    cost_no_cache = calc_llm_cost("no_cached", 1000, 0, CACHED_PRICING, cached_tokens=0)
    cost_with_cache = calc_llm_cost("no_cached", 1000, 0, CACHED_PRICING, cached_tokens=500)
    assert cost_no_cache == cost_with_cache


def test_resolve_llm_cost_recomputes_when_existing_cost_missing():
    cost = resolve_llm_cost(
        "test-model",
        prompt_tokens=1_000,
        completion_tokens=500,
        pricing=PRICING_FIXTURE,
        existing_cost=0.0,
    )
    assert cost == 0.003


def test_resolve_llm_cost_preserves_existing_cost_when_present():
    cost = resolve_llm_cost(
        "test-model",
        prompt_tokens=1_000,
        completion_tokens=500,
        pricing=PRICING_FIXTURE,
        existing_cost=0.123456,
    )
    assert cost == 0.123456


def test_normalize_run_costs_backfills_generation_and_rewrite():
    costs = normalize_run_costs(
        llm_model="test-model",
        pricing=PRICING_FIXTURE,
        generation_cost_usd=0.0,
        rewrite_cost_usd=0.0,
        prompt_tokens=1_000,
        completion_tokens=500,
        rewrite_prompt_tokens=500,
        rewrite_completion_tokens=250,
        judge_cost_usd=0.1,
    )
    assert costs["generation_cost_usd"] == 0.003
    assert costs["rewrite_cost_usd"] == 0.0015
    assert costs["judge_cost_usd"] == 0.1
    assert costs["total_cost_usd"] == 0.1045
