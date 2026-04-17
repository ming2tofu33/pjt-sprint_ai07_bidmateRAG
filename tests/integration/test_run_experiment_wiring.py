"""Integration tests for scripts/run_experiment.py wiring.

build_index → run_eval 사이에 같은 collection을 보는지를 검증합니다.
이전에 build_index가 --experiment-config를 못 받아서 full_rag 모드에서
collection mismatch가 발생하던 버그의 회귀를 방지합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import yaml

import scripts.run_experiment as run_experiment
from bidmate_rag.config.settings import (
    ExperimentConfig,
    ProjectConfig,
    ProviderConfig,
    RuntimeConfig,
)
from bidmate_rag.pipelines.runtime import collection_name_for_config


def _make_args(eval_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        experiment_config="<set per test>",
        eval_path=str(eval_path),
        skip_ingest=True,  # subprocess 호출 줄임
        skip_judge=True,
    )


def _full_rag_runtime() -> RuntimeConfig:
    """full_rag_compare + openai_gpt5mini 시나리오 (실제 yaml과 동일)."""
    return RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(
            provider="openai",
            model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
            collection_name="bidmate-scenario-b-text-embedding-3-small",
        ),
        experiment=ExperimentConfig(name="full-rag-compare", mode="full_rag"),
    )


def _ad_hoc_runtime() -> RuntimeConfig:
    """experiment-config 없이 build_index가 받는 default runtime."""
    return RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(
            provider="openai",
            model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
            collection_name="bidmate-scenario-b-text-embedding-3-small",
        ),
        experiment=ExperimentConfig(name="ad-hoc", mode="full_rag"),
    )


# ---------------------------------------------------------------------------
# 회귀 증명: 두 runtime의 collection_name이 다름
# ---------------------------------------------------------------------------


def test_full_rag_collection_diverges_when_experiment_config_missing():
    """full_rag 모드에서 experiment-config가 있을 때와 없을 때 collection_name이
    달라야 함. 이게 같으면 본 fix가 의미 없음."""
    with_exp = collection_name_for_config(_full_rag_runtime())
    without_exp = collection_name_for_config(_ad_hoc_runtime())
    assert with_exp != without_exp
    # 명시 prefix
    assert with_exp == "full-rag-compare-bidmate-scenario-b-text-embedding-3-small"
    assert without_exp == "bidmate-scenario-b-text-embedding-3-small"


# ---------------------------------------------------------------------------
# subprocess args 캡처: build_index가 --experiment-config를 받는가
# ---------------------------------------------------------------------------


def test_run_single_experiment_passes_experiment_config_to_build_index(tmp_path):
    """build_index 호출 시 --experiment-config 인자가 포함되어야 함.
    이전엔 누락되어 collection mismatch 발생."""
    captured: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        captured.append(list(cmd))
        return None

    eval_path = tmp_path / "fake_eval.csv"
    eval_path.write_text("id\n", encoding="utf-8")

    args = _make_args(eval_path)
    args.skip_ingest = True  # ingest는 캡처에서 제외
    args.skip_judge = True

    exp_cfg = {
        "name": "full-rag-compare",
        "mode": "full_rag",
        "provider_configs": ["configs/providers/openai_gpt5mini.yaml"],
    }
    experiment_config_path = tmp_path / "exp.yaml"
    experiment_config_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

    with patch("scripts.run_experiment.subprocess.run", side_effect=fake_run):
        run_experiment._run_single_experiment(
            exp_cfg,
            args,
            experiment_config_path=str(experiment_config_path),
        )

    # build_index 호출 찾기
    build_index_calls = [
        cmd for cmd in captured if any("build_index.py" in arg for arg in cmd)
    ]
    assert len(build_index_calls) >= 1, "build_index가 호출되지 않음"

    for cmd in build_index_calls:
        assert "--experiment-config" in cmd, (
            f"build_index 호출에 --experiment-config 누락: {cmd}"
        )
        idx = cmd.index("--experiment-config")
        assert cmd[idx + 1] == str(experiment_config_path)


def test_build_index_and_run_eval_receive_same_experiment_config(tmp_path):
    """build_index와 run_eval이 같은 --experiment-config 값을 받아야 같은
    collection_name으로 동작."""
    captured: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        captured.append(list(cmd))
        return None

    eval_path = tmp_path / "fake_eval.csv"
    eval_path.write_text("id\n", encoding="utf-8")

    args = _make_args(eval_path)
    exp_cfg = {
        "name": "full-rag-compare",
        "mode": "full_rag",
        "provider_configs": ["configs/providers/openai_gpt5mini.yaml"],
    }
    experiment_config_path = tmp_path / "exp.yaml"
    experiment_config_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

    with patch("scripts.run_experiment.subprocess.run", side_effect=fake_run):
        run_experiment._run_single_experiment(
            exp_cfg,
            args,
            experiment_config_path=str(experiment_config_path),
        )

    def _extract_exp_config(cmd: list[str]) -> str | None:
        if "--experiment-config" not in cmd:
            return None
        return cmd[cmd.index("--experiment-config") + 1]

    build_paths = {
        _extract_exp_config(c) for c in captured if any("build_index.py" in a for a in c)
    }
    eval_paths = {
        _extract_exp_config(c) for c in captured if any("run_eval.py" in a for a in c)
    }
    # 둘 다 같은 path를 받아야 함 (None 이면 누락)
    assert None not in build_paths, "build_index에 --experiment-config 누락"
    assert None not in eval_paths, "run_eval에 --experiment-config 누락"
    assert build_paths == eval_paths, (
        f"build_index와 run_eval의 experiment-config 불일치: "
        f"build={build_paths}, eval={eval_paths}"
    )


# ---------------------------------------------------------------------------
# Streamlit run_live_query 시그니처 + debug_tab wiring 검증
# ---------------------------------------------------------------------------


def test_run_live_query_exposes_metadata_filter_and_chat_history():
    """Streamlit debug_tab이 평가셋의 metadata_filter / history를 전달할 수
    있도록 시그니처에 노출되어야 함."""
    import inspect

    from app.api.routes import run_live_query

    sig = inspect.signature(run_live_query)
    params = set(sig.parameters.keys())
    assert "metadata_filter" in params
    assert "chat_history" in params


def test_debug_tab_passes_metadata_filter_and_history_to_run_live_query():
    """app/eval_ui.py의 _render_debug_tab이 selected_q의 metadata_filter와
    history를 run_live_query 호출에 전달해야 함 (소스 inspect)."""
    import inspect

    from app import eval_ui

    src = inspect.getsource(eval_ui._render_debug_tab)
    assert "normalize_metadata_filter" in src, (
        "debug_tab이 metadata_filter를 정규화해서 전달해야 함"
    )
    assert "metadata_filter=normalized_filter" in src
    assert "chat_history=history" in src


def test_streamlit_main_chat_history_builder_keeps_previous_turns_only():
    from app.main import _build_chat_history

    messages = [
        {"role": "user", "content": "국민연금공단 사업 알려줘"},
        {"role": "assistant", "content": "차세대 ERP 사업입니다.", "metadata": {"tokens": 12}},
        {"role": "assistant", "content": ""},
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": None},
    ]

    assert _build_chat_history(messages) == [
        {"role": "user", "content": "국민연금공단 사업 알려줘"},
        {"role": "assistant", "content": "차세대 ERP 사업입니다."},
    ]


def test_streamlit_main_chat_passes_history_to_run_live_query():
    import inspect

    from app import main

    src = inspect.getsource(main._render_streamlit_app)
    assert "chat_history = _build_chat_history" in src
    assert "chat_history=chat_history" in src
