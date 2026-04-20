"""generated2 yaml 프롬프트 컴포넌트 조합 모듈."""

from __future__ import annotations
from pathlib import Path
import yaml


def build_system_prompt_from_components(config_path: str | Path) -> str:
    """generated2 yaml의 컴포넌트들을 조합해서 system_prompt 생성."""
    config = yaml.safe_load(Path(config_path).read_text())

    parts = []

    if config.get("persona") and config["persona"].strip():
        parts.append(config["persona"].strip())

    if config.get("constraints"):
        for c in config["constraints"]:
            if c and str(c).strip():
                parts.append(str(c).strip())

    if config.get("output_format"):
        for f in config["output_format"]:
            if f and str(f).strip():
                parts.append(str(f).strip())

    if config.get("few_shot_examples"):
        for s in config["few_shot_examples"]:
            if s and str(s).strip():
                parts.append(str(s).strip())

    if config.get("thought_trigger"):
        for t in config["thought_trigger"]:
            if t and str(t).strip():
                parts.append(str(t).strip())

    return "\n\n".join(parts)


def is_component_config(config_path: str | Path) -> bool:
    """generated2 형식의 yaml인지 확인."""
    try:
        config = yaml.safe_load(Path(config_path).read_text())
        return "00_metadata" in config
    except Exception:
        return False
