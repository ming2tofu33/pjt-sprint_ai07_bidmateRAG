"""Unit tests for slash command registry."""

from __future__ import annotations

import pytest

from bidmate_rag.web_api.commands import COMMAND_REGISTRY, SlashCommand


def test_registry_has_twelve_commands() -> None:
    expected = {
        "요약", "요구사항", "일정", "예산", "비교",
        "자격요건", "평가기준", "리스크", "기본정보",
        "제출서류", "도움말", "초기화",
    }
    assert set(COMMAND_REGISTRY.keys()) == expected


def test_every_command_has_label_and_description() -> None:
    for key, cmd in COMMAND_REGISTRY.items():
        assert cmd.id == key
        assert cmd.label.startswith("/")
        assert cmd.description
        assert cmd.icon


def test_compare_command_requires_multi_doc() -> None:
    cmd = COMMAND_REGISTRY["비교"]
    assert cmd.requires_doc is True
    assert cmd.requires_multi_doc is True


def test_static_commands_have_payload() -> None:
    for key in ("도움말", "초기화"):
        cmd = COMMAND_REGISTRY[key]
        assert cmd.static_response is True
        assert cmd.static_payload is not None


def test_non_static_commands_have_prompt() -> None:
    for key, cmd in COMMAND_REGISTRY.items():
        if cmd.static_response:
            continue
        assert cmd.system_prompt
        assert cmd.query_augmentation or key == "비교"


def test_top_k_is_positive() -> None:
    for cmd in COMMAND_REGISTRY.values():
        if cmd.static_response:
            continue
        assert cmd.top_k > 0
