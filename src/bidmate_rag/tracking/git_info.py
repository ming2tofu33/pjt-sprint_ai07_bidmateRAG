"""Capture git commit/branch info for experiment metadata.

실험 실행 시 현재 git 상태(커밋 해시, 브랜치명, 변경 여부)를
메타데이터에 기록하여 재현성을 보장한다.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def capture_git_info() -> dict[str, Any]:
    """현재 git 커밋, 브랜치, dirty 상태를 반환한다.

    Returns:
        commit, commit_short, branch, dirty 키를 포함하는 딕셔너리.
        git이 없거나 저장소가 아닌 경우 기본값 반환.
    """
    try:
        commit = _git("rev-parse", "HEAD")              # 전체 커밋 해시
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")  # 현재 브랜치명
        status = _git("status", "--porcelain")           # 변경 사항 확인
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not capture git info: %s", exc)
        return {"commit": "unknown", "branch": "unknown", "dirty": False}
    return {
        "commit": commit,
        "commit_short": commit[:7],  # 짧은 해시 (7자)
        "branch": branch,
        "dirty": bool(status),       # 커밋되지 않은 변경이 있으면 True
    }


def _git(*args: str) -> str:
    """git 명령을 실행하고 stdout을 반환한다.

    Args:
        *args: git 하위 명령과 인자들.

    Returns:
        명령 실행 결과 문자열 (앞뒤 공백 제거).
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
