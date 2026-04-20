"""Matrix grid search expansion for experiment yaml.

`configs/experiments/*.yaml`의 ``matrix:`` 섹션을 카르테시안 곱으로 펼쳐
N개의 sub-experiment 정의로 확장합니다. 사용자가 yaml 파일을 N개 손으로
만드는 부담을 없애고, ``run_experiment.py``가 자동으로 순차 실행할 수 있게
합니다.

예::

    matrix:
      chunk_size: [500, 1000]
      retrieval_top_k: [3, 5, 8]

→ 6개 cell:
   - chunk_size500_retrieval_top_k3
   - chunk_size500_retrieval_top_k5
   - chunk_size500_retrieval_top_k8
   - chunk_size1000_retrieval_top_k3
   - chunk_size1000_retrieval_top_k5
   - chunk_size1000_retrieval_top_k8
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass
class MatrixCell:
    """매트릭스의 한 조합 — 실험 이름 suffix + override할 파라미터들."""

    name_suffix: str
    overrides: dict[str, Any]


def expand_matrix(matrix: dict[str, list[Any]]) -> list[MatrixCell]:
    """yaml ``matrix:`` 섹션을 카르테시안 곱으로 펼친다.

    Args:
        matrix: 키는 ExperimentConfig 필드명, 값은 후보 값 리스트.

    Returns:
        모든 조합의 ``MatrixCell`` 리스트. 빈 dict이면 빈 리스트 반환
        (caller가 단일 실행으로 fallback하도록).

    키 순서는 ``sorted()``로 결정론적이며, suffix는 ``{key}{value}_...`` 형식.
    """
    if not matrix:
        return []
    keys = sorted(matrix.keys())
    value_lists = [matrix[k] for k in keys]
    cells: list[MatrixCell] = []
    for combo in product(*value_lists):
        overrides = dict(zip(keys, combo))
        suffix = "_".join(f"{k}{v}" for k, v in overrides.items())
        cells.append(MatrixCell(name_suffix=suffix, overrides=overrides))
    return cells


def apply_overrides_to_yaml_dict(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """기존 experiment yaml dict에 overrides를 병합한 새 dict 반환.

    base는 변형하지 않음 (얕은 복사 후 갱신).
    """
    return {**base, **overrides}
