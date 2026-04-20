"""Tests for experiments/matrix.py — yaml grid search expansion."""

from __future__ import annotations

from bidmate_rag.experiments.matrix import (
    apply_overrides_to_yaml_dict,
    expand_matrix,
)

# ---------------------------------------------------------------------------
# expand_matrix
# ---------------------------------------------------------------------------


def test_empty_matrix_returns_empty_list():
    assert expand_matrix({}) == []
    assert expand_matrix({}) == []


def test_one_dim_three_values():
    cells = expand_matrix({"chunk_size": [500, 1000, 1500]})
    assert len(cells) == 3
    assert [c.overrides["chunk_size"] for c in cells] == [500, 1000, 1500]
    assert [c.name_suffix for c in cells] == [
        "chunk_size500",
        "chunk_size1000",
        "chunk_size1500",
    ]


def test_two_dim_cartesian_product():
    cells = expand_matrix(
        {"chunk_size": [500, 1000], "retrieval_top_k": [3, 5, 8]}
    )
    assert len(cells) == 6
    # 카르테시안 곱: chunk_size × retrieval_top_k (sorted keys)
    expected = [
        (500, 3), (500, 5), (500, 8),
        (1000, 3), (1000, 5), (1000, 8),
    ]
    for cell, (cs, tk) in zip(cells, expected):
        assert cell.overrides == {"chunk_size": cs, "retrieval_top_k": tk}


def test_three_dim_grid():
    cells = expand_matrix(
        {
            "chunk_size": [500, 1000],
            "retrieval_top_k": [3, 5],
            "chunk_overlap": [50, 100],
        }
    )
    assert len(cells) == 2 * 2 * 2  # 8


def test_keys_sorted_deterministic():
    """입력 dict 순서가 달라도 항상 같은 순서로 expand."""
    a = expand_matrix({"chunk_size": [500], "retrieval_top_k": [3]})
    b = expand_matrix({"retrieval_top_k": [3], "chunk_size": [500]})
    assert a[0].name_suffix == b[0].name_suffix
    assert a[0].overrides == b[0].overrides


def test_suffix_format():
    cells = expand_matrix({"chunk_size": [800], "retrieval_top_k": [5]})
    assert cells[0].name_suffix == "chunk_size800_retrieval_top_k5"


def test_single_value_lists():
    """각 키에 값이 1개씩이면 1개 cell만."""
    cells = expand_matrix({"chunk_size": [1000], "retrieval_top_k": [5]})
    assert len(cells) == 1
    assert cells[0].overrides == {"chunk_size": 1000, "retrieval_top_k": 5}


def test_string_values_in_matrix():
    """matrix는 int뿐 아니라 임의 타입도 지원해야 함 (예: provider_config)."""
    cells = expand_matrix({"chunk_strategy": ["recursive", "header_aware"]})
    assert len(cells) == 2
    suffixes = [c.name_suffix for c in cells]
    assert "chunk_strategyrecursive" in suffixes
    assert "chunk_strategyheader_aware" in suffixes


# ---------------------------------------------------------------------------
# apply_overrides_to_yaml_dict
# ---------------------------------------------------------------------------


def test_apply_overrides_merges_dicts():
    base = {"name": "exp", "mode": "full_rag", "chunk_size": 1000}
    overrides = {"chunk_size": 500, "retrieval_top_k": 8}
    result = apply_overrides_to_yaml_dict(base, overrides)
    assert result == {
        "name": "exp",
        "mode": "full_rag",
        "chunk_size": 500,  # overridden
        "retrieval_top_k": 8,  # added
    }


def test_apply_overrides_does_not_mutate_base():
    base = {"chunk_size": 1000}
    apply_overrides_to_yaml_dict(base, {"chunk_size": 500})
    assert base == {"chunk_size": 1000}  # base is unchanged


def test_apply_overrides_with_empty_overrides():
    base = {"name": "x", "mode": "y"}
    result = apply_overrides_to_yaml_dict(base, {})
    assert result == base
    assert result is not base  # 새 dict 반환
