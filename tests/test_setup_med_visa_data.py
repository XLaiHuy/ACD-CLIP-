from pathlib import Path
import pytest
from tools.setup_med_visa_data import (
    CANONICAL_PARTS, discover_candidate_roots, is_complete_root,
    materialize_layout, select_candidate_root,
)


def make_tree(root: Path):
    for relative in CANONICAL_PARTS:
        (root / relative).mkdir(parents=True, exist_ok=True)


def test_discovers_one_nested_complete_root(tmp_path):
    expected = tmp_path / "upload" / "nested" / "data"
    make_tree(expected)
    assert discover_candidate_roots(tmp_path) == [expected.resolve()]
    assert select_candidate_root(tmp_path) == expected.resolve()


def test_refuses_ambiguous_roots(tmp_path):
    make_tree(tmp_path / "one"); make_tree(tmp_path / "two")
    with pytest.raises(RuntimeError, match="ambiguous"):
        select_candidate_root(tmp_path)


def test_symlink_materialization_without_duplicate_copy(tmp_path):
    source, target = tmp_path / "source", tmp_path / "repo-data"
    make_tree(source)
    materialize_layout(source, target, "auto")
    assert is_complete_root(target)
    assert all((target / name).is_symlink() for name in ("VisA_20220922", "MedAD", "Colon"))


def test_existing_complete_data_root_is_detected(tmp_path):
    make_tree(tmp_path)
    assert is_complete_root(tmp_path)
