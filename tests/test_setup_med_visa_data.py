from pathlib import Path
import pytest
from tools.setup_med_visa_data import (
    CANONICAL_PARTS, discover_candidate_roots, is_complete_root,
    materialize_layout, select_candidate_root, verify_manifests,
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


def _write_brain_manifest(tmp_path: Path, record: str):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "Brain.jsonl").write_text(record + "\n", encoding="utf-8")
    return manifest_dir


def test_anomaly_record_requires_nonempty_existing_mask(tmp_path, monkeypatch):
    import tools.setup_med_visa_data as setup

    sample_root = tmp_path / "data" / "MedAD" / "Brain_AD" / "test"
    image = sample_root / "bad" / "img" / "one.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    manifest_dir = _write_brain_manifest(
        tmp_path,
        '{"image_path":"bad/img/one.png","label":1,"class_name":"Brain","mask_path":""}',
    )
    monkeypatch.setattr(setup, "MANIFEST_ROOTS", {"Brain": Path("MedAD/Brain_AD/test")})
    monkeypatch.setattr(setup, "EXPECTED_LABEL_COUNTS", {"Brain": {0: 0, 1: 1}})
    report = verify_manifests(tmp_path / "data", manifest_dir)
    assert report["ok"] is False
    assert report["datasets"]["Brain"]["missing_mask_count"] == 1
    assert "anomaly mask_path missing" in report["missing_masks"][0]


def test_class_name_must_match_dataset_info_metadata(tmp_path, monkeypatch):
    import tools.setup_med_visa_data as setup

    sample_root = tmp_path / "data" / "MedAD" / "Brain_AD" / "test"
    image = sample_root / "bad" / "img" / "one.png"
    mask = sample_root / "bad" / "anomaly_mask" / "one.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    manifest_dir = _write_brain_manifest(
        tmp_path,
        '{"image_path":"bad/img/one.png","label":1,"class_name":"brain","mask_path":"bad/anomaly_mask/one.png"}',
    )
    monkeypatch.setattr(setup, "MANIFEST_ROOTS", {"Brain": Path("MedAD/Brain_AD/test")})
    monkeypatch.setattr(setup, "EXPECTED_LABEL_COUNTS", {"Brain": {0: 0, 1: 1}})
    report = verify_manifests(tmp_path / "data", manifest_dir)
    assert report["ok"] is False
    assert report["datasets"]["Brain"]["invalid_class_name_count"] == 1
    assert "not in ['Brain']" in report["invalid_class_names"][0]
