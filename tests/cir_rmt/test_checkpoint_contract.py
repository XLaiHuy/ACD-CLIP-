from pathlib import Path
import pytest

from tools.cir_rmt.identity import canonical_json, checkpoint_metadata, config_sha256, load_cir_config, sha256_file, validate_checkpoint_identity, validate_cir_config


def test_checkpoint_identity_round_trip_and_hard_fail():
    config = load_cir_config()
    metadata = checkpoint_metadata(config, source_dataset="VisA", epoch=4, git_sha="test-sha")
    validate_checkpoint_identity(metadata, config, source_dataset="VisA", expected_git_sha="test-sha", expected_epoch=4)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_identity(metadata, config, source_dataset="VisA", expected_git_sha="test-sha", expected_epoch=5)
    broken = dict(metadata)
    broken["rmt_peer_count"] = 7
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_identity(broken, config, source_dataset="VisA", expected_git_sha="test-sha")
    assert len(config_sha256(config)) == 64


def test_canonical_hashes_are_deterministic_and_bind_freeze():
    config = load_cir_config()
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    freeze = config["architecture_freeze_path"]
    path = Path(__file__).resolve().parents[2] / freeze
    assert config["architecture_freeze_sha256"] == sha256_file(path)
    assert config_sha256(config) == config_sha256(dict(config))


def test_legacy_delta_layout_is_rejected():
    config = load_cir_config()
    assert config["rmt_delta_layout"] == "per_stage_per_group"
    legacy = dict(config)
    legacy["rmt_delta_layout"] = "per_stage_to_group_vector"
    with pytest.raises(ValueError, match="CIR peer geometry/delta layout is not frozen"):
        validate_cir_config(legacy)
