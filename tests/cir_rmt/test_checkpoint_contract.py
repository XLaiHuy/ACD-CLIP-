import pytest

from tools.cir_rmt.identity import checkpoint_metadata, config_sha256, load_cir_config, validate_checkpoint_identity


def test_checkpoint_identity_round_trip_and_hard_fail():
    config = load_cir_config()
    metadata = checkpoint_metadata(config, source_dataset="VisA", epoch=4, git_sha="test-sha")
    validate_checkpoint_identity(metadata, config, source_dataset="VisA", expected_git_sha="test-sha")
    broken = dict(metadata)
    broken["rmt_peer_count"] = 7
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_identity(broken, config, source_dataset="VisA", expected_git_sha="test-sha")
    assert len(config_sha256(config)) == 64
