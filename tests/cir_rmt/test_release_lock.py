import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.cir_rmt.identity import git_identity, load_cir_config, release_identity_fields
from tools.cir_rmt.release_lock import (
    LOCK_SCHEMA,
    REAL_EVIDENCE_KINDS,
    REQUIRED_GATES,
    ReleaseNotAuthorized,
    validate_gate_manifest,
    validate_lock_payload,
)


def _fake_gate(name: str, identity: dict[str, object]) -> dict[str, object]:
    requirement = REQUIRED_GATES[name]
    return {
        "gate": name,
        "status": "PASS",
        "scope": requirement["scope"],
        "real": bool(requirement["real"]),
        "real_asset": bool(requirement["real"]),
        "identity": dict(identity),
        "evidence": {"kind": REAL_EVIDENCE_KINDS[name]} if requirement["real"] else {"kind": "local"},
    }


def _fake_payload(config: dict[str, object], git: dict[str, object]) -> dict[str, object]:
    identity = release_identity_fields(config)
    return {
        "schema_version": LOCK_SCHEMA,
        "release_lock": True,
        **identity,
        "rmt_alpha_status": config["rmt_alpha_status"],
        "branch": git["branch"],
        "git_sha": git["head"],
        "gate_statuses": {name: _fake_gate(name, identity) for name in REQUIRED_GATES},
    }


def test_missing_lock_cli_fails_closed_with_real_gate_message(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    missing = tmp_path / "missing-release-lock.json"
    completed = subprocess.run(
        [sys.executable, "-m", "tools.cir_rmt.release_lock", "--verify", "--lock", str(missing)],
        cwd=repo_root, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 3
    assert "RELEASE NOT AUTHORIZED:" in completed.stdout
    assert "missing real G2/G3/G4/G5 PASS" in completed.stdout


def test_mismatched_config_sha_is_rejected_even_with_fake_passes():
    config = load_cir_config()
    git = git_identity()
    payload = _fake_payload(config, git)
    payload["config_sha256"] = "0" * 64
    with pytest.raises(ReleaseNotAuthorized, match="lock mismatch: config_sha256"):
        validate_lock_payload(payload, config, git)


def test_fake_pass_manifest_cannot_generate_lock_with_provisional_alpha():
    config = load_cir_config()
    manifest = {"gates": [_fake_gate(name, release_identity_fields(config)) for name in REQUIRED_GATES]}
    with pytest.raises(ReleaseNotAuthorized, match="rmt_transport_alpha is PROVISIONAL"):
        validate_gate_manifest(manifest, config)


def test_fake_payload_is_rejected_while_alpha_is_provisional():
    config = load_cir_config()
    git = git_identity()
    with pytest.raises(ReleaseNotAuthorized, match="rmt_transport_alpha is PROVISIONAL"):
        validate_lock_payload(_fake_payload(config, git), config, git)


def test_fake_passes_without_real_execution_evidence_are_rejected():
    config = dict(load_cir_config())
    config["rmt_alpha_status"] = "FROZEN"
    identity = release_identity_fields(config)
    manifest = {"gates": [_fake_gate(name, identity) for name in REQUIRED_GATES]}
    with pytest.raises(ReleaseNotAuthorized, match="real execution evidence"):
        validate_gate_manifest(manifest, config)
