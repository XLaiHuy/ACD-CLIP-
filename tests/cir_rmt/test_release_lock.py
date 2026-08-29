import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.cir_rmt.identity import architecture_branch, git_identity, load_cir_config, release_identity_fields
from tools.cir_rmt.release_lock import (
    LOCK_SCHEMA,
    REAL_EVIDENCE_KINDS,
    REQUIRED_GATES,
    ReleaseNotAuthorized,
    canonical_lock_path,
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


def test_canonical_lock_path_is_architecture_aware():
    repo_root = Path(__file__).resolve().parents[2]
    v1 = load_cir_config(repo_root / "configs/cir_dfg_rmt_v1.json")
    v2 = load_cir_config(repo_root / "configs/cir_dfg_rmt_v2.json")
    assert canonical_lock_path(v1) == repo_root / "runs/cir_rmt/CIR_DFG_RMT_V1/release_lock.json"
    assert canonical_lock_path(v2) == repo_root / "runs/cir_rmt/CIR_DFG_RMT_V2/release_lock.json"
    assert canonical_lock_path(v1) != canonical_lock_path(v2)


def test_cross_version_lock_path_is_rejected():
    config = dict(load_cir_config())
    config["rmt_alpha_status"] = "FROZEN"
    git = dict(git_identity())
    git["branch"] = architecture_branch(config)
    git["head"] = "test-git-sha"
    git["status_short"] = []
    git["clean"] = True
    payload = _fake_payload(config, git)
    payload["generated_at_utc"] = "2026-01-01T00:00:00Z"
    for name in ("G2_REAL", "G3_REAL", "G4_GPU", "G5_REAL"):
        payload["gate_statuses"][name]["evidence"].update({"real_execution": True, "artifact": {"path": "test"}})
    v2 = load_cir_config(Path(__file__).resolve().parents[2] / "configs/cir_dfg_rmt_v2.json")
    with pytest.raises(ReleaseNotAuthorized, match="release lock path is not canonical"):
        validate_lock_payload(payload, config, git, canonical_lock_path(v2))


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
