"""Generate and verify the only release authorization for CIR_DFG_RMT_V1."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .identity import (
    ARCH_ID,
    BRANCH,
    REPO_ROOT,
    config_sha256,
    git_identity,
    load_cir_config,
    release_identity_fields,
)

LOCK_SCHEMA = "CIR_RELEASE_LOCK_V1"
DEFAULT_LOCK_PATH = REPO_ROOT / "runs/cir_rmt/CIR_DFG_RMT_V1/release_lock.json"
REQUIRED_GATES = {
    "G0": {"scope": "identity", "real": False},
    "G1": {"scope": "unit", "real": False},
    "G2_REAL": {"scope": "real", "real": True},
    "G3_REAL": {"scope": "real", "real": True},
    "G4_GPU": {"scope": "gpu", "real": True},
    "G5_REAL": {"scope": "real", "real": True},
}
IDENTITY_KEYS = (
    "arch_id",
    "architecture_version",
    "config_sha256",
    "architecture_freeze_sha256",
    "parent_config_sha256",
    "rmt_transport_alpha",
    "n_groups",
    "rmt_peer_count",
    "rmt_score_mode",
    "evaluator_protocol",
)


REAL_EVIDENCE_KINDS = {
    "G2_REAL": "alpha0_parity_real",
    "G3_REAL": "source_preflight_real",
    "G4_GPU": "production_gpu",
    "G5_REAL": "train_smoke_real",
}
class ReleaseNotAuthorized(RuntimeError):
    """Raised when a release lock cannot be safely generated or verified."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseNotAuthorized("missing gate or lock artifact: " + str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ReleaseNotAuthorized("invalid JSON artifact: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ReleaseNotAuthorized("JSON artifact must be an object: " + str(path))
    return payload


def _gate_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source = payload.get("gates")
    records: list[Mapping[str, Any]] = []
    if isinstance(source, Mapping):
        records = [value for value in source.values() if isinstance(value, Mapping)]
    elif isinstance(source, list):
        records = [value for value in source if isinstance(value, Mapping)]
    elif source is None:
        records = [
            value
            for key, value in payload.items()
            if str(key) in REQUIRED_GATES and isinstance(value, Mapping)
        ]
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        name = str(record.get("gate", record.get("name", ""))).upper()
        if name:
            result[name] = dict(record)
    return result


def _record_identity(record: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = record.get("identity")
    return nested if isinstance(nested, Mapping) else record


def _gate_errors(
    gates: Mapping[str, Mapping[str, Any]],
    expected_identity: Mapping[str, Any],
    *,
    require_identity: bool,
) -> list[str]:
    errors: list[str] = []
    for name, requirement in REQUIRED_GATES.items():
        record = gates.get(name)
        if record is None:
            errors.append(name + " missing")
            continue
        if str(record.get("status", "")).upper() != "PASS":
            errors.append(name + " status=" + str(record.get("status", "MISSING")))
        if str(record.get("scope", "")).lower() != requirement["scope"]:
            errors.append(name + " scope must be " + requirement["scope"])
        if requirement["real"] and record.get("real") is not True:
            errors.append(name + " is not marked real")
        if requirement["real"] and record.get("real_asset") is not True:
            errors.append(name + " is not backed by real assets")
        evidence = record.get("evidence")
        if requirement["real"] and (not isinstance(evidence, Mapping) or not evidence):
            errors.append(name + " evidence is missing")
        elif requirement["real"] and evidence.get("kind") != REAL_EVIDENCE_KINDS[name]:
            errors.append(name + " evidence kind must be " + REAL_EVIDENCE_KINDS[name])
        if requirement["real"] and isinstance(evidence, Mapping) and evidence.get("real_execution") is not True:
            errors.append(name + " real execution evidence is not verified")
        if requirement["real"] and isinstance(evidence, Mapping) and not evidence.get("artifact"):
            errors.append(name + " evidence artifact is missing")
        identity = _record_identity(record)
        for key in IDENTITY_KEYS:
            if require_identity and key not in identity:
                errors.append(name + " missing identity." + key)
            elif key in identity:
                actual = identity[key]
                wanted = expected_identity[key]
                if key == "rmt_transport_alpha":
                    try:
                        mismatch = abs(float(actual) - float(wanted)) > 1e-12
                    except (TypeError, ValueError):
                        mismatch = True
                else:
                    mismatch = actual != wanted
                if mismatch:
                    errors.append(name + " identity mismatch: " + key)
    return errors


def validate_gate_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected = release_identity_fields(config)
    gates = _gate_map(manifest)
    errors = _gate_errors(gates, expected, require_identity=True)
    if str(config.get("rmt_alpha_status")) != "FROZEN":
        errors.append("rmt_transport_alpha is PROVISIONAL")
    if errors:
        raise ReleaseNotAuthorized("; ".join(errors))
    return gates


def _lock_identity(config: Mapping[str, Any], git: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identity = release_identity_fields(config)
    identity.update(
        {
            "schema_version": LOCK_SCHEMA,
            "release_lock": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "branch": str(git["branch"]),
            "git_sha": str(git["head"]),
            "rmt_alpha_status": str(config["rmt_alpha_status"]),
            "gate_statuses": {
                name: {
                    "status": str(gates[name].get("status")),
                    "scope": str(gates[name].get("scope")),
                    "real": bool(gates[name].get("real")),
                    "real_asset": bool(gates[name].get("real_asset")),
                    "evidence": dict(gates[name].get("evidence", {})),
                }
                for name in REQUIRED_GATES
            },
        }
    )
    return identity


def generate_release_lock(
    config_path: str | Path,
    gates_manifest_path: str | Path,
    output_path: str | Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    config = load_cir_config(config_path)
    manifest = _read_json(Path(gates_manifest_path).expanduser().resolve())
    gates = validate_gate_manifest(manifest, config)
    git = git_identity()
    if git["branch"] != BRANCH or Path(git["worktree"]).resolve() != REPO_ROOT.resolve():
        raise ReleaseNotAuthorized("repository identity is not the CIR release worktree")
    if not git["clean"]:
        raise ReleaseNotAuthorized("G0 requires a clean worktree before lock generation")
    payload = _lock_identity(config, git, gates)
    output = Path(output_path).expanduser().resolve()
    if output != DEFAULT_LOCK_PATH.resolve():
        raise ReleaseNotAuthorized("release lock output must be the canonical worktree path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def _lock_status_records(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("gate_statuses")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key).upper(): dict(value) for key, value in raw.items() if isinstance(value, Mapping)}


def validate_lock_payload(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    git: Mapping[str, Any],
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    errors: list[str] = []
    expected = release_identity_fields(config)
    if payload.get("schema_version") != LOCK_SCHEMA:
        errors.append("schema_version mismatch")
    if payload.get("release_lock") is not True:
        errors.append("release_lock is not true")
    if str(config.get("rmt_alpha_status")) != "FROZEN":
        errors.append("rmt_transport_alpha is PROVISIONAL")
    if payload.get("rmt_alpha_status") != config.get("rmt_alpha_status"):
        errors.append("lock mismatch: rmt_alpha_status")
    if not payload.get("generated_at_utc"):
        errors.append("generated_at_utc missing")
    for key in IDENTITY_KEYS:
        actual = payload.get(key)
        wanted = expected[key]
        if key == "rmt_transport_alpha":
            try:
                mismatch = abs(float(actual) - float(wanted)) > 1e-12
            except (TypeError, ValueError):
                mismatch = True
        else:
            mismatch = actual != wanted
        if mismatch:
            errors.append("lock mismatch: " + key)
    if str(payload.get("branch")) != BRANCH or str(git.get("branch")) != BRANCH:
        errors.append("branch mismatch")
    if str(payload.get("git_sha")) != str(git.get("head")):
        errors.append("git SHA mismatch")
    gates = _lock_status_records(payload)
    errors.extend(_gate_errors(gates, expected, require_identity=False))
    if errors:
        raise ReleaseNotAuthorized("; ".join(errors))
    lock = Path(lock_path).expanduser().resolve()
    expected_lock = DEFAULT_LOCK_PATH.resolve()
    if lock != expected_lock:
        raise ReleaseNotAuthorized("release lock path is not canonical")
    allowed_dirty = "?? " + str(lock.relative_to(REPO_ROOT))
    dirty = [entry for entry in git.get("status_short", []) if entry != allowed_dirty]
    if dirty:
        raise ReleaseNotAuthorized("unrelated dirty files: " + ", ".join(dirty))
    return dict(payload)


def verify_release_lock(
    config_path: str | Path = REPO_ROOT / "configs/cir_dfg_rmt_v1.json",
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    lock = Path(lock_path).expanduser().resolve()
    if not lock.is_file():
        raise ReleaseNotAuthorized("missing release_lock.json")
    config = load_cir_config(config_path)
    payload = _read_json(lock)
    git = git_identity()
    if Path(git["worktree"]).resolve() != REPO_ROOT.resolve():
        raise ReleaseNotAuthorized("repository identity is not the CIR release worktree")
    return validate_lock_payload(payload, config, git, lock)


def verify_checkpoint_contract(
    config_path: str | Path,
    checkpoint_path: str | Path,
    source: str,
    epoch: int,
) -> dict[str, Any]:
    """Verify a checkpoint against the full CIR identity and requested epoch."""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise ReleaseNotAuthorized("missing checkpoint: " + str(checkpoint_file))
    try:
        import torch
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ReleaseNotAuthorized("invalid checkpoint: " + str(checkpoint_file)) from exc
    if not isinstance(payload, Mapping):
        raise ReleaseNotAuthorized("checkpoint must contain an object: " + str(checkpoint_file))
    config = load_cir_config(config_path)
    current_git_sha = git_identity()["head"]
    try:
        from .identity import validate_checkpoint_identity
        validate_checkpoint_identity(
            payload,
            config,
            source_dataset=str(source),
            expected_git_sha=current_git_sha,
            expected_epoch=int(epoch),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ReleaseNotAuthorized(str(exc)) from exc
    return dict(payload)


def _targets(source: str) -> str:
    value = str(source).lower()
    industrial = "MVTec-AD" if value == "visa" else "VisA" if value == "mvtec" else "MVTec-AD + VisA"
    medical = "ColonDB, ClinicDB, Kvasir, BrainMRI, Liver CT, Retina OCT"
    return industrial + " + " + medical


def describe_release(
    config_path: str | Path,
    lock_path: str | Path,
    source: str,
    device: str,
) -> str:
    config = load_cir_config(config_path)
    lock_state = "ABSENT"
    try:
        verify_release_lock(config_path, lock_path)
        lock_state = "VALID"
    except ReleaseNotAuthorized as exc:
        lock_state = "BLOCKED: " + str(exc)
    git = git_identity()
    source_value = str(source).lower()
    source_label = "VisA" if source_value == "visa" else "MVTec-AD" if source_value == "mvtec" else "VisA + MVTec-AD"
    lines = [
        "ARCH        : " + ARCH_ID,
        "SOURCE      : " + source_label,
        "TRAIN       : 20 epochs",
        "EVAL EPOCHS : 12,14,16,18,20",
        "TARGETS     : " + _targets(source),
        "CONFIG SHA  : " + config_sha256(config),
        "FREEZE SHA  : " + str(config["architecture_freeze_sha256"]),
        "GIT SHA     : " + str(git["head"]),
        "RMT alpha   : " + str(config["rmt_transport_alpha"]) + " (" + str(config["rmt_alpha_status"]) + ")",
        "Peers       : " + str(config["rmt_peer_count"]),
        "Groups      : " + str(config["n_groups"]),
        "Score mode  : " + str(config["rmt_score_mode"]),
        "Release lock: " + lock_state,
        "Device      : " + str(device),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cir_dfg_rmt_v1.json")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--gates-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--source", choices=["visa", "mvtec", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.verify_checkpoint:
            if args.checkpoint is None or args.epoch is None or args.source == "both":
                parser.error("--verify-checkpoint requires --checkpoint, --epoch, and --source visa|mvtec")
            verify_checkpoint_contract(args.config, args.checkpoint, args.source, args.epoch)
            print("CHECKPOINT VERIFIED: source=" + args.source + " epoch=" + str(args.epoch) + " path=" + str(args.checkpoint))
            return 0
        if args.verify:
            verify_release_lock(args.config, args.lock)
            print("RELEASE LOCK VERIFIED: " + str(args.lock))
            return 0
        if args.describe:
            print(describe_release(args.config, args.lock, args.source, args.device))
            return 0
        if args.gates_manifest is None:
            parser.error("--gates-manifest is required when generating a release lock")
        payload = generate_release_lock(args.config, args.gates_manifest, args.output or DEFAULT_LOCK_PATH)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except ReleaseNotAuthorized as exc:
        message = str(exc)
        if args.verify and message == "missing release_lock.json":
            print("RELEASE NOT AUTHORIZED:")
            print("missing real G2/G3/G4/G5 PASS")
            print("detail: missing release_lock.json")
        else:
            print("RELEASE NOT AUTHORIZED: " + message)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
