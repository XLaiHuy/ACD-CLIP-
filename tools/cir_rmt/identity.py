"""Identity and checkpoint-contract helpers for CIR_DFG_RMT_V1."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

ARCH_ID = "CIR_DFG_RMT_V1"
ARCH_VERSION = 1
BRANCH = "research/cir-dfg-rmt-v1"
PARENT_PROTOCOL = "PHASE2B_CANONICAL_V1"
EVALUATOR_PROTOCOL = "CIR_FINAL_EXACT_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/cir_dfg_rmt_v1.json"


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def load_cir_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    validate_cir_config(payload, config_path=config_path)
    return payload


def validate_cir_config(config: Mapping[str, Any], *, config_path: str | Path | None = None) -> None:
    required = {
        "arch_id", "architecture_version", "architecture_freeze_path", "architecture_freeze_sha256",
        "parent_protocol", "parent_config_sha256",
        "n_groups", "rmt_enabled", "rmt_peer_count", "rmt_center", "rmt_scale",
        "rmt_mad_constant", "rmt_eps", "rmt_transform", "rmt_transport",
        "rmt_transport_alpha", "rmt_alpha_status", "rmt_spatial_radius", "rmt_delta_layout", "rmt_delta_stopgrad", "rmt_score_mode",
        "rmt_gradient_contract",
        "precision", "evaluator_protocol",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"CIR config missing fields: {missing}")
    if str(config["arch_id"]) != ARCH_ID or int(config["architecture_version"]) != ARCH_VERSION:
        raise ValueError("CIR architecture identity mismatch")
    if str(config["parent_protocol"]) != PARENT_PROTOCOL or int(config["n_groups"]) != 3 or int(config["rmt_peer_count"]) != 8:
        raise ValueError("CIR parent/group/peer identity mismatch")
    if config["rmt_enabled"] is not True or str(config["rmt_center"]) != "midpoint_median" or str(config["rmt_scale"]) != "mad":
        raise ValueError("CIR robust settings are not frozen")
    if abs(float(config["rmt_mad_constant"]) - 1.4826) > 1e-12 or float(config["rmt_eps"]) <= 0 or str(config["rmt_transform"]) != "tanh":
        raise ValueError("CIR MAD/transform settings are invalid")
    if int(config["rmt_spatial_radius"]) != 3 or str(config["rmt_delta_layout"]) != "per_stage_per_group":
        raise ValueError("CIR peer geometry/delta layout is not frozen")
    if str(config["rmt_gradient_contract"]) != "peer_search_detached_delta_stopgrad_native_dfg_differentiable":
        raise ValueError("CIR gradient contract is not frozen")
    if str(config["rmt_transport"]) != "kl_antisymmetric" or float(config["rmt_transport_alpha"]) < 0:
        raise ValueError("CIR transport settings are invalid")
    if str(config["rmt_alpha_status"]) not in {"PROVISIONAL", "FROZEN"}:
        raise ValueError("CIR alpha status must be PROVISIONAL or FROZEN")
    if config["rmt_delta_stopgrad"] is not True or str(config["rmt_score_mode"]) not in {"exact_score_space", "reference", "optimized"}:
        raise ValueError("CIR delta/score settings are invalid")
    if str(config["precision"]) != "fp32" or str(config["evaluator_protocol"]) != EVALUATOR_PROTOCOL:
        raise ValueError("CIR precision/evaluator protocol mismatch")
    if config_path is not None and config.get("architecture_freeze_path"):
        freeze = Path(str(config["architecture_freeze_path"]))
        if not freeze.is_absolute():
            freeze = Path(config_path).parent.parent / freeze
        if not freeze.is_file():
            raise FileNotFoundError(f"architecture freeze missing: {freeze}")
        expected = config.get("architecture_freeze_sha256")
        if expected and sha256_file(freeze) != str(expected):
            raise ValueError("architecture freeze SHA256 mismatch")


def release_identity_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact identity fields bound into release/checkpoint artifacts."""
    validate_cir_config(config)
    return {
        "arch_id": str(config["arch_id"]),
        "architecture_version": int(config["architecture_version"]),
        "config_sha256": config_sha256(config),
        "architecture_freeze_sha256": str(config["architecture_freeze_sha256"]),
        "parent_config_sha256": str(config["parent_config_sha256"]),
        "rmt_transport_alpha": float(config["rmt_transport_alpha"]),
        "n_groups": int(config["n_groups"]),
        "rmt_peer_count": int(config["rmt_peer_count"]),
        "rmt_score_mode": str(config["rmt_score_mode"]),
        "evaluator_protocol": str(config["evaluator_protocol"]),
    }
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"ERROR:{exc}"


def git_identity() -> dict[str, Any]:
    status = _git("status", "--short")
    return {
        "repo_root": str(REPO_ROOT),
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "worktree": _git("rev-parse", "--show-toplevel"),
        "status_short": status.splitlines() if status else [],
        "clean": not bool(status),
        "remote_origin": _git("remote", "get-url", "origin"),
    }


def environment_identity() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    gpus: list[dict[str, Any]] = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": props.name, "total_memory": int(props.total_memory), "capability": [props.major, props.minor]})
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpus": gpus,
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_root": os.environ.get("ACDCLIP_DATA_ROOT"),
        "visa_root": os.environ.get("VISA_ROOT"),
        "mvtec_root": os.environ.get("MVTEC_ROOT") or os.environ.get("ACDCLIP_MVTEC_ROOT"),
        "medical_root": os.environ.get("MEDICAL_ROOT"),
    }


def build_run_identity(config: Mapping[str, Any], *, source_dataset: str, git_sha: str | None = None, evaluator_protocol: str = EVALUATOR_PROTOCOL) -> dict[str, Any]:
    validate_cir_config(config)
    parent_path = REPO_ROOT / str(config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    parent_sha = str(config["parent_config_sha256"])
    if parent_path.is_file():
        raw = json.loads(parent_path.read_text(encoding="utf-8"))
        if config_sha256(raw) != parent_sha:
            raise ValueError("parent config SHA256 does not match canonical config")
    return {
        "arch_id": ARCH_ID,
        "architecture_version": ARCH_VERSION,
        "git_sha": git_sha or git_identity()["head"],
        "config_sha256": config_sha256(config),
        "parent_protocol": PARENT_PROTOCOL,
        "parent_config_sha256": parent_sha,
        "source_dataset": str(source_dataset),
        "n_groups": int(config["n_groups"]),
        "rmt_peer_count": int(config["rmt_peer_count"]),
        "rmt_transport_alpha": float(config["rmt_transport_alpha"]),
        "rmt_score_mode": str(config["rmt_score_mode"]),
        "evaluator_protocol": str(evaluator_protocol),
        "architecture_freeze_sha256": config.get("architecture_freeze_sha256"),
    }


def checkpoint_metadata(config: Mapping[str, Any], *, source_dataset: str, epoch: int, git_sha: str | None = None, parent_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    identity = build_run_identity(config, source_dataset=source_dataset, git_sha=git_sha)
    identity.update({
        "checkpoint_contract": "CIR_CHECKPOINT_V1",
        "epoch": int(epoch),
        "delta_stopgrad": True,
        "source": str(source_dataset),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    })
    return identity


def validate_checkpoint_identity(checkpoint: Mapping[str, Any], config: Mapping[str, Any], *, source_dataset: str | None = None, expected_git_sha: str | None = None, evaluator_protocol: str = EVALUATOR_PROTOCOL, expected_epoch: int | None = None) -> None:
    source = str(source_dataset or checkpoint.get("source", checkpoint.get("source_dataset", "")))
    expected = build_run_identity(config, source_dataset=source, git_sha=expected_git_sha or checkpoint.get("git_sha"), evaluator_protocol=evaluator_protocol)
    required_identity = ["arch_id", "architecture_version", "config_sha256", "architecture_freeze_sha256", "parent_config_sha256", "n_groups", "rmt_peer_count", "rmt_transport_alpha", "rmt_score_mode", "evaluator_protocol", "delta_stopgrad"]
    required_metadata = ["source", "epoch"]
    missing = [key for key in required_identity + required_metadata if key not in checkpoint]
    if missing:
        raise ValueError(f"CIR checkpoint missing identity fields: {missing}")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key in required_identity:
        actual = checkpoint.get(key)
        wanted = True if key == "delta_stopgrad" else expected[key]
        different = abs(float(actual) - float(wanted)) > 1e-12 if key == "rmt_transport_alpha" else actual != wanted
        if different:
            mismatches[key] = (actual, wanted)
    if source_dataset is not None and str(checkpoint.get("source", checkpoint.get("source_dataset"))) != str(source_dataset):
        mismatches["source"] = (checkpoint.get("source", checkpoint.get("source_dataset")), source_dataset)
    if expected_epoch is not None and int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        mismatches["epoch"] = (checkpoint.get("epoch"), expected_epoch)
    if mismatches:
        raise ValueError(f"CIR checkpoint identity mismatch: {mismatches}")


def assert_g0(*, allow_dirty: bool = False, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_cir_config(config_path)
    git = git_identity()
    if git["branch"] != BRANCH or Path(git["worktree"]).resolve() != REPO_ROOT.resolve():
        raise RuntimeError(f"G0 repository identity mismatch: {git}")
    if not allow_dirty and not git["clean"]:
        raise RuntimeError(f"G0 requires a clean worktree: {git['status_short']}")
    return {"stage": "CIR/G0-IDENTITY", "status": "PASS", "git": git, "config_sha256": config_sha256(config), "config": config, "environment": environment_identity()}
