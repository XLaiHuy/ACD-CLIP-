#!/usr/bin/env python3
"""Canonical P26 SABRA final entry point.

The frozen branch is intentionally native-only.  `--run` remains governance
locked; this module never silently enables external evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_CONFIG = Path("research/sabra_cure/final_architecture/SABRA_FINAL_CONFIG.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("final config must be a JSON object")
    return payload


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "SABRA_FINAL_CONFIG_V1":
        raise ValueError("unsupported SABRA final config schema")
    if config.get("status") != "P26_FINAL_ARCHITECTURE_FROZEN":
        raise ValueError("architecture is not frozen")
    architecture = config.get("architecture", {})
    if architecture.get("policy") != "NATIVE_ONLY":
        raise ValueError("P26 final policy must be NATIVE_ONLY")
    if architecture.get("reachable_actions") != ["KEEP"]:
        raise ValueError("P26 reachable action set must be exactly KEEP")
    if architecture.get("intervention_enabled") is not False:
        raise ValueError("P26 intervention must be disabled")
    if architecture.get("coverage_fraction") != 0.0:
        raise ValueError("P26 correction coverage must be zero")
    model = config.get("model", {})
    expected_model = {
        "protocol_version": "PHASE2B_CANONICAL_V1",
        "model_name": "ViT-L-14-336",
        "precision": "fp32",
        "n_groups": 3,
        "patch_grid": [37, 37],
        "patch_count": 1369,
        "h6_enabled": False,
        "legacy_branch_enabled": False,
    }
    for key, value in expected_model.items():
        if model.get(key) != value:
            raise ValueError(f"frozen model field mismatch: {key}")
    post = config.get("postprocessing", {})
    expected_post = {
        "gaussian_kernel": [7, 7],
        "gaussian_sigma": [1.0, 1.0],
        "interpolation": "bilinear",
        "align_corners": True,
        "stage_aggregation": "mean_logits",
        "probability": "softmax_after_stage_mean",
        "anomaly_channel": 1,
    }
    for key, value in expected_post.items():
        if post.get(key) != value:
            raise ValueError(f"frozen postprocessing field mismatch: {key}")
    if config.get("external_validation", {}).get("authorized") is not False:
        raise ValueError("P26 external validation must remain unauthorized")
    if not config.get("required_artifacts"):
        raise ValueError("required artifact inventory is empty")


def check_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    missing_dependencies = [
        name for name in ("torch", "torchvision", "numpy", "PIL")
        if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        raise RuntimeError(f"missing dependencies: {missing_dependencies}")
    verified = 0
    for record in config["required_artifacts"]:
        path = repo_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"required artifact missing: {record['path']}")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {record['path']}")
        verified += 1
    return {
        "status": "PASS",
        "mode": "check-only",
        "required_artifacts_verified": verified,
        "scientific_evaluation": False,
        "external_dataset_reads": 0,
        "clip_forwards": 0,
        "phase2b_steps": 0,
    }


def dry_run(config_path: Path, repo_root: Path) -> dict[str, Any]:
    check_only(config_path, repo_root)
    config = load_config(config_path)
    repo_root_text = str(repo_root.resolve())
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    import torch
    from model.phase2b_runtime import deploy_native_logits

    torch.manual_seed(int(config["determinism"]["torch_seed"]))
    logits = torch.zeros((3, 1, 1369, 2), dtype=torch.float32)
    logits[:, :, :, 1] = torch.linspace(-1.0, 1.0, 1369, dtype=torch.float32)
    probabilities, _ = deploy_native_logits(
        logits,
        patch_grid=tuple(config["model"]["patch_grid"]),
        image_size=int(config["preprocessing"]["resize"][0]),
        domain="Industrial",
    )
    anomaly = probabilities[:, int(config["postprocessing"]["anomaly_channel"])]
    if not torch.isfinite(anomaly).all():
        raise RuntimeError("synthetic native postprocessing produced nonfinite values")
    return {
        "status": "PASS",
        "mode": "dry-run",
        "fixture": "SYNTHETIC_NATIVE_LOGITS",
        "output_shape": list(anomaly.shape),
        "output_dtype": str(anomaly.dtype),
        "scientific_evaluation": False,
        "external_dataset_reads": 0,
        "clip_forwards": 0,
        "phase2b_steps": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.run:
        raise SystemExit(
            "EXTERNAL VALIDATION AUTHORIZATION REQUIRED: P26 freezes inference "
            "but does not authorize MVTec or any other external run"
        )
    result = dry_run(args.config, args.repo_root) if args.dry_run else check_only(args.config, args.repo_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
