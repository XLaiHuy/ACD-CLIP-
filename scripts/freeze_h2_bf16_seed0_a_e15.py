#!/usr/bin/env python3
"""Freeze and validate the one authorized BF16 Seed0 A E15 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("/workspace/h2_bf16_screening/fresh_seed0_a_e15")
DEFAULT_JSON = REPO / "results/H2_BF16_SEED0_A_E15_MANIFEST.json"
DEFAULT_MD = REPO / "audit/H2_BF16_SEED0_A_E15_NUMERICAL_AUDIT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def parse_skips(log: Path) -> list[dict[str, int]]:
    rows = []
    pattern = re.compile(r"skip_counts epoch=(\d+) non_finite_loss=(\d+) non_finite_grad=(\d+)")
    for epoch, loss, grad in pattern.findall(log.read_text(encoding="utf-8")):
        rows.append({"epoch": int(epoch), "non_finite_loss": int(loss), "non_finite_grad": int(grad)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    shared = args.run_root / "shared_e1/adapter_1.pth"
    final = args.run_root / "A/adapter_15.pth"
    assert shared.is_file() and final.is_file(), "missing fresh E1 or A E15 checkpoint"
    e1 = torch.load(shared, map_location="cpu", weights_only=False)
    e15 = torch.load(final, map_location="cpu", weights_only=False)
    assert (e1["epoch"], e1["global_step"]) == (1, 361)
    assert (e15["epoch"], e15["global_step"]) == (15, 5415)
    for checkpoint in (e1, e15):
        assert checkpoint["precision"] == "bf16"
        assert checkpoint["gradscaler_enabled"] is False
        assert checkpoint["scaler_state"] == {}
        assert checkpoint["tf32_enabled"] is False
        assert finite(checkpoint["model_state"])
        assert finite(checkpoint["optimizer_state"])

    skips = parse_skips(args.run_root / "shared_e1/train.log") + parse_skips(args.run_root / "A/train.log")
    assert [row["epoch"] for row in skips] == list(range(1, 16)), skips
    assert all(not row["non_finite_loss"] and not row["non_finite_grad"] for row in skips), skips
    ratios = [float(value) for value in re.findall(
        r'"max_effective_active_family_ratio": ([0-9.eE+-]+)',
        (args.run_root / "A/train.log").read_text(encoding="utf-8"),
    )]
    assert ratios and max(ratios) <= 0.1000001
    config = e15["resolved_scientific_config"]
    assert config["seed"] == 0 and config["use_safe_anchor"] is True
    assert config["anchor_lambda"] == 0.0021633926715180626
    assert config["anchor_family_budget"] == 0.1

    result = {
        "manifest_name": "H2_BF16_SEED0_A_E15_MANIFEST",
        "status": "PASS",
        "last_full_train": "A_BF16_SEED0_E15",
        "run_root": str(args.run_root),
        "checkpoints": {
            "shared_e1": {"path": str(shared), "sha256": sha256(shared), "epoch": 1, "global_step": 361},
            "A_E15": {"path": str(final), "sha256": sha256(final), "epoch": 15, "global_step": 5415},
        },
        "numerical_validity": {
            "precision": e15["precision"], "amp_enabled": e15["amp_enabled"],
            "gradscaler_enabled": e15["gradscaler_enabled"], "scaler_state": e15["scaler_state"],
            "tf32_enabled": e15["tf32_enabled"], "model_state_finite": True,
            "optimizer_state_finite": True, "nonfinite_loss_events": 0,
            "nonfinite_gradient_events": 0, "successful_optimizer_steps": 5415,
            "attempted_batches": 5415,
        },
        "anchor": {"lambda": config["anchor_lambda"], "family_budget": config["anchor_family_budget"],
                   "max_observed_effective_family_ratio": max(ratios)},
        "config_sha256": e15["config_sha256"], "scientific_config": config,
        "git_sha": e15["git_sha"], "implementation_git_sha": e15["implementation_git_sha"],
        "dataset_manifest_sha256": e15["dataset_manifest_sha256"], "clip_sha256": e15["clip_sha256"],
        "skip_counts": skips,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(
        "# H2 BF16 Seed0 A E15 numerical audit\n\n"
        "`A_BF16_SEED0_E15_VALIDITY=PASS`\n\n"
        "Fresh BF16 Seed0 E1 and resumed A E15 completed exactly 5,415 optimizer steps. "
        "All model/optimizer tensors are finite; loss/gradient nonfinite counts are zero; "
        "GradScaler is disabled and TF32 is disabled. The maximum observed effective Anchor "
        f"family ratio was `{max(ratios):.9g}`, within `rho=0.1`.\n",
        encoding="utf-8",
    )
    print("A_BF16_SEED0_E15_VALIDITY=PASS")


if __name__ == "__main__":
    main()
