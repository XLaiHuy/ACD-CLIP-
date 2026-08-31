#!/usr/bin/env python3
"""Audit the resumed E14 image-anchor run through the E20 cursor.

This verifier is read-only with respect to checkpoints.  It checks the exact
identity, optimizer, scheduler, precision, anchor, RNG, and candidate policy
contracts before any source or target result is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tools.cir_rmt.identity import config_sha256, load_cir_config, validate_checkpoint_identity


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
GROUPS = ("image_adapter", "text_adapter", "soft_prompt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _audit_checkpoint(
    path: Path,
    *,
    epoch: int,
    config: dict[str, Any],
    training_git_sha: str,
    anchor_sha: str,
) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing candidate checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(payload, config, source_dataset="visa", expected_git_sha=training_git_sha, expected_epoch=epoch)
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
        _fail(f"FP32 contract failed at E{epoch}")
    anchor = payload.get("image_anchor", {})
    if anchor.get("enabled") is not True or float(anchor.get("lambda_image_anchor", 0.0)) != 0.001:
        _fail(f"image-anchor coefficient failed at E{epoch}")
    if anchor.get("reference_checkpoint_sha256") != anchor_sha or int(anchor.get("reference_epoch", -1)) != 14:
        _fail(f"image-anchor reference failed at E{epoch}")
    groups = payload.get("optimizer_state", {}).get("param_groups", [])
    names = [str(group.get("name")) for group in groups]
    if names != list(GROUPS):
        _fail(f"optimizer group identity failed at E{epoch}: {names}")
    expected_lrs = (0.001 * (0.9**epoch), 0.0005 * (0.9**epoch), 9.0e-5)
    for index, group in enumerate(groups):
        if abs(float(group.get("lr", -1.0)) - expected_lrs[index]) > 1.0e-15:
            _fail(f"optimizer LR failed at E{epoch}, group={names[index]}")
        if tuple(float(value) for value in group.get("betas", ())) != (0.9, 0.999):
            _fail(f"Adam betas failed at E{epoch}, group={names[index]}")
        if abs(float(group.get("eps", -1.0)) - 1.0e-8) > 1.0e-20:
            _fail(f"Adam eps failed at E{epoch}, group={names[index]}")
        if abs(float(group.get("weight_decay", -1.0))) > 1.0e-20:
            _fail(f"weight decay failed at E{epoch}, group={names[index]}")
    scheduler = payload.get("scheduler_state", {})
    if int(scheduler.get("last_epoch", -1)) != epoch or int(scheduler.get("_step_count", -1)) != epoch + 1:
        _fail(f"scheduler state failed at E{epoch}: {scheduler}")
    for key in ("torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state"):
        if key not in payload:
            _fail(f"resume RNG state missing at E{epoch}: {key}")
    return {
        "epoch": epoch,
        "path": str(path),
        "checkpoint_sha256": _sha256(path),
        "git_sha": payload.get("git_sha"),
        "config_sha256": payload.get("config_sha256"),
        "optimizer_group_names": names,
        "optimizer_lrs": [float(group["lr"]) for group in groups],
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "anchor_reference_sha256": anchor_sha,
        "rng_state_present": True,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    config = load_cir_config(args.config.expanduser().resolve())
    checkpoint_root = run_root / "visa" / "seed0" / "checkpoints"
    last_path = run_root / "visa" / "seed0" / "last.pth"
    anchor_path = args.anchor_checkpoint.expanduser().resolve()
    anchor_sha = _sha256(anchor_path)
    if args.expected_anchor_sha256 and anchor_sha != args.expected_anchor_sha256:
        _fail(f"anchor checkpoint SHA mismatch: {anchor_sha}")
    if not args.training_git_sha:
        manifest = json.loads((run_root / "visa" / "seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        training_git_sha = str(manifest.get("git_sha", ""))
    else:
        training_git_sha = str(args.training_git_sha)
    if not training_git_sha:
        _fail("training git SHA is empty")
    rows = [_audit_checkpoint(checkpoint_root / f"epoch_{epoch:02d}.pth", epoch=epoch, config=config, training_git_sha=training_git_sha, anchor_sha=anchor_sha) for epoch in EPOCHS]
    if not last_path.is_file():
        _fail(f"missing final resume cursor: {last_path}")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(last, config, source_dataset="visa", expected_git_sha=training_git_sha, expected_epoch=20)
    if int(last.get("epoch", -1)) != 20:
        _fail(f"final resume cursor is not E20: {last.get('epoch')}")
    if last.get("scheduler_state", {}).get("last_epoch") != 20:
        _fail("final resume cursor scheduler state is not E20")
    manifest_path = run_root / "visa" / "seed0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history_epochs = [int(row["epoch"]) for row in manifest.get("history", [])]
    if history_epochs != list(range(1, 21)):
        _fail(f"manifest history is not contiguous E1-E20: {history_epochs}")
    if manifest.get("status") != "COMPLETED" or int(manifest.get("max_epoch", -1)) != 20:
        _fail("extension run manifest is not COMPLETED through E20")
    if [int(value) for value in manifest.get("target_epochs", [])] != list(EPOCHS):
        _fail("extension candidate policy is not E10/E12/E14/E16/E18/E20")
    if any((checkpoint_root / f"epoch_{epoch:02d}.pth").exists() for epoch in (11, 13, 15, 17, 19)):
        _fail("unexpected non-candidate checkpoint exists")
    result = {
        "status": "PASS",
        "scope": "CIR_V2_IMAGE_ANCHOR_RESUME_E14_TO_E20",
        "arch_id": config["arch_id"],
        "config_sha256": config_sha256(config),
        "architecture_freeze_sha256": config["architecture_freeze_sha256"],
        "training_git_sha": training_git_sha,
        "anchor_checkpoint_sha256": anchor_sha,
        "resume_cursor_before_extension": {"epoch": 14, "path": str(run_root / "visa" / "seed0" / "last.pth")},
        "final_cursor": {"epoch": 20, "path": str(last_path), "sha256": _sha256(last_path)},
        "candidate_checkpoints": rows,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_status": manifest.get("status"),
        "history_epochs": history_epochs,
        "target_evaluation": "NOT_RUN",
        "mvtec": "NOT_RUN",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-git-sha")
    parser.add_argument("--expected-anchor-sha256")
    audit(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
