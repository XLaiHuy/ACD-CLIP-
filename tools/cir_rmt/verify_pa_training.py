#!/usr/bin/env python3
"""Verify the completed PA control without loading model weights to GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from model.phase2b_runtime import load_json_config
from tools.cir_rmt.parameter_anchor import sha256_file


EPOCHS = (10, 12, 14, 16, 18, 20)
CONTROL_ID = "PA_PHASE2B_IMAGE_ANCHOR_V1"


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    config.update({
        "micro_batch_size": 6,
        "batch_size": 6,
        "grad_accum_steps": 1,
        "effective_batch_size": 6,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    })
    return config


def _pretty_config_sha(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve() / "visa" / "seed0"
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _effective_config(load_json_config(args.config.expanduser().resolve()))
    expected_config_sha = _pretty_config_sha(config)
    parent_manifest = json.loads((args.parent_run_root.expanduser().resolve() / "phase2b" / "run_manifest.json").read_text(encoding="utf-8"))
    if expected_config_sha != parent_manifest.get("config_sha256"):
        raise ValueError(f"PA effective config SHA {expected_config_sha} != frozen parent {parent_manifest.get('config_sha256')}")
    if manifest.get("status") != "COMPLETED" or manifest.get("control_id") != CONTROL_ID:
        raise ValueError("PA manifest is not completed with the expected control identity")
    if manifest.get("target_tuning_occurred") is not False or manifest.get("mvtec") != "NOT_RUN":
        raise ValueError("PA manifest target/MVTec policy mismatch")
    expected_anchor_sha = sha256_file(args.anchor_checkpoint.expanduser().resolve())
    if expected_anchor_sha != "3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2":
        raise ValueError("PA anchor reference SHA mismatch")
    expected = []
    for epoch in EPOCHS:
        path = run_root / "checkpoints" / f"adapter_{epoch}.pth"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        control = payload.get("control_identity", {})
        anchor = payload.get("image_anchor", {})
        optimizer = payload.get("optimizer_state", {})
        scheduler = payload.get("scheduler_state", {})
        groups = optimizer.get("param_groups", [])
        if payload.get("epoch") != epoch:
            raise ValueError(f"checkpoint epoch mismatch at E{epoch}")
        if payload.get("precision") != "fp32" or payload.get("amp_enabled") is not False or payload.get("tf32_enabled") is not False:
            raise ValueError(f"precision mismatch at E{epoch}")
        if payload.get("config_sha256") != expected_config_sha:
            raise ValueError(f"config SHA mismatch at E{epoch}")
        if control.get("control_id") != CONTROL_ID or control.get("cir_training") is not False or control.get("rmt_training") is not False:
            raise ValueError(f"PA control identity mismatch at E{epoch}")
        if control.get("training_forward") != "native_phase2b":
            raise ValueError(f"PA native-forward identity mismatch at E{epoch}")
        if abs(float(anchor.get("lambda_image_anchor", 0.0)) - 0.001) > 1e-15 or anchor.get("train_only") is not True or anchor.get("scope") != "image_adapter_parameters_only":
            raise ValueError(f"anchor identity mismatch at E{epoch}")
        if anchor.get("reference_checkpoint_sha256") != expected_anchor_sha or int(anchor.get("reference_epoch", -1)) != 14:
            raise ValueError(f"anchor reference mismatch at E{epoch}")
        if len(groups) != 3 or [group.get("name") for group in groups] != ["image_adapter", "text_adapter", "soft_prompt"]:
            raise ValueError(f"optimizer group mismatch at E{epoch}")
        if optimizer.get("state") is None or scheduler.get("last_epoch") != epoch or scheduler.get("_step_count") != epoch + 1:
            raise ValueError(f"optimizer/scheduler state mismatch at E{epoch}")
        if abs(float(groups[0]["lr"]) - 0.001 * (0.9 ** epoch)) > 1e-15:
            raise ValueError(f"image LR mismatch at E{epoch}")
        if abs(float(groups[1]["lr"]) - 0.0005 * (0.9 ** epoch)) > 1e-15:
            raise ValueError(f"text LR mismatch at E{epoch}")
        if abs(float(groups[2]["lr"]) - 9e-5) > 1e-15:
            raise ValueError(f"soft-prompt LR mismatch at E{epoch}")
        defaults = optimizer.get("param_groups", [{}])[0]
        expected.append({
            "epoch": epoch,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "optimizer_lr": [float(group["lr"]) for group in groups],
            "scheduler_last_epoch": int(scheduler["last_epoch"]),
            "scheduler_step_count": int(scheduler["_step_count"]),
            "adam_betas": defaults.get("betas"),
            "adam_eps": defaults.get("eps"),
            "weight_decay": defaults.get("weight_decay"),
        })
    candidate_names = sorted(path.name for path in (run_root / "checkpoints").glob("*.pth"))
    if candidate_names != [f"adapter_{epoch}.pth" for epoch in EPOCHS]:
        raise ValueError(f"unexpected PA checkpoint files: {candidate_names}")
    history = manifest.get("history", [])
    if [int(row["epoch"]) for row in history] != list(range(1, 21)):
        raise ValueError("PA manifest history is not the complete E1-E20 sequence")
    if [int(row["epoch"]) for row in history if row.get("checkpoint_saved")] != list(EPOCHS):
        raise ValueError("PA candidate checkpoint history is incomplete")
    output = {
        "status": "PASS",
        "control_id": CONTROL_ID,
        "run_root": str(run_root),
        "config_sha256": expected_config_sha,
        "parent_config_sha256": "d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb",
        "anchor_reference_checkpoint_sha256": expected_anchor_sha,
        "anchor_lambda_image": 0.001,
        "epochs": expected,
        "candidate_epochs": list(EPOCHS),
        "adam_defaults": {
            "betas": expected[0]["adam_betas"],
            "eps": expected[0]["adam_eps"],
            "weight_decay": expected[0]["weight_decay"],
        },
        "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.9, "timing": "after_epoch_before_checkpoint"},
        "precision": {"precision": "fp32", "amp": False, "tf32": False},
        "resume": {"resume_from_epoch": manifest.get("resume_from_epoch"), "last_checkpoint": str(run_root / "last.pth")},
        "target_tuning_occurred": False,
        "mvtec": "NOT_RUN",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = verify(parser.parse_args())
    print(json.dumps({"status": result["status"], "epochs": result["candidate_epochs"], "anchor_sha256": result["anchor_reference_checkpoint_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
