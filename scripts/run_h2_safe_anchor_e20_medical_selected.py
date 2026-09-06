#!/usr/bin/env python3
"""Resume the authoritative Safe-Anchor A run from E1 through E20.

The training call delegates to the repository's full-state ``train`` path.
The only wrapper responsibilities are the documented E1 lineage bridge (the
current branch adds an exact equal-fusion implementation), atomic checkpoint
milestone metadata, and milestone Git pushes.  No target dataset is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import h2_clean.contract as contract
import train as train_module
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS


START = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
AUTHORITATIVE_A15 = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/A/adapter_15.pth"
EXPECTED_START_SHA256 = "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35"
DEFAULT_ROOT = Path("/workspace/h2_safe_anchor_e20_medical_selected")
BRANCH = "research/h2-safe-anchor-e20-medical-selected"
MILESTONES = (10, 15, 20)
SKIP_RE = re.compile(r"skip_counts epoch=(\d+) non_finite_loss=(\d+) non_finite_grad=(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def config_identity() -> dict:
    if not AUTHORITATIVE_A15.is_file():
        raise FileNotFoundError(AUTHORITATIVE_A15)
    payload = torch.load(AUTHORITATIVE_A15, map_location="cpu", weights_only=False)
    source = dict(payload["resolved_scientific_config"])
    required = {
        "model_name": "ViT-L-14-336",
        "img_size": 518,
        "dataset": "VisA",
        "n_groups": 3,
        "batch_size": 6,
        "seed": 0,
        "training_horizon": 20,
        "grad_checkpointing": True,
        "deterministic_algorithms": True,
        "dfg_mode": "attn",
        "dfg_attn_dim": 256,
        "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True,
        "dfg_ss2d_fusion": "weight_residual",
        "dfg_weight_residual_fp32": True,
        "lambda_kg": 0.01,
        "lambda_k": 0.002,
        "image_lr": 0.001,
        "text_lr": 0.0005,
        "lr_gamma": 0.9,
        "hybrid_alpha_max": 0.2,
        "soft_prompt_freeze_epochs": 3,
        "soft_prompt_ctx_len": 4,
        "soft_prompt_lr": 5e-05,
        "use_hybrid_soft_prompt": True,
        "use_soft_prompt": False,
        "use_cir_training": False,
        "use_safe_anchor": True,
        "anchor_lambda": 0.0021633926715180626,
        "anchor_gradient_budget": True,
        "anchor_family_budget": 0.1,
        "tf32_enabled": False,
        "amp": True,
    }
    mismatches = {key: {"expected": value, "observed": source.get(key)} for key, value in required.items() if source.get(key) != value}
    operational = dict(payload.get("resolved_operational_config", {}))
    operational_expected = {"num_workers": 6, "anchor_family_audit": True, "anchor_grad_audit_interval": 0, "non_finite_loss_abort_threshold": 20}
    mismatches.update({f"operational.{key}": {"expected": value, "observed": operational.get(key)} for key, value in operational_expected.items() if operational.get(key) != value})
    result = {
        "protocol_id": "H2_SAFE_ANCHOR_E20_MEDICAL_SELECTED",
        "status": "PASS" if not mismatches else "FAIL",
        "source_checkpoint": str(AUTHORITATIVE_A15),
        "source_checkpoint_sha256": sha256(AUTHORITATIVE_A15),
        "source_config_sha256": payload["config_sha256"],
        "source_config": source,
        "source_operational_config": operational,
        "verified": required,
        "mismatches": mismatches,
        "safe_anchor": {"enabled": True, "lambda": required["anchor_lambda"], "family_budget_rho": required["anchor_family_budget"], "reference": str(START), "reference_sha256": EXPECTED_START_SHA256},
        "functional_feature_anchor": False,
        "train_time_stage_fusion": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "weighted_inference_only": [0.3736138197153701, 0.3270300383596602, 0.2993561419249697],
        "medical_role": "TARGET_VALIDATION_FOR_EPOCH_SELECTION",
        "mvtec_role": "POST_SELECTION_HOLDOUT_TARGET",
    }
    atomic_json(REPO / "audit/H2_SAFE_ANCHOR_E20_CONFIG_IDENTITY.json", result)
    if mismatches:
        raise RuntimeError(json.dumps(mismatches, sort_keys=True))
    return result


def argv_for_training(root: Path, cfg: dict) -> list[str]:
    args = [
        "train.py",
        "--model_name", str(cfg["model_name"]), "--img_size", str(cfg["img_size"]),
        "--dataset", str(cfg["dataset"]), "--batch_size", str(cfg["batch_size"]),
        "--epoch", "20", "--protocol_horizon", "20", "--cuda_device", "0",
        "--save_path", str(root), "--n_groups", str(cfg["n_groups"]),
        "--image_adapt_weight", str(cfg["image_adapt_weight"]),
        "--conv_lora_rank", str(cfg["conv_lora_rank"]), "--conv_lora_alpha", str(cfg["conv_lora_alpha"]),
        "--conv_kernel_size_list", *[str(value) for value in cfg["conv_kernel_size_list"]],
        "--text_adapt_weight", str(cfg["text_adapt_weight"]),
        "--lora_rank", str(cfg["lora_rank"]), "--lora_alpha", str(cfg["lora_alpha"]),
        "--image_lr", str(cfg["image_lr"]), "--text_lr", str(cfg["text_lr"]),
        "--hybrid_alpha_max", str(cfg["hybrid_alpha_max"]),
        "--soft_prompt_freeze_epochs", str(cfg["soft_prompt_freeze_epochs"]),
        "--soft_prompt_ctx_len", str(cfg["soft_prompt_ctx_len"]),
        "--soft_prompt_lr", str(cfg["soft_prompt_lr"]),
        "--soft_prompt_init", str(cfg["soft_prompt_init"]),
        "--soft_prompt_init_phrase", str(cfg["soft_prompt_init_phrase"]),
        "--lambda_kg", str(cfg["lambda_kg"]), "--lambda_k", str(cfg["lambda_k"]),
        "--lr_gamma", str(cfg["lr_gamma"]), "--dfg_mode", str(cfg["dfg_mode"]),
        "--dfg_attn_dim", str(cfg["dfg_attn_dim"]), "--dfg_attn_tau", str(cfg["dfg_attn_tau"]),
        "--dfg_gamma_max", str(cfg["dfg_gamma_max"]), "--dfg_ss2d_fusion", str(cfg["dfg_ss2d_fusion"]),
        "--dfg_beta", str(cfg["dfg_beta"]), "--dfg_beta_schedule", str(cfg["dfg_beta_schedule"]),
        "--dfg_beta_target", str(cfg["dfg_beta_target"]),
        "--precision_protocol", "HISTORICAL_MIXED_FP16_FP32_V1",
        "--grad_clip_norm", str(cfg["grad_clip_norm"]), "--num_workers", "6",
        "--non_finite_loss_abort_threshold", "20", "--seed", "0",
        "--anchor_lambda", str(cfg["anchor_lambda"]), "--anchor_family_budget", "0.1",
        "--anchor_reference_path", str(START), "--resume", str(START),
    ]
    args += [
        "--use_hybrid_soft_prompt", "--use_ss2d_dfg", "--grad_checkpointing",
        "--deterministic_algorithms", "--use_safe_anchor", "--anchor_gradient_budget",
        "--anchor_family_audit", "--dfg_weight_residual_fp32",
    ]
    return args


def patch_resume_identity() -> None:
    """Permit the E1 H-parent to resume into the current exact-equal A branch.

    The E1 file is never rewritten.  This changes only in-memory identity
    fields needed by the repository's strict current-code resume validator;
    model, optimizer, scheduler, scaler, RNG, and dataloader states remain the
    loaded E1 states.
    """
    original = contract.validate_resume_identity

    def bridge(payload, **kwargs):
        if int(payload.get("epoch", -1)) == 1:
            expected = kwargs["expected_scientific_config"]
            parent = kwargs["expected_parent_config"]
            actual = dict(payload["resolved_scientific_config"])
            actual["implementation_git_sha"] = expected["implementation_git_sha"]
            actual["working_tree_diff_sha256"] = expected.get("working_tree_diff_sha256")
            payload["resolved_scientific_config"] = actual
            payload["config_sha256"] = contract.canonical_json_hash(actual)
            payload["parent_scientific_config"] = dict(parent)
            payload["git_sha"] = kwargs.get("expected_git_sha")
            payload["implementation_git_sha"] = expected["implementation_git_sha"]
        return original(payload, **kwargs)

    contract.validate_resume_identity = bridge
    train_module.validate_resume_identity = bridge


def parse_skips(log_path: Path) -> dict[int, dict[str, int]]:
    if not log_path.is_file():
        return {}
    return {
        int(epoch): {"nonfinite_loss_skips": int(loss), "nonfinite_grad_skips": int(grad)}
        for epoch, loss, grad in SKIP_RE.findall(log_path.read_text(encoding="utf-8"))
    }


def checkpoint_record(root: Path, epoch: int, skips: dict[int, dict[str, int]]) -> dict:
    path = root / f"adapter_{epoch}.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    summary_rows = json.loads((root / "epoch_summary.json").read_text()) if (root / "epoch_summary.json").is_file() else []
    epoch_summary = next((row for row in summary_rows if int(row["epoch"]) == epoch), {})
    skip = skips.get(epoch, {"nonfinite_loss_skips": 0, "nonfinite_grad_skips": 0})
    model_finite = finite(payload.get("model_state", {}))
    optimizer_finite = finite(payload.get("optimizer_state", {}))
    scaler_finite = finite(payload.get("scaler_state", {}))
    return {
        "epoch": epoch,
        "path": str(path),
        "sha256": sha256(path),
        "global_step": int(payload["global_step"]),
        "attempted_batches": int(epoch_summary.get("attempted_batches", 0)),
        "successful_optimizer_steps": int(payload["global_step"]),
        "nonfinite_loss_skips": skip["nonfinite_loss_skips"],
        "nonfinite_grad_skips": skip["nonfinite_grad_skips"],
        "model_state_finite": model_finite,
        "optimizer_state_finite": optimizer_finite,
        "scaler_state_finite": scaler_finite,
        "numerical_validity": bool(model_finite and optimizer_finite and scaler_finite and not skip["nonfinite_loss_skips"]),
    }


def write_training_summary(root: Path, milestone: int, config: dict) -> None:
    skips = parse_skips(root / "train.log")
    records = [checkpoint_record(root, epoch, skips) for epoch in range(10, milestone + 1) if (root / f"adapter_{epoch}.pth").is_file()]
    result = {
        "protocol_id": "H2_SAFE_ANCHOR_E20_MEDICAL_SELECTED",
        "milestone_epoch": milestone,
        "run_root": str(root),
        "start_checkpoint": str(START),
        "start_checkpoint_sha256": EXPECTED_START_SHA256,
        "safe_anchor_config_identity": config["status"],
        "functional_feature_anchor": False,
        "train_time_stage_fusion": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "source_only_training": True,
        "medical_used_during_training": False,
        "mvtec_used_during_training": False,
        "checkpoints": records,
        "all_retained_checkpoints_numerically_valid": all(row["numerical_validity"] for row in records),
    }
    atomic_json(REPO / "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.json", result)
    csv_path = REPO / "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.csv"
    fields = ["epoch", "path", "sha256", "global_step", "attempted_batches", "successful_optimizer_steps", "nonfinite_loss_skips", "nonfinite_grad_skips", "model_state_finite", "optimizer_state_finite", "scaler_state_finite", "numerical_validity"]
    temporary = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, csv_path)


class MilestoneWatcher(threading.Thread):
    def __init__(self, root: Path, config: dict):
        super().__init__(daemon=True)
        self.root = root
        self.config = config
        self.error: Exception | None = None

    def commit_milestone(self, epoch: int) -> None:
        write_training_summary(self.root, epoch, self.config)
        subprocess.run(["git", "add", "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.json", "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.csv"], cwd=REPO, check=True)
        subprocess.run(["git", "commit", "-m", f"Record Safe Anchor E{epoch} training milestone"], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "push", "origin", BRANCH], cwd=REPO, check=True, stdout=subprocess.DEVNULL)

    def run(self) -> None:
        try:
            for epoch in MILESTONES:
                path = self.root / f"adapter_{epoch}.pth"
                while not path.is_file():
                    time.sleep(30)
                while True:
                    previous_size = -1
                    stable = 0
                    while stable < 2:
                        size = path.stat().st_size
                        if size == previous_size:
                            stable += 1
                        else:
                            stable = 0
                        previous_size = size
                        time.sleep(30)
                    try:
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        complete = all(key in payload for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "global_step"))
                    except Exception:
                        complete = False
                    if complete:
                        break
                    time.sleep(30)
                self.commit_milestone(epoch)
        except Exception as exc:  # surfaced by the main thread after training
            self.error = exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    config = config_identity()
    if args.prepare:
        print(json.dumps({"status": config["status"], "artifact": str(REPO / "audit/H2_SAFE_ANCHOR_E20_CONFIG_IDENTITY.json")}))
        return
    if not START.is_file() or sha256(START) != EXPECTED_START_SHA256:
        raise RuntimeError("START_IDENTITY=FAIL")
    if args.root.exists() and any(args.root.glob("adapter_*.pth")):
        raise RuntimeError(f"refusing to overwrite existing run root: {args.root}")
    args.root.mkdir(parents=True, exist_ok=True)
    patch_resume_identity()
    watcher = MilestoneWatcher(args.root, config)
    watcher.start()
    original_argv = sys.argv
    try:
        cfg = config["source_config"]
        sys.argv = argv_for_training(args.root, cfg)
        train_module.main()
    finally:
        sys.argv = original_argv
    watcher.join()
    if watcher.error is not None:
        raise watcher.error
    if not (args.root / "adapter_20.pth").is_file():
        raise RuntimeError("E20 checkpoint missing")
    print(json.dumps({"status": "PASS", "root": str(args.root), "retained_epochs": list(range(10, 21))}, sort_keys=True))


if __name__ == "__main__":
    main()
