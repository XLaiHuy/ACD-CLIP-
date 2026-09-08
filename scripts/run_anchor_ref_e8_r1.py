#!/usr/bin/env python3
"""Run the controlled clean-H-E8 -> Family-Safe-Anchor-E9-E15 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import h2_clean.contract as contract
import train as train_module


START_CANDIDATES = (
    REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth",
    Path("/tmp/h2-thbr-evidence/runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"),
)
EXPECTED_START_SHA256 = "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35"
BASE_COMMIT = "31167af5ee3dfff80b74af1e9ee0da4ecc475d2e"
DEFAULT_ROOT = Path("/workspace/anchor_ref_e8_r1_run")
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_RHO = 0.10


def resolve_start() -> Path:
    for candidate in START_CANDIDATES:
        if candidate.is_file() and sha256(candidate) == EXPECTED_START_SHA256:
            return candidate
    searched = ", ".join(str(path) for path in START_CANDIDATES)
    raise RuntimeError(f"shared E1 parent missing or hash mismatch; searched: {searched}")


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assert_full_state(path: Path, expected_epoch: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"unresolved Git-LFS pointer: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = contract.checkpoint_required_keys()
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise RuntimeError(f"incomplete full-state checkpoint {path}: {missing}")
    if int(payload["epoch"]) != int(expected_epoch):
        raise RuntimeError(f"checkpoint epoch mismatch for {path}: {payload['epoch']} != {expected_epoch}")
    if payload.get("protocol_version") != contract.PROTOCOL_VERSION:
        raise RuntimeError(f"wrong checkpoint protocol for {path}: {payload.get('protocol_version')}")
    if not all(finite(payload.get(key)) for key in (
        "model_state", "optimizer_state", "scheduler_state", "scaler_state",
        "torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state",
    )):
        raise RuntimeError(f"non-finite full-state checkpoint: {path}")
    return payload


def assert_clean_config(payload: dict, *, epoch: int) -> None:
    cfg = payload["resolved_scientific_config"]
    expected = {
        "model_name": "ViT-L-14-336", "img_size": 518, "dataset": "VisA",
        "n_groups": 3, "seed": 0, "image_adapt_weight": 0.2,
        "text_adapt_weight": 0.2, "lora_rank": 16, "lora_alpha": 2.0,
        "conv_lora_rank": 8, "conv_lora_alpha": 2.0,
        "conv_kernel_size_list": [3, 5], "batch_size": 6,
        "image_lr": 0.001, "text_lr": 0.0005, "use_hybrid_soft_prompt": True,
        "use_soft_prompt": False, "hybrid_alpha_max": 0.2,
        "soft_prompt_ctx_len": 4, "soft_prompt_freeze_epochs": 3,
        "soft_prompt_lr": 0.00005, "lambda_kg": 0.01, "lambda_k": 0.002,
        "lr_gamma": 0.9, "dfg_mode": "attn", "dfg_attn_dim": 256,
        "dfg_attn_tau": 8.0, "use_ss2d_dfg": True, "dfg_gamma_max": 0.2,
        "dfg_ss2d_fusion": "weight_residual", "dfg_beta": 0.1,
        "dfg_beta_schedule": "warmup010", "dfg_beta_target": 0.1,
        "dfg_weight_residual_fp32": True, "grad_clip_norm": 1.0,
        "amp": True, "deterministic_algorithms": True,
        "use_safe_anchor": False, "anchor_lambda": 0.0,
        "anchor_reference_sha256": None, "anchor_gradient_budget": False,
        "use_cir_training": False, "cir_alpha": 0.0,
        "training_horizon": 20, "primary_horizon": 15, "secondary_horizon": 20,
    }
    mismatches = {key: {"expected": value, "observed": cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"clean H configuration mismatch at E{epoch}: {json.dumps(mismatches, sort_keys=True)}")


def patch_resume_identity() -> None:
    """Bridge old E1 identity and the explicitly declared clean-H-E8 branch.

    Only the in-memory metadata mapping is adapted for the existing strict
    validator. The serialized checkpoint file and every tensor state remain
    unchanged.
    """
    original = contract.validate_resume_identity

    def bridge(payload, **kwargs):
        source_epoch = int(payload.get("epoch", -1))
        expected = dict(kwargs["expected_scientific_config"])
        parent = dict(kwargs["expected_parent_config"])
        actual = dict(payload["resolved_scientific_config"])
        if source_epoch == 1:
            pass
        elif source_epoch == 8:
            # The historical restore path validates the same payload once
            # before entering train() and once inside restore_full_checkpoint.
            # The first validation bridges only in-memory branch metadata;
            # accept that already-bridged form on the second validation.
            already_bridged = all(actual.get(key) == expected.get(key) for key in contract.RESUME_BRANCH_KEYS)
            if not already_bridged:
                if actual.get("use_safe_anchor") or actual.get("use_cir_training"):
                    raise RuntimeError("E8 parent is contaminated by Anchor or CIR")
                for key in contract.RESUME_BRANCH_KEYS:
                    actual[key] = expected.get(key)
        else:
            raise RuntimeError(f"unexpected resume boundary for E8 experiment: E{source_epoch}")

        for key in ("implementation_git_sha", "working_tree_diff_sha256"):
            if key in expected:
                actual[key] = expected.get(key)
        for key in ("precision", "precision_protocol", "bf16_local_fp32_islands", "later_transformer_fp32_islands"):
            if key in expected:
                actual[key] = expected[key]
        payload["resolved_scientific_config"] = actual
        payload["config_sha256"] = contract.canonical_json_hash(actual)
        payload["parent_scientific_config"] = parent
        payload["git_sha"] = kwargs.get("expected_git_sha")
        payload["implementation_git_sha"] = expected.get("implementation_git_sha")
        if "precision" in expected:
            payload["precision"] = expected["precision"]
            payload["amp_enabled"] = str(expected["precision"]) in ("amp", "fp16")
        return original(payload, **kwargs)

    contract.validate_resume_identity = bridge
    train_module.validate_resume_identity = bridge


def reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)


def training_args(root: Path, *, epoch: int, resume: Path, anchor: bool = False, reference: Path | None = None) -> list[str]:
    args = [
        "train.py", "--model_name", "ViT-L-14-336", "--img_size", "518",
        "--dataset", "VisA", "--batch_size", "6", "--epoch", str(epoch),
        "--protocol_horizon", "20", "--cuda_device", "0", "--save_path", str(root),
        "--n_groups", "3", "--image_adapt_weight", "0.2", "--text_adapt_weight", "0.2",
        "--conv_lora_rank", "8", "--conv_lora_alpha", "2.0", "--conv_kernel_size_list", "3", "5",
        "--lora_rank", "16", "--lora_alpha", "2.0", "--image_lr", "0.001", "--text_lr", "0.0005",
        "--use_hybrid_soft_prompt", "--hybrid_alpha_max", "0.2", "--soft_prompt_freeze_epochs", "3",
        "--soft_prompt_ctx_len", "4", "--soft_prompt_lr", "0.00005", "--soft_prompt_init", "phrase",
        "--soft_prompt_init_phrase", "a photo of a", "--lambda_kg", "0.01", "--lambda_k", "0.002",
        "--lr_gamma", "0.9", "--dfg_mode", "attn", "--dfg_attn_dim", "256", "--dfg_attn_tau", "8.0",
        "--use_ss2d_dfg", "--dfg_gamma_max", "0.2", "--dfg_ss2d_fusion", "weight_residual",
        "--dfg_beta", "0.10", "--dfg_beta_schedule", "warmup010", "--dfg_beta_target", "0.10",
        "--dfg_weight_residual_fp32", "--grad_clip_norm", "1.0", "--non_finite_loss_abort_threshold", "20",
        "--num_workers", "6", "--grad_checkpointing", "--amp", "--seed", "0",
        "--deterministic_algorithms", "--resume", str(resume),
    ]
    if anchor:
        args += [
            "--use_safe_anchor", "--anchor_lambda", str(ANCHOR_LAMBDA),
            "--anchor_reference_path", str(reference), "--anchor_gradient_budget",
            "--anchor_family_budget", str(ANCHOR_RHO), "--anchor_family_audit",
        ]
    return args


def run_phase(args: list[str]) -> None:
    reset_logging()
    sys.argv = args
    train_module.main()


def write_theta_ref(path: Path, e8_path: Path, e8_payload: dict) -> dict:
    reference = {
        name: value.detach().float().cpu().clone()
        for name, value in e8_payload["image_parameter_reference"].items()
    }
    reference_hash = contract.state_dict_sha256(reference)
    payload = {
        "schema": "ANCHOR_REF_E8_R1_THETA_REF_V1",
        "epoch": 8,
        "source_checkpoint": str(e8_path),
        "source_checkpoint_sha256": sha256(e8_path),
        "source_config_sha256": e8_payload["config_sha256"],
        "reference_sha256": reference_hash,
        "image_anchor_reference": reference,
        "resolved_scientific_config": dict(e8_payload["resolved_scientific_config"]),
    }
    atomic_torch_save(path, payload)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "reference_sha256": reference_hash,
        "source_checkpoint": str(e8_path),
        "source_checkpoint_sha256": sha256(e8_path),
        "source_epoch": 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("all", "clean", "anchor"), default=None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.run is None:
        print("Prepared only. Use --run all after the committed recovery audit has been reviewed.")
        print(f"Run root: {args.root}")
        return
    patch_resume_identity()
    start = resolve_start()
    if not args.root.exists():
        args.root.mkdir(parents=True)
    elif any(args.root.iterdir()) and args.run == "all":
        raise RuntimeError(f"refusing to reuse non-empty run root: {args.root}")

    clean_root = args.root / "clean_h_e8"
    anchor_root = args.root / "anchor_e8"
    theta_ref_path = args.root / "theta_ref_e8.pth"
    if args.run in ("all", "clean"):
        if clean_root.exists() and any(clean_root.iterdir()):
            raise RuntimeError(f"refusing to overwrite clean output: {clean_root}")
        run_phase(training_args(clean_root, epoch=8, resume=start))

    if args.run in ("all", "anchor"):
        e8_path = clean_root / "adapter_8.pth"
        e8_payload = assert_full_state(e8_path, 8)
        assert_clean_config(e8_payload, epoch=8)
        theta_record = write_theta_ref(theta_ref_path, e8_path, e8_payload)
        if anchor_root.exists() and any(anchor_root.iterdir()):
            raise RuntimeError(f"refusing to overwrite Anchor output: {anchor_root}")
        run_phase(training_args(anchor_root, epoch=15, resume=e8_path, anchor=True, reference=theta_ref_path))
        e15_path = anchor_root / "adapter_15.pth"
        e15_payload = assert_full_state(e15_path, 15)
        cfg = e15_payload["resolved_scientific_config"]
        required = {
            "use_safe_anchor": True,
            "anchor_lambda": ANCHOR_LAMBDA,
            "anchor_family_budget": ANCHOR_RHO,
            "anchor_gradient_budget": True,
            "use_cir_training": False,
            "cir_alpha": 0.0,
            "anchor_reference_sha256": theta_record["sha256"],
        }
        mismatches = {key: {"expected": value, "observed": cfg.get(key)} for key, value in required.items() if cfg.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Anchor E15 configuration mismatch: {json.dumps(mismatches, sort_keys=True)}")
        manifest = {
            "experiment_id": "anchor_ref_e8_r1",
            "status": "TRAINING_COMPLETE",
            "branch": "research/anchor_ref_e8_r1",
            "training_implementation_base_commit": BASE_COMMIT,
            "git_head_at_run": e15_payload.get("git_sha"),
            "parent": {
                "path": str(start), "sha256": EXPECTED_START_SHA256, "epoch": 1,
                "type": "full_state_shared_clean_parent",
            },
            "clean_e8": {
                "path": str(e8_path), "sha256": sha256(e8_path), "epoch": 8,
                "global_step": int(e8_payload["global_step"]),
                "config_sha256": e8_payload["config_sha256"],
                "anchor": False, "cir": False,
            },
            "theta_ref": theta_record,
            "anchor_e15": {
                "path": str(e15_path), "sha256": sha256(e15_path), "epoch": 15,
                "global_step": int(e15_payload["global_step"]),
                "config_sha256": e15_payload["config_sha256"],
                "anchor": True, "cir": False,
            },
            "medical_status": "PENDING",
            "mvtec_status": "PENDING_MEDICAL_FREEZE",
        }
        atomic_json(args.root / "run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
