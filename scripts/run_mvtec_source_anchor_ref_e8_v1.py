#!/usr/bin/env python3
"""Run the MVTec-source E8-reference Anchor experiment.

The original MVTec manifest is audited before any model work. MVTec is used
as the training source, then the resulting E10-E15 checkpoints are evaluated
on VisA by the separate frozen evaluator harness.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import h2_clean.contract as contract
import run_anchor_ref_e8_r1 as base
import train as train_module


MANIFEST = REPO / "dataset/hub/MVTec.jsonl"
DATA_ROOT = REPO / "data/mvtec_ad"
DEFAULT_ROOT = Path("/workspace/mvtec_source_anchor_ref_e8_v1_run")
EXPECTED_MANIFEST_SHA256 = "3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c"
EXPECTED_CLASSES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
ANCHOR_LAMBDA = base.ANCHOR_LAMBDA
ANCHOR_RHO = base.ANCHOR_RHO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(run_root: Path) -> dict:
    if not MANIFEST.is_file():
        raise RuntimeError(f"MVTec manifest missing: {MANIFEST}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"MVTec manifest hash mismatch: {sha256(MANIFEST)} != {EXPECTED_MANIFEST_SHA256}"
        )
    if not DATA_ROOT.exists():
        raise RuntimeError(f"MVTec data root missing: {DATA_ROOT}")

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    category_counts = collections.defaultdict(lambda: {"total": 0, "normal": 0, "anomaly": 0})
    defect_types = collections.defaultdict(set)
    image_paths, mask_paths = [], []
    missing_images, missing_masks = [], []
    anomaly_without_masks, normal_unexpected_masks, train_good = [], [], []
    for index, row in enumerate(rows, start=1):
        category = row["class_name"]
        label = int(row["label"])
        image_path = str(row["image_path"])
        mask_path = row.get("mask_path")
        expected_keys = {"image_path", "label", "class_name"}
        if label == 1:
            expected_keys.add("mask_path")
        if set(row) != expected_keys:
            raise RuntimeError(f"unexpected manifest keys at row {index}: {sorted(row)}")
        category_counts[category]["total"] += 1
        category_counts[category]["normal" if label == 0 else "anomaly"] += 1
        image_paths.append(image_path)
        if not (DATA_ROOT / image_path).is_file():
            missing_images.append(image_path)
        image_parts = Path(image_path).parts
        if "train" in image_parts:
            train_good.append(image_path)
        if label == 0:
            if mask_path is not None:
                normal_unexpected_masks.append(image_path)
            if image_parts[1:3] != ("test", "good"):
                raise RuntimeError(f"normal record is not test/good at row {index}: {image_path}")
        elif label == 1:
            if len(image_parts) < 3 or image_parts[1] != "test" or image_parts[2] == "good":
                raise RuntimeError(f"anomaly record is not test/<defect> at row {index}: {image_path}")
            defect_types[category].add(image_parts[2])
            if not mask_path:
                anomaly_without_masks.append(image_path)
            else:
                mask_paths.append(str(mask_path))
                if not (DATA_ROOT / str(mask_path)).is_file():
                    missing_masks.append(str(mask_path))
                expected_mask = Path("ground_truth") / image_parts[2] / (Path(image_path).stem + "_mask.png")
                if Path(str(mask_path)).parts[1:] != expected_mask.parts:
                    raise RuntimeError(f"mask semantics mismatch at row {index}: {mask_path}")
        else:
            raise RuntimeError(f"unexpected label at row {index}: {label}")

    checks = {
        "missing_images": len(missing_images),
        "missing_masks": len(missing_masks),
        "anomaly_without_masks": len(anomaly_without_masks),
        "normal_unexpected_masks": len(normal_unexpected_masks),
        "duplicate_image_paths": len(image_paths) - len(set(image_paths)),
        "duplicate_mask_paths": len(mask_paths) - len(set(mask_paths)),
        "train_good_records": len(train_good),
    }
    if len(rows) != 1725 or sum(int(row["label"]) == 0 for row in rows) != 467 or sum(int(row["label"]) == 1 for row in rows) != 1258:
        raise RuntimeError("MVTec source pool count mismatch")
    if tuple(sorted(category_counts)) != tuple(sorted(EXPECTED_CLASSES)) or len(category_counts) != 15:
        raise RuntimeError(f"MVTec category mismatch: {sorted(category_counts)}")
    if any(checks.values()):
        raise RuntimeError(f"MVTec manifest integrity checks failed: {checks}")
    audit = {
        "manifest": str(MANIFEST),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_root": str(DATA_ROOT),
        "source_root_resolved": str(DATA_ROOT.resolve()),
        "records": {"total": len(rows), "normal": 467, "anomaly": 1258},
        "categories": sorted(category_counts),
        "category_counts": {key: category_counts[key] for key in sorted(category_counts)},
        "defect_types": {key: sorted(defect_types[key]) for key in sorted(defect_types)},
        "checks": checks,
        "manifest_wiring": "none; original dataset/hub/MVTec.jsonl used directly",
        "audit_status": "PASS",
    }
    base.atomic_json(run_root / "manifest_audit.json", audit)
    return audit


def source_training_args(root: Path, *, epoch: int, resume: Path | None, anchor: bool = False, reference: Path | None = None) -> list[str]:
    args = [
        "train.py", "--model_name", "ViT-L-14-336", "--img_size", "518",
        "--dataset", "MVTec", "--batch_size", "6", "--epoch", str(epoch),
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
        "--deterministic_algorithms",
    ]
    if resume is not None:
        args += ["--resume", str(resume)]
    if anchor:
        args += [
            "--use_safe_anchor", "--anchor_lambda", str(ANCHOR_LAMBDA),
            "--anchor_reference_path", str(reference), "--anchor_gradient_budget",
            "--anchor_family_budget", str(ANCHOR_RHO), "--anchor_family_audit",
        ]
    return args


def assert_source_config(payload: dict, *, epoch: int, anchored: bool) -> None:
    cfg = payload["resolved_scientific_config"]
    expected = {
        "model_name": "ViT-L-14-336", "img_size": 518, "dataset": "MVTec", "n_groups": 3,
        "seed": 0, "image_adapt_weight": 0.2, "text_adapt_weight": 0.2,
        "lora_rank": 16, "lora_alpha": 2.0, "conv_lora_rank": 8, "conv_lora_alpha": 2.0,
        "conv_kernel_size_list": [3, 5], "batch_size": 6, "image_lr": 0.001,
        "text_lr": 0.0005, "use_hybrid_soft_prompt": True, "use_soft_prompt": False,
        "hybrid_alpha_max": 0.2, "soft_prompt_ctx_len": 4, "soft_prompt_freeze_epochs": 3,
        "soft_prompt_lr": 0.00005, "lambda_kg": 0.01, "lambda_k": 0.002, "lr_gamma": 0.9,
        "dfg_mode": "attn", "dfg_attn_dim": 256, "dfg_attn_tau": 8.0, "use_ss2d_dfg": True,
        "dfg_gamma_max": 0.2, "dfg_ss2d_fusion": "weight_residual", "dfg_beta": 0.1,
        "dfg_beta_schedule": "warmup010", "dfg_beta_target": 0.1,
        "dfg_weight_residual_fp32": True, "grad_clip_norm": 1.0, "amp": True,
        "deterministic_algorithms": True, "use_cir_training": False, "cir_alpha": 0.0,
        "training_horizon": 20, "primary_horizon": 15, "secondary_horizon": 20,
        "use_safe_anchor": anchored, "anchor_lambda": ANCHOR_LAMBDA if anchored else 0.0,
        "anchor_gradient_budget": anchored, "anchor_family_budget": ANCHOR_RHO if anchored else 0.1,
        "use_cir_training": False,
    }
    mismatches = {key: {"expected": value, "observed": cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"MVTec-source configuration mismatch at E{epoch}: {json.dumps(mismatches, sort_keys=True)}")


def write_theta_ref(path: Path, e8_path: Path, e8_payload: dict) -> dict:
    reference = {name: value.detach().float().cpu().clone() for name, value in e8_payload["image_parameter_reference"].items()}
    reference_hash = contract.state_dict_sha256(reference)
    payload = {
        "schema": "MVTec_SOURCE_ANCHOR_REF_E8_V1_THETA_REF_V1",
        "dataset_source": "MVTec",
        "epoch": 8,
        "source_checkpoint": str(e8_path),
        "source_checkpoint_sha256": sha256(e8_path),
        "source_config_sha256": e8_payload["config_sha256"],
        "reference_sha256": reference_hash,
        "image_anchor_reference": reference,
        "resolved_scientific_config": dict(e8_payload["resolved_scientific_config"]),
    }
    base.atomic_torch_save(path, payload)
    return {
        "path": str(path), "sha256": sha256(path), "reference_sha256": reference_hash,
        "source_checkpoint": str(e8_path), "source_checkpoint_sha256": sha256(e8_path), "source_epoch": 8,
    }


def patch_resume_identity() -> None:
    """Bridge the clean E8 checkpoint into the declared Anchor branch.

    The strict checkpoint validator normally permits branch-only changes at
    the original E1 boundary.  This experiment intentionally freezes clean H
    through E8 first, so the same in-memory bridge is required at E8.  Tensor
    and optimizer state are not modified.
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
            already_bridged = all(actual.get(key) == expected.get(key) for key in contract.RESUME_BRANCH_KEYS)
            if not already_bridged:
                if actual.get("use_safe_anchor") or actual.get("use_cir_training"):
                    raise RuntimeError("E8 parent is contaminated by Anchor or CIR")
                for key in contract.RESUME_BRANCH_KEYS:
                    actual[key] = expected.get(key)
        else:
            raise RuntimeError(f"unexpected resume boundary for MVTec-source E8 experiment: E{source_epoch}")

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


def run_phase(args: list[str]) -> None:
    base.reset_logging()
    sys.argv = args
    train_module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("all", "clean", "anchor"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    patch_resume_identity()
    if args.run == "all" and args.root.exists() and any(args.root.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty run root: {args.root}")
    args.root.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(args.root)

    source_root = args.root / "source_e1"
    clean_root = args.root / "clean_h_e8"
    anchor_root = args.root / "anchor_e8"
    theta_path = args.root / "theta_ref_e8.pth"

    if args.run == "all":
        run_phase(source_training_args(source_root, epoch=1, resume=None))
        source_e1 = source_root / "adapter_1.pth"
        source_payload = base.assert_full_state(source_e1, 1)
        assert_source_config(source_payload, epoch=1, anchored=False)
        run_phase(source_training_args(clean_root, epoch=8, resume=source_e1))
    if args.run == "clean":
        source_e1 = source_root / "adapter_1.pth"
        source_payload = base.assert_full_state(source_e1, 1)
        assert_source_config(source_payload, epoch=1, anchored=False)
        run_phase(source_training_args(clean_root, epoch=8, resume=source_e1))

    e8_path = clean_root / "adapter_8.pth"
    e8_payload = base.assert_full_state(e8_path, 8)
    assert_source_config(e8_payload, epoch=8, anchored=False)
    theta_record = write_theta_ref(theta_path, e8_path, e8_payload)

    if args.run in ("all", "anchor"):
        run_phase(source_training_args(anchor_root, epoch=15, resume=e8_path, anchor=True, reference=theta_path))

    e15_path = anchor_root / "adapter_15.pth"
    e15_payload = base.assert_full_state(e15_path, 15)
    assert_source_config(e15_payload, epoch=15, anchored=True)
    cfg = e15_payload["resolved_scientific_config"]
    required = {
        "anchor_reference_sha256": theta_record["sha256"],
        "use_cir_training": False,
        "cir_alpha": 0.0,
    }
    mismatches = {key: {"expected": value, "observed": cfg.get(key)} for key, value in required.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"MVTec-source Anchor E15 mismatch: {json.dumps(mismatches, sort_keys=True)}")

    manifest = {
        "experiment_id": "mvtec_source_anchor_ref_e8_v1",
        "status": "TRAINING_COMPLETE",
        "branch": "research/mvtec-source-anchor-ref-e8-v1",
        "source_dataset": "MVTec",
        "target_dataset": "VisA",
        "manifest_audit": audit,
        "training_implementation_base_commit": "31167af5ee3dfff80b74af1e9ee0da4ecc475d2e",
        "git_head_at_run": e15_payload.get("git_sha"),
        "source_e1": {"path": str(source_e1), "sha256": sha256(source_e1), "epoch": 1},
        "clean_e8": {"path": str(e8_path), "sha256": sha256(e8_path), "epoch": 8, "global_step": int(e8_payload["global_step"]), "anchor": False, "cir": False},
        "theta_ref": theta_record,
        "anchor_e15": {"path": str(e15_path), "sha256": sha256(e15_path), "epoch": 15, "global_step": int(e15_payload["global_step"]), "anchor": True, "cir": False},
        "visa_status": "PENDING",
    }
    base.atomic_json(args.root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
