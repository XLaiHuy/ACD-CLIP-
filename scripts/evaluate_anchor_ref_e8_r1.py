#!/usr/bin/env python3
"""Evaluate the E8-reference Anchor E10-E15 trajectory under frozen protocols.

Medical evaluation is performed with the pinned phase2cd evaluator. MVTec is
guarded behind completion of all Medical epochs and uses the repository's
benchmark_exact evaluator at pixel stride 1. No checkpoint or metric selection
is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("/workspace/anchor_ref_e8_r1_run")
DEFAULT_MEDICAL_REPO = Path("/workspace/ACD-CLIP-medical-test")
MEDICAL_EVALUATOR_COMMIT = "6bd932fbce0a425af5c8d3f7230dd7dc041568bd"
EPOCHS = tuple(range(10, 16))
MVTEC_CLASSES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "transistor",
    "toothbrush", "wood", "zipper",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def checkpoint_provenance(run_root: Path, epoch: int) -> dict:
    checkpoint = run_root / "anchor_e8" / f"adapter_{epoch}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256(checkpoint)
    with checkpoint.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Unresolved Git-LFS pointer: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_version", "protocol_version", "model_state", "image_adapter",
        "text_adapter", "soft_prompt", "optimizer_state", "scheduler_state",
        "scaler_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
        "dataloader_generator_state", "epoch", "global_step",
        "resolved_scientific_config", "config_sha256", "git_sha",
        "implementation_git_sha", "base_h2_commit", "clip_sha256",
        "dataset_manifest_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"incomplete checkpoint E{epoch}: {missing}")
    if int(payload["epoch"]) != epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {payload['epoch']} != {epoch}")
    cfg = payload["resolved_scientific_config"]
    expected = {
        "use_safe_anchor": True,
        "use_cir_training": False,
        "anchor_gradient_budget": True,
        "anchor_reference_sha256": sha256(run_root / "theta_ref_e8.pth"),
    }
    mismatches = {key: {"expected": value, "observed": cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Anchor provenance mismatch at E{epoch}: {json.dumps(mismatches, sort_keys=True)}")
    return {
        "epoch": epoch,
        "checkpoint": str(checkpoint.resolve()),
        "sha256": actual_sha,
        "global_step": int(payload["global_step"]),
        "checkpoint_version": payload["checkpoint_version"],
        "protocol_version": payload["protocol_version"],
        "config_sha256": payload["config_sha256"],
        "git_sha": payload["git_sha"],
        "implementation_git_sha": payload["implementation_git_sha"],
        "base_h2_commit": payload["base_h2_commit"],
        "clip_sha256": payload["clip_sha256"],
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "anchor_reference_sha256": cfg["anchor_reference_sha256"],
        "use_safe_anchor": cfg["use_safe_anchor"],
        "use_cir_training": cfg["use_cir_training"],
    }


def medical_command(medical_repo: Path, provenance: dict, output: Path, python_bin: str) -> list[str]:
    return [
        python_bin, str(medical_repo / "phase2cd_medical_eval.py"),
        "--state", f"ANCHOR_REF_E8_R1_E{provenance['epoch']}",
        "--checkpoint", provenance["checkpoint"],
        "--expected-sha256", provenance["sha256"],
        "--output-dir", str(output),
        "--batch-size", "8", "--num-workers", "6", "--cuda-device", "0",
        "--pixel-stride", "1",
    ]


def run_medical(run_root: Path, medical_repo: Path, python_bin: str) -> None:
    commit = subprocess.check_output(["git", "-C", str(medical_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != MEDICAL_EVALUATOR_COMMIT:
        raise RuntimeError(f"Medical evaluator commit mismatch: {commit} != {MEDICAL_EVALUATOR_COMMIT}")
    medical_root = run_root / "medical_eval"
    medical_root.mkdir(parents=True, exist_ok=True)
    for epoch in EPOCHS:
        provenance = checkpoint_provenance(run_root, epoch)
        output = medical_root / f"E{epoch}"
        if (output / "complete").exists():
            continue
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite incomplete Medical output: {output}")
        command = medical_command(medical_repo, provenance, output, python_bin)
        write_json(medical_root / f"E{epoch}.planned_provenance.json", {
            "domain": "Medical",
            "protocol": "phase2cd_frozen_raw_exact",
            "pixel_stride": 1,
            "checkpoint_provenance": provenance,
            "evaluator_repo": str(medical_repo.resolve()),
            "evaluator_commit": commit,
            "command": command,
        })
        with (medical_root / f"E{epoch}.evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=medical_repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(output / "provenance.json", {
            "domain": "Medical",
            "protocol": "phase2cd_frozen_raw_exact",
            "pixel_stride": 1,
            "checkpoint_provenance": provenance,
            "evaluator_repo": str(medical_repo.resolve()),
            "evaluator_commit": commit,
            "command": command,
        })
    rows = []
    pixel_rows = []
    for epoch in EPOCHS:
        output = medical_root / f"E{epoch}"
        if not (output / "complete").exists():
            raise RuntimeError(f"Medical output is not complete: {output}")
        with (output / "macro_metrics.csv").open(newline="", encoding="utf-8") as handle:
            macro = next(csv.DictReader(handle))
        rows.append({
            "epoch": epoch,
            "pixel_auroc": float(macro["pixel_auc_6"]),
            "pixel_ap": float(macro["pixel_ap_6"]),
            "image_auroc": float(macro["image_auc_3"]),
            "image_ap": float(macro["image_ap_3"]),
            "image_scope": "valid Brain/Liver/Retina only",
        })
        with (output / "dataset_metrics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["metric_type"] == "pixel":
                    pixel_rows.append({
                        "epoch": epoch,
                        "dataset": row["dataset"],
                        "pixel_auroc": float(row["pixel_auc"]),
                        "pixel_ap": float(row["pixel_ap"]),
                    })
    write_csv(medical_root / "trajectory.csv", rows)
    write_csv(medical_root / "per_dataset_pixel.csv", pixel_rows)
    (medical_root / "COMPLETE").write_text("Medical E10-E15 complete; frozen before MVTec.\n", encoding="utf-8")


def mvtec_command(provenance: dict, output: Path, python_bin: str) -> list[str]:
    return [
        python_bin, str(REPO / "test.py"), "--dataset", "MVTec",
        "--model_name", "ViT-L-14-336", "--img_size", "518", "--n_groups", "3",
        "--lora_rank", "16", "--lora_alpha", "2.0", "--conv_lora_rank", "8",
        "--conv_lora_alpha", "2.0", "--conv_kernel_size_list", "3", "5",
        "--dfg_mode", "attn", "--dfg_attn_dim", "256", "--dfg_attn_tau", "8.0",
        "--use_ss2d_dfg", "--dfg_gamma_max", "0.2", "--dfg_ss2d_fusion", "weight_residual",
        "--dfg_beta", "0.10", "--dfg_beta_schedule", "warmup010", "--dfg_beta_target", "0.10",
        "--batch_size", "8", "--cuda_device", "0", "--num_workers", "6",
        "--save_path", str(output), "--epochs", str(provenance["epoch"]),
        "--evaluator_mode", "benchmark_exact", "--pixel_stride", "1",
    ]


def parse_mvtec_log(path: Path, expected_epoch: int) -> tuple[dict, list[dict]]:
    current_epoch = None
    macro = None
    per_class = []
    class_pattern = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in MVTEC_CLASSES) + r")\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)"
    )
    number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for line in path.read_text(encoding="utf-8").splitlines():
        epoch_match = re.search(r"load model from epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        if current_epoch != expected_epoch:
            continue
        if "Average" in line:
            numbers = number_pattern.findall(line)
            if len(numbers) >= 4:
                macro = {
                    "epoch": expected_epoch,
                    "pixel_auroc": float(numbers[-4]),
                    "pixel_ap": float(numbers[-3]),
                    "image_auroc": float(numbers[-2]),
                    "image_ap": float(numbers[-1]),
                }
        match = class_pattern.search(line)
        if match and not any(row["class"] == match.group(1) for row in per_class):
            per_class.append({
                "epoch": expected_epoch,
                "class": match.group(1),
                "pixel_auroc": float(match.group(2)),
                "pixel_ap": float(match.group(3)),
                "image_auroc": float(match.group(4)),
                "image_ap": float(match.group(5)),
            })
    if macro is None:
        raise RuntimeError(f"MVTec Average row missing for E{expected_epoch}: {path}")
    if {row["class"] for row in per_class} != set(MVTEC_CLASSES):
        raise RuntimeError(f"MVTec per-class rows incomplete for E{expected_epoch}: {path}")
    return macro, sorted(per_class, key=lambda row: MVTEC_CLASSES.index(row["class"]))


def run_mvtec(run_root: Path, python_bin: str) -> None:
    medical_root = run_root / "medical_eval"
    if not (medical_root / "COMPLETE").exists():
        raise RuntimeError("MVTec is guarded: complete Medical E10-E15 evaluation is required first")
    mvtec_root = run_root / "mvtec_eval"
    mvtec_root.mkdir(parents=True, exist_ok=True)
    macro_rows, class_rows = [], []
    for epoch in EPOCHS:
        provenance = checkpoint_provenance(run_root, epoch)
        output = mvtec_root / f"E{epoch}"
        if (output / "COMPLETE").exists():
            macro, per_class = parse_mvtec_log(output / "test.log", epoch)
        else:
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite incomplete MVTec output: {output}")
            output.mkdir(parents=True)
            checkpoint_link = output / f"adapter_{epoch}.pth"
            checkpoint_link.symlink_to(Path(provenance["checkpoint"]))
            command = mvtec_command(provenance, output, python_bin)
            write_json(output / "provenance.json", {
                "domain": "MVTec AD",
                "protocol": "matched_industrial_transfer_benchmark_exact",
                "pixel_stride": 1,
                "metric_thresholds": None,
                "checkpoint_provenance": provenance,
                "command": command,
            })
            with (output / "evaluation.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            macro, per_class = parse_mvtec_log(output / "test.log", epoch)
            (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        macro_rows.append(macro)
        class_rows.extend(per_class)
    write_csv(mvtec_root / "trajectory.csv", macro_rows)
    write_csv(mvtec_root / "per_class_pixel.csv", [
        {key: row[key] for key in ("epoch", "class", "pixel_auroc", "pixel_ap")}
        for row in class_rows
    ])
    write_csv(mvtec_root / "per_class_all_metrics.csv", class_rows)
    (mvtec_root / "COMPLETE").write_text("MVTec E10-E15 complete.\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("medical", "mvtec", "all"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--medical-repo", type=Path, default=DEFAULT_MEDICAL_REPO)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    args = parser.parse_args()
    if args.phase in ("medical", "all"):
        run_medical(args.run_root, args.medical_repo, args.python_bin)
    if args.phase in ("mvtec", "all"):
        run_mvtec(args.run_root, args.python_bin)


if __name__ == "__main__":
    main()
