#!/usr/bin/env python3
"""
Evaluate S0 and S1 on actual P1-v8 structural smoke checkpoints (Epochs 1, 2, 3).

Modes:
  S0: hard_anchor global, rho = 0.0 (local disabled), experts disabled
  S1: hard_anchor global, dense CoPS local enabled, trained rho enabled, experts disabled

Outputs:
  - runs/phase4/p1_v8_evidence/smoke_s0_s1_by_epoch.csv
  - runs/phase4/p1_v8_evidence/smoke_s0_s1_by_dataset.csv
"""
import argparse
import os
import sys
import subprocess
import json
import shutil
import csv
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Validation-only S0/S1 triage for a completed P1-v8 smoke run."
)
parser.add_argument(
    "--save-path",
    default="runs/phase4/progress1_v8_structural_smoke_seed0",
    help="Completed run containing adapter_<epoch>.pth checkpoints.",
)
parser.add_argument(
    "--manifest-root",
    default=None,
    help="Validation manifest directory; defaults to <save-path>/protocol/medical_manifests.",
)
parser.add_argument(
    "--output-dir",
    default=None,
    help="Directory for triage artifacts; defaults to <save-path>/s0_s1_validation.",
)
parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"],
)
args = parser.parse_args()

SAVE_PATH = args.save_path
MANIFEST_ROOT = args.manifest_root or f"{SAVE_PATH}/protocol/medical_manifests"
OUTPUT_DIR = Path(args.output_dir or f"{SAVE_PATH}/s0_s1_validation")
DATASETS = args.datasets
EPOCHS = args.epochs
PYTHON_EXEC = sys.executable


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_metadata():
    manifest_hash = hashlib.sha256()
    counts = {}
    for dataset in DATASETS:
        path = Path(MANIFEST_ROOT) / f"{dataset}_val.jsonl"
        raw = path.read_bytes()
        manifest_hash.update(path.name.encode("utf-8") + b"\n" + raw)
        counts[dataset] = sum(1 for line in raw.splitlines() if line.strip())
    return manifest_hash.hexdigest(), counts

# 1. Prepare medical manifests if not exists
if not os.path.exists(MANIFEST_ROOT):
    print("Preparing medical manifests...")
    cmd = [
        PYTHON_EXEC, "tools/prepare_phase4_medical_splits.py",
        "--output-root", MANIFEST_ROOT,
        "--val-ratio", "0.30",
        "--seed", "0"
    ]
    subprocess.run(cmd, check=True)

# Function to run test.py for a dataset, epoch, and mode
def run_eval_mode(mode_name, epoch, extra_args):
    out_dir = OUTPUT_DIR / f"eval_{mode_name}_epoch_{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==========================================")
    print(f"Running {mode_name} for Epoch {epoch}")
    print(f"==========================================")

    expected_results = [
        Path(SAVE_PATH) / f"exact_results_{dataset}_val_epoch_{epoch}.csv"
        for dataset in DATASETS
    ]
    for dataset, result_path in zip(DATASETS, expected_results):
        print(f"--- Dataset: {dataset} (Epoch {epoch}, Mode {mode_name}) ---")
        if result_path.exists() and result_path.stat().st_size > 0:
            print(f"Reusing existing raw result: {result_path}")
            continue
        cmd = [
            PYTHON_EXEC, "test.py",
            "--dataset", dataset,
            "--img_size", "518",
            "--cuda_device", "0",
            "--save_path", SAVE_PATH,
            "--batch_size", "1",
            "--num_workers", "4",
            "--medical_split", "val",
            "--medical_manifest_root", MANIFEST_ROOT,
            "--external_exact_pixel_metrics",
            "--external_metric_chunk_pixels", "5000000",
            "--pixel_stride", "1",
            "--epochs", str(epoch),
            "--n_groups", "4",
            "--lora_rank", "16",
            "--lora_alpha", "2.0",
            "--conv_lora_rank", "8",
            "--conv_lora_alpha", "2.0",
            "--conv_kernel_size_list", "3", "5",
            "--dfg_mode", "mlp",
            "--h6_progress", "1",
            "--h6_progress_version", "P1-v8-minimal",
            "--h6_diagnostics_mode", "light",
            "--h6_num_factors", "4",
            "--h6_top_k", "2",
            "--h6_bank_dim", "256",
            "--h6_router_dim", "128",
            "--h6_factor_generator_specialization_enabled",
            "--h6_factor_head_init_scale", "0.001",
            "--h6_factor_local_dynamic_mix", "0.05",
            "--h6_factor_id_scale", "0.02",
            "--h6_factor_id_max_ratio", "0.05",
            "--h6_router_temperature", "1.0",
            "--h6_global_text_mode", "hard_anchor",
            "--no-h6_expert_enabled",
        ] + extra_args

        subprocess.run(cmd, check=True)

    # Re-aggregate support aware for this run
    agg_cmd = [
        PYTHON_EXEC, "tools/reaggregate_support_aware.py",
        "--save_path", SAVE_PATH,
        "--split", "val",
        "--epochs", str(epoch),
        "--output_dir", str(out_dir),
        "--manifest_root", MANIFEST_ROOT,
        "--allow-missing-original-selection",
    ]
    subprocess.run(agg_cmd, check=True)

    # Move raw results only after successful aggregation.  This makes a
    # failed aggregation resumable without rerunning completed inference.
    for f in Path(SAVE_PATH).glob(f"exact_results_*_val_epoch_{epoch}.csv"):
        shutil.move(str(f), str(out_dir / f.name))

    # Read the selection_support_aware_v2.json
    with open(out_dir / "selection_support_aware_v2.json") as f:
        data = json.load(f)
    if len(data["macro_by_epoch"][0]["per_dataset"]) != len(DATASETS):
        raise RuntimeError(f"{mode_name} epoch {epoch}: incomplete dataset aggregation")
    return data

# Evaluate S0 and S1 across epochs 1, 2, 3
results_s0 = {}
results_s1 = {}

for ep in EPOCHS:
    # S0: rho = 0.0 (local contribution disabled)
    s0_data = run_eval_mode("S0", ep, ["--h6_test_rho_override", "0.0"])
    results_s0[ep] = s0_data

    # S1: dense CoPS local enabled (trained rho)
    s1_data = run_eval_mode("S1", ep, ["--h6_prediction_routing", "dense"])
    results_s1[ep] = s1_data

# Save candidate-local validation-only triage artifacts.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
manifest_hash, sample_counts = manifest_metadata()
checkpoint_hashes = {
    epoch: sha256_file(Path(SAVE_PATH) / f"adapter_{epoch}.pth") for epoch in EPOCHS
}

# 1. smoke_s0_s1_by_epoch.csv
epoch_csv_path = OUTPUT_DIR / "s0_s1_metrics.csv"
with open(epoch_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "mode",
        "support_aware_image_macro", "support_aware_pixel_macro", "support_aware_combined_score",
        "legacy_combined_score",
        "pixel_ap_macro", "pixel_auroc_macro", "image_ap_macro", "image_auroc_macro",
        "sample_count", "manifest_sha256", "checkpoint_sha256"
    ])
    for ep in EPOCHS:
        for mode_name, res_dict in [("S0", results_s0[ep]), ("S1", results_s1[ep])]:
            macro = res_dict["macro_by_epoch"][0]
            per_ds = macro["per_dataset"]

            pix_ap_macro = sum(d["pixel_AP"] for d in per_ds) / len(per_ds)
            pix_auroc_macro = sum(d["pixel_AUROC"] for d in per_ds) / len(per_ds)

            valid_img_ds = [d for d in per_ds if d["image_valid"]]
            img_ap_macro = sum(d["image_AP"] for d in valid_img_ds) / len(valid_img_ds) if valid_img_ds else 0.0
            img_auroc_macro = sum(d["image_AUROC"] for d in valid_img_ds) / len(valid_img_ds) if valid_img_ds else 0.0

            writer.writerow([
                ep, mode_name,
                macro.get("support_aware_image_macro"),
                macro.get("support_aware_pixel_macro"),
                macro.get("support_aware_combined_score"),
                macro.get("legacy_combined_score"),
                pix_ap_macro, pix_auroc_macro, img_ap_macro, img_auroc_macro,
                sum(sample_counts.values()), manifest_hash, checkpoint_hashes[ep]
            ])

print(f"\nSaved {epoch_csv_path}")

# 2. smoke_s0_s1_by_dataset.csv
dataset_csv_path = OUTPUT_DIR / "s0_s1_by_dataset.csv"
with open(dataset_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "mode", "dataset",
        "pixel_AUROC", "pixel_AP", "pixel_score", "pixel_valid",
        "image_AUROC", "image_AP", "image_score", "image_valid",
        "sample_count", "manifest_sha256", "checkpoint_sha256"
    ])
    for ep in EPOCHS:
        for mode_name, res_dict in [("S0", results_s0[ep]), ("S1", results_s1[ep])]:
            macro = res_dict["macro_by_epoch"][0]
            for ds_info in macro["per_dataset"]:
                writer.writerow([
                    ep, mode_name, ds_info["dataset"],
                    ds_info["pixel_AUROC"], ds_info["pixel_AP"], ds_info["pixel_score"], ds_info["pixel_valid"],
                    ds_info["image_AUROC"], ds_info["image_AP"], ds_info["image_score"], ds_info["image_valid"],
                    sample_counts[ds_info["dataset"]], manifest_hash, checkpoint_hashes[ep]
                ])

print(f"Saved {dataset_csv_path}")

summary = {
    "protocol": {
        "global_text_mode": "hard_anchor",
        "S0": "rho=0; local CoPS contribution disabled; experts disabled",
        "S1": "dense CoPS local enabled; trained rho; experts disabled",
        "manifest_sha256": manifest_hash,
        "sample_counts": sample_counts,
    },
    "checkpoint_sha256": checkpoint_hashes,
    "by_epoch": {},
}
for ep in EPOCHS:
    s0_macro = results_s0[ep]["macro_by_epoch"][0]
    s1_macro = results_s1[ep]["macro_by_epoch"][0]
    s0_pixel_ap = sum(d["pixel_AP"] for d in s0_macro["per_dataset"]) / len(s0_macro["per_dataset"])
    s1_pixel_ap = sum(d["pixel_AP"] for d in s1_macro["per_dataset"]) / len(s1_macro["per_dataset"])
    summary["by_epoch"][str(ep)] = {
        "S0": s0_macro,
        "S1": s1_macro,
        "S1_minus_S0_pixel_AP": s1_pixel_ap - s0_pixel_ap,
    }
summary_path = OUTPUT_DIR / "s0_s1_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {summary_path}")
