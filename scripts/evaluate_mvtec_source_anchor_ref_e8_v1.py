#!/usr/bin/env python3
"""Evaluate the MVTec-source Anchor trajectory on VisA.

The source experiment is trained with the original MVTec manifest and is
evaluated on VisA only. Every checkpoint is provenance-checked before the
frozen benchmark-exact, pixel-stride-1 evaluator is invoked.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_anchor_ref_e8_r1 as base


DEFAULT_RUN_ROOT = Path("/workspace/mvtec_source_anchor_ref_e8_v1_run")
EPOCHS = tuple(range(10, 16))
VISA_CLASSES = (
    "candle", "pcb3", "capsules", "pipe_fryum", "pcb4", "macaroni2",
    "pcb2", "chewinggum", "macaroni1", "cashew", "fryum", "pcb1",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def checkpoint_provenance(run_root: Path, epoch: int) -> dict:
    provenance = base.checkpoint_provenance(run_root, epoch)
    checkpoint = Path(provenance["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["resolved_scientific_config"]
    if cfg.get("dataset") != "MVTec":
        raise RuntimeError(f"source dataset mismatch at E{epoch}: {cfg.get('dataset')!r}")
    provenance["source_dataset"] = "MVTec"
    provenance["target_dataset"] = "VisA"
    provenance["target_manifest_sha256"] = base.sha256(REPO / "dataset/hub/VisA.jsonl")
    return provenance


def visa_command(provenance: dict, output: Path, python_bin: str) -> list[str]:
    return [
        python_bin, str(REPO / "test.py"), "--dataset", "VisA",
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


def parse_visa_log(path: Path, expected_epoch: int) -> tuple[dict, list[dict]]:
    current_epoch = None
    macro = None
    per_class = []
    class_pattern = __import__("re").compile(
        r"\b(" + "|".join(__import__("re").escape(name) for name in VISA_CLASSES) + r")\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)"
    )
    number_pattern = __import__("re").compile(r"[-+]?\d+(?:\.\d+)?")
    for line in path.read_text(encoding="utf-8").splitlines():
        epoch_match = __import__("re").search(r"load model from epoch (\d+)", line)
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
        raise RuntimeError(f"VisA Average row missing for E{expected_epoch}: {path}")
    if {row["class"] for row in per_class} != set(VISA_CLASSES):
        raise RuntimeError(f"VisA per-class rows incomplete for E{expected_epoch}: {path}")
    return macro, sorted(per_class, key=lambda row: VISA_CLASSES.index(row["class"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(run_root: Path, python_bin: str) -> None:
    training_manifest = run_root / "manifest_audit.json"
    if not training_manifest.is_file():
        raise RuntimeError("MVTec-source training manifest audit is missing")
    audit = json.loads(training_manifest.read_text(encoding="utf-8"))
    if audit.get("audit_status") != "PASS":
        raise RuntimeError(f"MVTec-source manifest audit did not pass: {audit.get('audit_status')}")
    visa_root = run_root / "visa_eval"
    visa_root.mkdir(parents=True, exist_ok=True)
    macro_rows, class_rows = [], []
    for epoch in EPOCHS:
        provenance = checkpoint_provenance(run_root, epoch)
        output = visa_root / f"E{epoch}"
        if (output / "COMPLETE").exists():
            macro, per_class = parse_visa_log(output / "test.log", epoch)
        else:
            if output.exists():
                raise RuntimeError(f"refusing to overwrite incomplete VisA output: {output}")
            output.mkdir(parents=True)
            (output / f"adapter_{epoch}.pth").symlink_to(Path(provenance["checkpoint"]))
            command = visa_command(provenance, output, python_bin)
            write_json(output / "provenance.json", {
                "domain": "VisA target",
                "protocol": "matched_industrial_transfer_benchmark_exact",
                "pixel_stride": 1,
                "checkpoint_provenance": provenance,
                "command": command,
            })
            with (output / "evaluation.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            macro, per_class = parse_visa_log(output / "test.log", epoch)
            (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        macro_rows.append(macro)
        class_rows.extend(per_class)
    write_csv(visa_root / "trajectory.csv", macro_rows)
    write_csv(visa_root / "per_class_pixel.csv", [
        {key: row[key] for key in ("epoch", "class", "pixel_auroc", "pixel_ap")}
        for row in class_rows
    ])
    write_csv(visa_root / "per_class_all_metrics.csv", class_rows)
    (visa_root / "COMPLETE").write_text("VisA E10-E15 complete.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    args = parser.parse_args()
    run(args.run_root, args.python_bin)


if __name__ == "__main__":
    main()
