#!/usr/bin/env python3
"""Serialize compact training telemetry and LR histories for the E20 anchor run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EPOCHS = tuple(range(1, 21))
CANDIDATES = (10, 12, 14, 16, 18, 20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_sha(path: Path) -> str:
    return _sha256(path) if path.is_file() else ""


def run(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    archive = args.output_root.expanduser().resolve()
    base = run_root / "visa" / "seed0"
    manifest_path = base / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = {int(row["epoch"]): dict(row) for row in manifest.get("history", [])}
    if sorted(history) != list(EPOCHS):
        raise ValueError(f"training history must cover E1-E20: {sorted(history)}")
    audit = json.loads((archive / "EXTENSION_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    checkpoint_rows = {int(row["epoch"]): row for row in audit.get("candidate_checkpoints", [])}
    telemetry: list[dict[str, Any]] = []
    for epoch in range(14, 21):
        row = history[epoch]
        checkpoint = checkpoint_rows.get(epoch, {})
        image_lr = float(row.get("image_lr", row.get("lr", 0.001 * (0.9**epoch))))
        text_lr = float(row.get("text_lr", image_lr * 0.5))
        prompt_lr = float(row.get("soft_prompt_lr", 0.0 if epoch <= 3 else 9.0e-5))
        telemetry.append({
            "epoch": epoch,
            "phase": "resume_cursor" if epoch == 14 else "extension",
            "elapsed_seconds": float(row.get("elapsed_seconds", "nan")),
            "images_per_sec": float(row.get("images_per_sec", "nan")),
            "batches": int(row.get("batches", 0)),
            "mean_loss": float(row.get("mean_loss", "nan")),
            "mean_cls": float(row.get("mean_cls", "nan")),
            "mean_seg": float(row.get("mean_seg", "nan")),
            "mean_image_anchor": float(row.get("mean_image_anchor", "nan")),
            "anchor_reference_distance": float(row.get("anchor_reference_distance", "nan")),
            "rmt_delta_abs_mean": float(row.get("rmt_delta_abs_mean", "nan")),
            "mad_p50": float(row.get("mad_p50", "nan")),
            "delta_p50": float(row.get("delta_p50", "nan")),
            "delta_saturation_percent": float(row.get("delta_saturation_percent", "nan")),
            "image_lr": image_lr,
            "text_lr": text_lr,
            "soft_prompt_lr": prompt_lr,
            "scheduler_last_epoch": epoch,
            "scheduler_step_count": epoch + 1,
            "beta": float(row.get("beta", "nan")),
            "peak_vram_bytes": int(row.get("peak_vram_bytes", 0)),
            "peak_reserved_vram_bytes": int(row.get("peak_reserved_vram_bytes", 0)),
            "host_rss_bytes": int(row.get("host_rss_bytes", 0)),
            "checkpoint_sha256": checkpoint.get("checkpoint_sha256", ""),
            "checkpoint_saved": bool(row.get("checkpoint_saved", False)),
            "gradient_probe": json.dumps(row.get("gradient_probe") or {}, sort_keys=True),
        })
    _write_csv(archive / "E14_TO_E20_TRAINING_TELEMETRY.csv", telemetry, [
        "epoch", "phase", "elapsed_seconds", "images_per_sec", "batches", "mean_loss", "mean_cls", "mean_seg", "mean_image_anchor", "anchor_reference_distance", "rmt_delta_abs_mean", "mad_p50", "delta_p50", "delta_saturation_percent", "image_lr", "text_lr", "soft_prompt_lr", "scheduler_last_epoch", "scheduler_step_count", "beta", "peak_vram_bytes", "peak_reserved_vram_bytes", "host_rss_bytes", "checkpoint_sha256", "checkpoint_saved", "gradient_probe"
    ])
    gradient_rows: list[dict[str, Any]] = []
    for epoch in (16, 18, 20):
        probe = history[epoch].get("gradient_probe")
        if not probe:
            raise ValueError(f"missing required gradient probe at E{epoch}")
        gradient_rows.append({"epoch": epoch, "base_grad_l2": float(probe["base_grad_l2"]), "anchor_grad_l2": float(probe["anchor_grad_l2"]), "anchor_to_base_ratio": float(probe["anchor_to_base_ratio"]), "probe_batch": 1, "source": "first_batch_bounded_probe"})
    _write_csv(archive / "ANCHOR_GRADIENT_TRAJECTORY.csv", gradient_rows, ["epoch", "base_grad_l2", "anchor_grad_l2", "anchor_to_base_ratio", "probe_batch", "source"])

    lr_rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        row = history[epoch]
        image = float(row.get("image_lr", row.get("lr", 0.001 * (0.9**epoch))))
        text = float(row.get("text_lr", image * 0.5))
        prompt = float(row.get("soft_prompt_lr", 0.0 if epoch <= 3 else 9e-5))
        lr_rows.append({"epoch": epoch, "start_image_lr": 0.001 * (0.9 ** (epoch - 1)), "post_image_lr": image, "start_text_lr": 0.0005 * (0.9 ** (epoch - 1)), "post_text_lr": text, "start_prompt_lr": 0.0 if epoch <= 3 else 1.0e-4, "post_prompt_lr": prompt, "soft_prompt_frozen": epoch <= 3, "scheduler_last_epoch": epoch, "scheduler_step_count": epoch + 1, "checkpoint_sha256": checkpoint_rows.get(epoch, {}).get("checkpoint_sha256", ""), "source": "checkpoint_or_frozen_history"})
    _write_csv(archive / "cir_lr_history.csv", lr_rows, ["epoch", "start_image_lr", "post_image_lr", "start_text_lr", "post_text_lr", "start_prompt_lr", "post_prompt_lr", "soft_prompt_frozen", "scheduler_last_epoch", "scheduler_step_count", "checkpoint_sha256", "source"])

    parent_lr_rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        checkpoint = args.parent_run_root.expanduser().resolve() / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
        parent_lr_rows.append({"epoch": epoch, "start_image_lr": 0.001 * (0.9 ** (epoch - 1)), "post_image_lr": 0.001 * (0.9**epoch), "start_text_lr": 0.0005 * (0.9 ** (epoch - 1)), "post_text_lr": 0.0005 * (0.9**epoch), "start_prompt_lr": 0.0 if epoch <= 3 else 1.0e-4, "post_prompt_lr": 0.0 if epoch <= 3 else 9.0e-5, "soft_prompt_frozen": epoch <= 3, "scheduler_last_epoch": epoch, "scheduler_step_count": epoch + 1, "checkpoint_sha256": _checkpoint_sha(checkpoint), "source": "matched_parent_checkpoint_or_canonical_schedule"})
    _write_csv(archive / "parent_lr_history.csv", parent_lr_rows, ["epoch", "start_image_lr", "post_image_lr", "start_text_lr", "post_text_lr", "start_prompt_lr", "post_prompt_lr", "soft_prompt_frozen", "scheduler_last_epoch", "scheduler_step_count", "checkpoint_sha256", "source"])
    (archive / "TRAINING_TELEMETRY_STATUS.json").write_text(json.dumps({"status": "PASS", "epochs": list(CANDIDATES), "telemetry_epochs": list(range(14, 21)), "gradient_probe_epochs": [16, 18, 20], "medical": "NOT_RUN", "mvtec": "NOT_RUN", "manifest_sha256": _sha256(manifest_path)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
