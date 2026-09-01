#!/usr/bin/env python3
"""Run the current exact forward/metric path on a legacy Phase2B checkpoint.

This is a diagnostic bridge for a legacy checkpoint that predates the current
checkpoint identity fields. It does not rewrite the checkpoint or its
weights. It deliberately bypasses only the current identity gate, then uses
the current frozen Phase2B builder, forward path, deployment operator, and
disk-backed metric evaluator. The bypass is recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32
from scripts.cir_rmt.eval_full import _RssMonitor, _target_dataset
from tools.cir_rmt.corrective_eval import _evaluate_model, _metrics, _evaluator_sha, _git_sha
from tools.cir_rmt.identity import config_sha256


TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status", "method", "epoch", "target", "n_images",
        *METRICS, "checkpoint", "checkpoint_sha256", "config_sha256",
        "evaluator_git_sha", "evaluator_sha256", "identity_gate",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    clip_asset = args.clip_asset.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    parent_config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", args.epoch))
    configure_canonical_fp32()
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    started = time.time()
    for target in TARGETS:
        cell_started = time.time()
        cell_root = output / "temporary_spools" / f"medical__legacy_same_checkpoint__E{epoch:02d}__{target}"
        model = None
        dataset = None
        try:
            model = build_phase2b_frozen(parent_config, payload, clip_asset, device)
            dataset = _target_dataset(target, target_root)
            monitor = _RssMonitor()
            monitor.start()
            spools, seen, telemetry = _evaluate_model(
                model,
                dataset,
                mode="parent",
                config=parent_config,
                dataset_name=target,
                domain="Medical",
                device=device,
                spool_root=cell_root,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                prefetch_factor=int(args.prefetch_factor),
                monitor=monitor,
            )
            metrics = _metrics(spools["native"])
            for spool in spools.values():
                spool.close()
                spool.cleanup()
            row = {
                "status": "COMPLETE",
                "method": "HISTORICAL_CKPT_CURRENT_EXACT",
                "epoch": epoch,
                "target": target,
                "n_images": int(seen),
                **{metric: metrics.get(metric) for metric in METRICS},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": config_sha256(parent_config),
                "evaluator_git_sha": _git_sha(),
                "evaluator_sha256": _evaluator_sha(),
                "identity_gate": "BYPASSED_LEGACY_METADATA_ONLY",
                "elapsed_seconds": round(time.time() - cell_started, 3),
            }
            rows.append(row)
            _write_csv(output / "same_checkpoint_current_evaluator.csv", rows)
            print(f"COMPLETE {target} images={seen} elapsed={row['elapsed_seconds']}", flush=True)
            del spools, telemetry
        except Exception as error:
            row = {
                "status": "FAILED",
                "method": "HISTORICAL_CKPT_CURRENT_EXACT",
                "epoch": epoch,
                "target": target,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": config_sha256(parent_config),
                "evaluator_git_sha": _git_sha(),
                "evaluator_sha256": _evaluator_sha(),
                "identity_gate": "BYPASSED_LEGACY_METADATA_ONLY",
                "elapsed_seconds": round(time.time() - cell_started, 3),
            }
            rows.append(row)
            _write_csv(output / "same_checkpoint_current_evaluator.csv", rows)
            (output / "FAILED.json").write_text(
                json.dumps({"status": "FAILED", "target": target, "error": repr(error)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        finally:
            shutil.rmtree(cell_root, ignore_errors=True)
            del model, dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(output / "same_checkpoint_current_evaluator.csv", rows)
    (output / "identity.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "scope": "six_medical_targets",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "epoch": epoch,
                "target_root": str(target_root),
                "clip_asset": str(clip_asset),
                "clip_asset_sha256": sha256_file(clip_asset),
                "parent_config": str(config_path),
                "parent_config_sha256": config_sha256(parent_config),
                "targets": list(TARGETS),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "prefetch_factor": int(args.prefetch_factor),
                "identity_gate": "BYPASSED_LEGACY_METADATA_ONLY",
                "weights_modified": False,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETED", "targets": len(rows), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
