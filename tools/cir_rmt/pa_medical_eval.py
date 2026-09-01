#!/usr/bin/env python3
"""Run the PA-only Medical control matrix with exact-cell resume safety.

The existing P/CIR evaluator is reused unchanged.  This wrapper evaluates
only the new PA checkpoints (native Phase2B inference), so the frozen P,
C_OLD, and A rows are never recomputed and no inference-RMT condition is
introduced for PA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.cir_rmt.corrective_eval import _evaluate_cell
from tools.cir_rmt.identity import config_sha256, load_cir_config
from tools.cir_rmt.parameter_anchor import sha256_file


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
CONTROL_ID = "PA_PHASE2B_IMAGE_ANCHOR_V1"
ANCHOR_SHA = "3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _cell_id(epoch: int, target: str) -> str:
    return f"medical__PA__E{int(epoch):02d}__{_safe(target)}"


def _validate_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError(f"PA Medical freeze status is not FROZEN: {freeze.get('status')!r}")
    if freeze.get("target_tuning_occurred") is not False:
        raise RuntimeError("PA Medical freeze records target tuning")
    if [int(value) for value in freeze.get("candidate_epochs", [])] != list(EPOCHS):
        raise RuntimeError("PA Medical freeze candidate epochs mismatch")
    if freeze.get("medical_status") not in {"NOT_RUN", "FROZEN_AFTER_SOURCE"}:
        raise RuntimeError("PA Medical freeze was created after Medical access")
    return freeze


def _validate_checkpoint(path: Path, epoch: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != int(epoch):
        raise RuntimeError(f"PA checkpoint epoch mismatch: {path}")
    control = payload.get("control_identity", {})
    if control.get("control_id") != CONTROL_ID or control.get("cir_training") is not False or control.get("rmt_training") is not False:
        raise RuntimeError(f"PA control identity mismatch: {path}")
    if control.get("training_forward") != "native_phase2b" or control.get("inference_path") != "native_phase2b":
        raise RuntimeError(f"PA path identity mismatch: {path}")
    anchor = payload.get("image_anchor", {})
    if abs(float(anchor.get("lambda_image_anchor", 0.0)) - 0.001) > 1.0e-15 or anchor.get("reference_checkpoint_sha256") != ANCHOR_SHA:
        raise RuntimeError(f"PA anchor identity mismatch: {path}")
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is not False or payload.get("tf32_enabled") is not False:
        raise RuntimeError(f"PA precision identity mismatch: {path}")
    return sha256_file(path)


def _load_completed(output: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_csv(output / "PA_MEDICAL_LEDGER.csv"):
        if row.get("status") != "COMPLETE":
            continue
        cell_id = str(row.get("cell_id", ""))
        path = output / str(row.get("cell_path", ""))
        if not cell_id or not path.is_file() or sha256_file(path) != row.get("cell_sha256"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE" and payload.get("cell_id") == cell_id:
            completed[cell_id] = payload
    return completed


def _record(output: Path, rows: Sequence[Mapping[str, Any]], ledger: dict[str, dict[str, Any]]) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for row in rows:
        cell_id = _cell_id(int(row["epoch"]), str(row["target"]))
        payload = dict(row)
        payload.update({"cell_id": cell_id, "method": "PA", "status": "COMPLETE", "source": "new_PA_checkpoint"})
        path = output / "cells" / f"{cell_id}.json"
        _atomic_json(path, payload)
        digest = sha256_file(path)
        ledger[cell_id] = {
            "cell_id": cell_id,
            "scope": "medical",
            "method": "PA",
            "epoch": int(row["epoch"]),
            "target": str(row["target"]),
            "status": "COMPLETE",
            "cell_path": str(path.relative_to(output)),
            "cell_sha256": digest,
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "n_images": int(row.get("n_images", 0)),
            "updated_at": now,
        }
    _write_csv(output / "PA_MEDICAL_LEDGER.csv", list(ledger.values()), [
        "cell_id", "scope", "method", "epoch", "target", "status", "cell_path",
        "cell_sha256", "checkpoint_sha256", "n_images", "updated_at",
    ])


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.freeze.expanduser().resolve()
    freeze = _validate_freeze(freeze_path)
    if args.resource_admission is not None:
        admission = json.loads(args.resource_admission.expanduser().resolve().read_text(encoding="utf-8"))
        if admission.get("status") != "PASS":
            raise RuntimeError("PA Medical resource admission did not PASS")

    parent_config = json.loads(args.parent_config.expanduser().resolve().read_text(encoding="utf-8"))
    parent_sha = config_sha256(parent_config)
    # PA uses the native parent evaluator.  Keeping the parent config here is
    # intentional: no CIR config or alpha condition is involved in this run.
    completed = _load_completed(output) if args.resume else {}
    if not args.resume and (output / "PA_MEDICAL_LEDGER.csv").is_file():
        raise RuntimeError("existing PA Medical ledger requires --resume")
    identity = {
        "status": "RUNNING",
        "scope": "medical_PA_only",
        "method": "PA",
        "parent_config_sha256": parent_sha,
        "source_root": str(args.source_root.expanduser().resolve()),
        "medical_root": str(args.medical_root.expanduser().resolve()),
        "clip_asset": str(args.clip_asset.expanduser().resolve()),
        "clip_asset_sha256": sha256_file(args.clip_asset.expanduser().resolve()),
        "freeze": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "epochs": list(EPOCHS),
        "targets": list(TARGETS),
        "planned_cells": len(EPOCHS) * len(TARGETS),
        "completed_cells": len(completed),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "target_tuning_occurred": False,
        "mvtec": "NOT_RUN",
    }
    _atomic_json(output / "identity.json", identity)
    ledger = {row["cell_id"]: row for row in _read_csv(output / "PA_MEDICAL_LEDGER.csv")}
    config = parent_config
    device = torch.device(args.device)
    try:
        for target in TARGETS:
            for epoch in EPOCHS:
                cell_id = _cell_id(epoch, target)
                if cell_id in completed:
                    continue
                checkpoint = args.pa_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"adapter_{epoch}.pth"
                checkpoint_sha = _validate_checkpoint(checkpoint, epoch)
                rows = _evaluate_cell(
                    method_group="P",
                    epoch=epoch,
                    target=target,
                    scope="medical",
                    checkpoint_path=checkpoint,
                    config=config,
                    parent_config=parent_config,
                    clip_asset=args.clip_asset.expanduser().resolve(),
                    target_root=args.medical_root.expanduser().resolve(),
                    source_root=args.source_root.expanduser().resolve(),
                    output_root=output,
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    prefetch_factor=int(args.prefetch_factor),
                )
                if len(rows) != 1 or rows[0].get("method") != "P":
                    raise RuntimeError(f"PA native evaluator returned unexpected rows: {rows}")
                row = dict(rows[0])
                row.update({"method": "PA", "checkpoint_sha256": checkpoint_sha, "pa_control_id": CONTROL_ID, "freeze_sha256": sha256_file(freeze_path)})
                _record(output, [row], ledger)
                completed[cell_id] = {**row, "cell_id": cell_id, "method": "PA", "status": "COMPLETE"}
            _atomic_json(output / "PROGRESS.json", {
                "status": "RUNNING",
                "scope": "medical_PA_only",
                "completed_cells": len(completed),
                "planned_cells": len(EPOCHS) * len(TARGETS),
                "last_completed_domain": target,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            print(f"completed PA Medical domain block {target}: {len(completed)}/{len(EPOCHS) * len(TARGETS)} cells", flush=True)
        expected = {_cell_id(epoch, target) for epoch in EPOCHS for target in TARGETS}
        if set(completed) != expected:
            raise RuntimeError(f"PA Medical ledger incomplete: missing={sorted(expected - set(completed))[:5]}")
        identity.update({"status": "COMPLETED", "completed_cells": len(completed)})
        _atomic_json(output / "identity.json", identity)
        _atomic_json(output / "COMPLETE.json", {
            "status": "COMPLETED",
            "scope": "medical_PA_only",
            "completed_cells": len(completed),
            "planned_cells": len(expected),
            "target_tuning_occurred": False,
            "mvtec": "NOT_RUN",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    except Exception as error:
        _atomic_json(output / "FAILED.json", {
            "status": "FAILED",
            "scope": "medical_PA_only",
            "completed_cells": len(completed),
            "planned_cells": len(EPOCHS) * len(TARGETS),
            "error": repr(error),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--resource-admission", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pa-run-root", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, default=ROOT / "configs/phase2b_canonical_v1.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
