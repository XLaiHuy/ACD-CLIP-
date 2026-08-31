#!/usr/bin/env python3
"""Run only new anchored Medical cells after the target-blind freeze.

P and C_OLD are reused from the frozen compact Medical decomposition.  Each
Anchor checkpoint forward produces two logical cells, A0 and A05, and the
ledger is updated atomically after every completed checkpoint/target group.
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


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
METHODS = ("P", "C_OLD_0", "C_OLD_05", "A0", "A05")
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_target(target: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in target)


def _cell_id(method: str, epoch: int, target: str) -> str:
    return f"medical__{method}__E{int(epoch):02d}__{_safe_target(target)}"


def _load_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("target_tuning_occurred") is not False:
        raise RuntimeError("target-blind pre-Medical freeze is not valid")
    if [int(value) for value in freeze.get("candidate_epochs", [])] != list(EPOCHS):
        raise RuntimeError("freeze candidate epochs do not match E10/E12/E14/E16/E18/E20")
    return freeze


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _load_frozen_rows(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows = _read_csv(path)
    output: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        target = str(row["target"])
        if epoch not in EPOCHS or target not in TARGETS:
            continue
        base = {
            "status": "COMPLETE",
            "scope": "medical",
            "epoch": epoch,
            "target": target,
            "n_images": int(row["n_images"]),
            "target_count": int(row.get("target_count") or 0),
            "image_metric_support": int(row.get("image_metric_support") or 0),
            "config_sha256": row.get("cir_config_sha256", ""),
            "evaluator_git_sha": row.get("evaluator_git_sha", ""),
            "evaluator_sha256": row.get("evaluator_sha256", ""),
            "source": "frozen_corrected_medical_decomposition.csv",
        }
        parent_sha = row.get("parent_checkpoint_sha256", "")
        old_sha = row.get("cir_checkpoint_sha256", "")
        for method, prefix, checkpoint_sha, alpha in (("P", "parent", parent_sha, None), ("C_OLD_0", "c0", old_sha, None), ("C_OLD_05", "c05", old_sha, 0.5)):
            value = dict(base)
            value.update({"method": method, "alpha": alpha, "checkpoint_sha256": checkpoint_sha})
            for metric in METRICS:
                value[metric] = _float_or_none(row.get(f"{prefix}_{metric}"))
            output[(epoch, method, target)] = value
    expected = {(epoch, method, target) for epoch in EPOCHS for target in TARGETS for method in ("P", "C_OLD_0", "C_OLD_05")}
    if set(output) != expected:
        raise RuntimeError(f"frozen Medical rows incomplete: missing={sorted(expected - set(output))[:5]}")
    return output


def _load_completed(output: Path) -> dict[str, dict[str, Any]]:
    ledger = output / "TARGET_EVAL_LEDGER.csv"
    completed: dict[str, dict[str, Any]] = {}
    if not ledger.is_file():
        return completed
    for row in _read_csv(ledger):
        if row.get("status") != "COMPLETE":
            continue
        path = output / row["cell_path"]
        if not path.is_file() or _sha256(path) != row.get("cell_sha256"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE" and payload.get("cell_id") == row.get("cell_id"):
            completed[str(row["cell_id"])] = payload
    return completed


def _upsert_ledger(output: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    values = [dict(row) for _, row in sorted(rows.items())]
    _write_csv(output / "TARGET_EVAL_LEDGER.csv", values, ["cell_id", "scope", "method", "epoch", "target", "status", "cell_path", "cell_sha256", "checkpoint_sha256", "n_images", "updated_at"])


def _new_cell_rows(
    *,
    checkpoint: Path,
    epoch: int,
    target: str,
    output: Path,
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    clip_asset: Path,
    medical_root: Path,
    source_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> list[dict[str, Any]]:
    rows = _evaluate_cell(
        method_group="CIR",
        epoch=epoch,
        target=target,
        scope="medical",
        checkpoint_path=checkpoint,
        config=config,
        parent_config=parent_config,
        clip_asset=clip_asset,
        target_root=medical_root,
        source_root=source_root,
        output_root=output,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    mapped: list[dict[str, Any]] = []
    for row in rows:
        method = {"C0": "A0", "C05": "A05"}.get(str(row["method"]))
        if method is None:
            raise RuntimeError(f"unexpected exact evaluator method: {row.get('method')}")
        value = dict(row)
        value["method"] = method
        value["source"] = "new_anchor_checkpoint"
        value["checkpoint_sha256"] = _sha256(checkpoint)
        mapped.append(value)
    if {str(row["method"]) for row in mapped} != {"A0", "A05"}:
        raise RuntimeError(f"incomplete A0/A05 result at E{epoch} {target}")
    return mapped


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_freeze(args.freeze.expanduser().resolve())
    if args.resource_admission:
        admission = json.loads(args.resource_admission.expanduser().resolve().read_text(encoding="utf-8"))
        if admission.get("status") != "PASS":
            raise RuntimeError("Medical resource admission did not PASS")
    config = load_cir_config(args.config.expanduser().resolve())
    parent_config = json.loads((ROOT / str(config["parent_config_path"])).read_text(encoding="utf-8"))
    old_rows = _load_frozen_rows(args.frozen_medical.expanduser().resolve())
    completed = _load_completed(output) if args.resume else {}
    if not args.resume and (output / "TARGET_EVAL_LEDGER.csv").exists():
        raise RuntimeError("existing target ledger requires --resume")
    identity = {
        "status": "RUNNING",
        "scope": "medical_anchor_only",
        "config_sha256": config_sha256(config),
        "source_root": str(args.source_root.expanduser().resolve()),
        "medical_root": str(args.medical_root.expanduser().resolve()),
        "clip_asset": str(args.clip_asset.expanduser().resolve()),
        "clip_asset_sha256": _sha256(args.clip_asset.expanduser().resolve()),
        "anchor_run_root": str(args.anchor_run_root.expanduser().resolve()),
        "freeze_sha256": _sha256(args.freeze.expanduser().resolve()),
        "epochs": list(EPOCHS),
        "targets": list(TARGETS),
        "new_methods": ["A0", "A05"],
        "new_logical_cells": 72,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "target_tuning": False,
        "mvtec": "NOT_RUN",
    }
    _atomic_json(output / "identity.json", identity)
    ledger_rows: dict[str, dict[str, Any]] = {}
    if (output / "TARGET_EVAL_LEDGER.csv").is_file():
        ledger_rows = {str(row["cell_id"]): row for row in _read_csv(output / "TARGET_EVAL_LEDGER.csv")}
    device = torch.device(args.device)
    try:
        for target in TARGETS:
            for epoch in EPOCHS:
                checkpoint = args.anchor_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                needed = {_cell_id(method, epoch, target) for method in ("A0", "A05")}
                if needed.issubset(completed):
                    continue
                rows = _new_cell_rows(checkpoint=checkpoint, epoch=epoch, target=target, output=output, config=config, parent_config=parent_config, clip_asset=args.clip_asset.expanduser().resolve(), medical_root=args.medical_root.expanduser().resolve(), source_root=args.source_root.expanduser().resolve(), device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers), prefetch_factor=int(args.prefetch_factor))
                for row in rows:
                    cell_id = _cell_id(str(row["method"]), epoch, target)
                    payload = dict(row)
                    payload["cell_id"] = cell_id
                    path = output / "cells" / f"{cell_id}.json"
                    _atomic_json(path, payload)
                    digest = _sha256(path)
                    ledger_rows[cell_id] = {"cell_id": cell_id, "scope": "medical", "method": row["method"], "epoch": epoch, "target": target, "status": "COMPLETE", "cell_path": str(path.relative_to(output)), "cell_sha256": digest, "checkpoint_sha256": row.get("checkpoint_sha256", ""), "n_images": row.get("n_images", ""), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    completed[cell_id] = payload
                _upsert_ledger(output, ledger_rows)
            _atomic_json(output / "PROGRESS.json", {"status": "RUNNING", "scope": "medical_anchor_only", "completed_logical_cells": len(completed), "planned_logical_cells": 72, "last_completed_domain": target, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"completed Medical Anchor domain block {target}: {len(completed)}/72 logical cells", flush=True)
        expected = {_cell_id(method, epoch, target) for epoch in EPOCHS for target in TARGETS for method in ("A0", "A05")}
        if set(completed) != expected:
            raise RuntimeError(f"target ledger incomplete: missing={sorted(expected - set(completed))[:5]}")
        identity["status"] = "COMPLETED"
        identity["completed_logical_cells"] = len(completed)
        _atomic_json(output / "identity.json", identity)
        _atomic_json(output / "COMPLETE.json", {"status": "COMPLETED", "scope": "medical_anchor_only", "completed_logical_cells": len(completed), "planned_logical_cells": 72, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    except Exception as error:
        _atomic_json(output / "FAILED.json", {"status": "FAILED", "scope": "medical_anchor_only", "error": repr(error), "completed_logical_cells": len(completed), "planned_logical_cells": 72, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--frozen-medical", type=Path, required=True)
    parser.add_argument("--resource-admission", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--old-cir-run-root", type=Path, required=True)
    parser.add_argument("--anchor-run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
