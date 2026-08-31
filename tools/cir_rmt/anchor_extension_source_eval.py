#!/usr/bin/env python3
"""Evaluate only new anchored source checkpoints and assemble the final matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.cir_rmt.identity import config_sha256, load_cir_config
from tools.cir_rmt.matched_horizon_source_eval import _evaluate_checkpoint, _frozen_sample, _parent_config
from scripts.cir_rmt.eval_full import ManifestDataset
from tools.cir_rmt.pre_full_run_diagnostics import IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
NEW_EPOCHS = (16, 18, 20)
METHODS = ("P", "C_OLD_0", "C_OLD_05", "A0", "A05")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_paths(args: argparse.Namespace, epoch: int, method: str) -> Path:
    if method == "P":
        return args.parent_run_root.expanduser().resolve() / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
    if method.startswith("C_OLD"):
        return args.old_cir_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
    return args.anchor_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"


def _map_method(method: str) -> str:
    return {"P0": "P", "P05": "P", "C0": "C_OLD_0", "C05": "C_OLD_05"}.get(method, method)


def _baseline_metrics(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _rows(path):
        epoch = int(row["epoch"])
        if epoch not in EPOCHS or row["method"] not in {"P0", "P05", "C0", "C05"}:
            continue
        method = _map_method(row["method"])
        if method == "P" and row["method"] != "P0":
            continue
        output[(epoch, method)] = {
            "epoch": epoch,
            "method": method,
            "n_images": int(row["n_images"]),
            "pixel_auroc": float(row["pixel_auroc"]),
            "pixel_ap": float(row["pixel_ap"]),
            "image_auroc": float(row["image_auroc"]),
            "image_ap": float(row["image_ap"]),
            "recomputed": False,
            "source": "frozen_SOURCE_BOUNDED_METRICS.csv",
        }
    expected = {(epoch, method) for epoch in EPOCHS for method in ("P", "C_OLD_0", "C_OLD_05")}
    if set(output) != expected:
        raise ValueError(f"frozen source baseline incomplete: missing={sorted(expected - set(output))}")
    return output


def _old_anchor_metrics(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _rows(path):
        if row.get("recomputed", "").lower() != "true" or int(row["epoch"]) not in {10, 12, 14}:
            continue
        method = {"C0": "A0", "C05": "A05"}.get(row["method"])
        if method is None:
            continue
        output[(int(row["epoch"]), method)] = {
            "epoch": int(row["epoch"]),
            "method": method,
            "n_images": int(row["n_images"]),
            "pixel_auroc": float(row["pixel_auroc"]),
            "pixel_ap": float(row["pixel_ap"]),
            "image_auroc": float(row["image_auroc"]),
            "image_ap": float(row["image_ap"]),
            "recomputed": True,
            "source": "matched_horizon_anchor_e14_source_eval.csv",
            "checkpoint_sha256": row.get("checkpoint_sha256", ""),
            "evaluation_seconds": row.get("evaluation_seconds", ""),
        }
    expected = {(epoch, method) for epoch in (10, 12, 14) for method in ("A0", "A05")}
    if set(output) != expected:
        raise ValueError(f"existing anchor source rows incomplete: missing={sorted(expected - set(output))}")
    return output


def _effect_rows(metrics: Mapping[tuple[int, str], Mapping[str, Any]], config_sha: str, evaluator_git_sha: str, evaluator_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        values: dict[str, Any] = {"target": "VisA_SOURCE_BOUNDED_96", "epoch": epoch, "n_images": int(metrics[(epoch, "P")]["n_images"]), "config_sha256": config_sha, "evaluator_git_sha": evaluator_git_sha, "evaluator_sha256": evaluator_sha}
        for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
            p = float(metrics[(epoch, "P")][metric])
            cold = float(metrics[(epoch, "C_OLD_0")][metric])
            cold5 = float(metrics[(epoch, "C_OLD_05")][metric])
            a = float(metrics[(epoch, "A0")][metric])
            a5 = float(metrics[(epoch, "A05")][metric])
            values.update({
                f"p_{metric}": p,
                f"c_old_0_{metric}": cold,
                f"c_old_05_{metric}": cold5,
                f"a0_{metric}": a,
                f"a05_{metric}": a5,
                f"anchor_train_effect_{metric}": a - p,
                f"anchor_rmt_inference_effect_{metric}": a5 - a,
                f"anchor_total_effect_{metric}": a5 - p,
                f"anchor_vs_old_cir_{metric}": a - cold,
                f"old_cir_rmt_inference_effect_{metric}": cold5 - cold,
            })
        rows.append(values)
    return rows


def _rename_rows(rows: Sequence[Mapping[str, Any]], *, method_map: Mapping[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        old = str(row.get("method", ""))
        if old not in method_map:
            continue
        value = dict(row)
        value["method"] = method_map[old]
        output.append(value)
    return output


def _merge_diagnostics(args: argparse.Namespace, new_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    current = args.anchor_archive_root.expanduser().resolve()
    old = args.sample_archive.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    mapping_old = {"P0": "P", "C0": "C_OLD_0", "C05": "C_OLD_05"}
    mapping_anchor = {"C0": "A0", "C05": "A05"}
    tails: list[dict[str, Any]] = []
    if (old / "AP_TAIL_DECOMPOSITION.csv").is_file():
        tails.extend(_rename_rows(_rows(old / "AP_TAIL_DECOMPOSITION.csv"), method_map=mapping_old))
    if (current / "MATCHED_HORIZON_AP_TAIL.csv").is_file():
        tails.extend(_rename_rows(_rows(current / "MATCHED_HORIZON_AP_TAIL.csv"), method_map=mapping_anchor))
    tails.extend(_rename_rows(new_rows["tails"], method_map=mapping_anchor))
    _write_csv(output / "FINAL_SOURCE_AP_TAIL.csv", tails, ["epoch", "method", "cohort", "stat", "value", "n", "checkpoint_sha256", "source"])

    deployment: list[dict[str, Any]] = []
    if (old / "DEPLOYMENT_CAUSAL_DIAGNOSTIC.csv").is_file():
        deployment.extend(_rename_rows(_rows(old / "DEPLOYMENT_CAUSAL_DIAGNOSTIC.csv"), method_map={"P0": "P", "C0": "C_OLD_0"}))
    if (current / "MATCHED_HORIZON_DEPLOYMENT.csv").is_file():
        deployment.extend(_rename_rows(_rows(current / "MATCHED_HORIZON_DEPLOYMENT.csv"), method_map={"C0": "A0"}))
    deployment.extend(_rename_rows(new_rows["deployment"], method_map={"C0": "A0"}))
    _write_csv(output / "FINAL_SOURCE_RAW_DEPLOYED.csv", deployment, ["epoch", "method", "metric", "value", "checkpoint_sha256", "source"])

    branches: list[dict[str, Any]] = []
    if (old / "IMAGE_BRANCH_DECOMPOSITION.csv").is_file():
        branches.extend(_rename_rows(_rows(old / "IMAGE_BRANCH_DECOMPOSITION.csv"), method_map={"P0": "P", "C0": "C_OLD_0"}))
    if (current / "MATCHED_HORIZON_BRANCH.csv").is_file():
        branches.extend(_rename_rows(_rows(current / "MATCHED_HORIZON_BRANCH.csv"), method_map={"C0": "A0"}))
    branches.extend(_rename_rows(new_rows["branches"], method_map={"C0": "A0"}))
    _write_csv(output / "FINAL_SOURCE_IMAGE_BRANCH.csv", branches, ["epoch", "method", "branch", "image_auroc", "image_ap", "mean_score", "n_images", "checkpoint_sha256", "source"])

    heldout: list[dict[str, Any]] = []
    if (old / "SOURCE_HELDOUT_RESULTS.csv").is_file():
        heldout.extend(_rename_rows(_rows(old / "SOURCE_HELDOUT_RESULTS.csv"), method_map={"P0": "P", "C0": "C_OLD_0", "C05": "C_OLD_05"}))
    if (current / "MATCHED_HORIZON_HELDOUT.csv").is_file():
        heldout.extend(_rename_rows(_rows(current / "MATCHED_HORIZON_HELDOUT.csv"), method_map={"C0": "A0", "C05": "A05"}))
    heldout.extend(_rename_rows(new_rows["heldout"], method_map={"C0": "A0", "C05": "A05"}))
    _write_csv(output / "FINAL_SOURCE_HELDOUT.csv", heldout, ["epoch", "method", "split", "category", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "checkpoint_sha256", "source"])


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_cir_config(args.config.expanduser().resolve())
    parent_config = _parent_config(config)
    indices, _, holdout = _frozen_sample(args.sample_archive.expanduser().resolve())
    dataset = ManifestDataset(args.source_root.expanduser().resolve(), ROOT / "dataset/hub/VisA.jsonl", IMAGE_SIZE)
    baseline = _baseline_metrics(args.sample_archive.expanduser().resolve() / "SOURCE_BOUNDED_METRICS.csv")
    existing_anchor = _old_anchor_metrics(args.anchor_archive_root.expanduser().resolve() / "MATCHED_HORIZON_SOURCE_RESULTS.csv")
    metrics: dict[tuple[int, str], dict[str, Any]] = {**baseline, **existing_anchor}
    new_diagnostics: dict[str, list[Mapping[str, Any]]] = {"tails": [], "deployment": [], "branches": [], "heldout": []}
    for epoch in NEW_EPOCHS:
        checkpoint = _checkpoint_paths(args, epoch, "A0")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = _evaluate_checkpoint(
            epoch=epoch,
            checkpoint=checkpoint,
            parent_config=parent_config,
            cir_config=config,
            dataset=dataset,
            indices=indices,
            clip_asset=args.clip_asset.expanduser().resolve(),
            device=__import__("torch").device(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            holdout=holdout,
        )
        epoch_metrics, tails, deployment, branches, heldout, _, timing = result
        for row in epoch_metrics:
            method = {"C0": "A0", "C05": "A05"}[str(row["method"])]
            metrics[(epoch, method)] = {**dict(row), "method": method, "recomputed": True, "source": "new_anchor_checkpoint", "checkpoint_sha256": row.get("checkpoint_sha256", ""), "evaluation_seconds": timing.get("total_evaluation_seconds", "")}
        new_diagnostics["tails"].extend(tails)
        new_diagnostics["deployment"].extend(deployment)
        new_diagnostics["branches"].extend(branches)
        new_diagnostics["heldout"].extend(heldout)
        print(f"completed new anchored source evaluation E{epoch:02d} ({len(indices)} images)", flush=True)
    expected = {(epoch, method) for epoch in EPOCHS for method in METHODS}
    if set(metrics) != expected:
        raise ValueError(f"final source matrix incomplete: missing={sorted(expected - set(metrics))}")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evaluator_sha = _sha256(ROOT / "scripts/cir_rmt/eval_full.py")
    compact_rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for method in METHODS:
            row = dict(metrics[(epoch, method)])
            checkpoint = _checkpoint_paths(args, epoch, method)
            row.update({"epoch": epoch, "method": method, "n_images": int(row["n_images"]), "checkpoint_sha256": _sha256(checkpoint), "config_sha256": config_sha256(config), "evaluator_git_sha": git_sha, "evaluator_sha256": evaluator_sha})
            compact_rows.append(row)
    fields = ["epoch", "method", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "recomputed", "source", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256", "evaluation_seconds"]
    _write_csv(output / "FINAL_SOURCE_MATRIX.csv", compact_rows, fields)
    effects = _effect_rows(metrics, config_sha256(config), git_sha, evaluator_sha)
    effect_fields = ["target", "epoch", "n_images", "config_sha256", "evaluator_git_sha", "evaluator_sha256"]
    for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
        effect_fields.extend([f"p_{metric}", f"c_old_0_{metric}", f"c_old_05_{metric}", f"a0_{metric}", f"a05_{metric}", f"anchor_train_effect_{metric}", f"anchor_rmt_inference_effect_{metric}", f"anchor_total_effect_{metric}", f"anchor_vs_old_cir_{metric}", f"old_cir_rmt_inference_effect_{metric}"])
    _write_csv(output / "FINAL_SOURCE_DECOMPOSITION.csv", effects, effect_fields)
    _merge_diagnostics(args, new_diagnostics)
    lines = [
        "# Final source trajectory",
        "",
        "Status: PASS. This is a source-only, deterministic 96-image VisA matrix.",
        "",
        "P is the matched Phase2B parent alpha-0 source row; C_OLD_0/C_OLD_05 are reused from the frozen prior CIR run; A0/A05 are the image-parameter-anchor continuation. New GPU forwarding was limited to A E16/E18/E20. A E10/E12/E14 rows were reused from the completed E14 anchor source stage.",
        "",
        "The primary decomposition is: anchor training effect = A0 - P; conditional inference RMT effect = A05 - A0; total anchored CIR effect = A05 - P. These are source-sample associations and do not substitute for the target-domain freeze.",
        "",
        "| epoch | P pixel AUROC | C_OLD_0 pixel AUROC | A0 pixel AUROC | A05 pixel AUROC | A0-P | A05-A0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in effects:
        lines.append(f"| E{int(row['epoch'])} | {float(row['p_pixel_auroc']):.6f} | {float(row['c_old_0_pixel_auroc']):.6f} | {float(row['a0_pixel_auroc']):.6f} | {float(row['a05_pixel_auroc']):.6f} | {float(row['anchor_train_effect_pixel_auroc']):+.6f} | {float(row['anchor_rmt_inference_effect_pixel_auroc']):+.6f} |")
    lines.extend([
        "",
        "AP-tail, raw-vs-deployed, image-branch, and held-out-category diagnostics are in the companion CSVs. Representation preservation is reported separately in SAME_EPOCH_FEATURE_DRIFT.csv and SAME_EPOCH_PARAMETER_DRIFT.csv.",
        "",
        "No Medical or MVTec data were accessed by this stage.",
    ])
    (output / "FINAL_SOURCE_TRAJECTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "FINAL_SOURCE_EVAL_STATUS.json").write_text(json.dumps({"status": "PASS", "source_only": True, "n_images": len(indices), "epochs": list(EPOCHS), "methods": list(METHODS), "new_anchor_epochs": list(NEW_EPOCHS), "reused_anchor_epochs": [10, 12, 14], "reused_parent_and_old_cir": True, "medical": "NOT_RUN", "mvtec": "NOT_RUN", "sample_archive": str(args.sample_archive.expanduser().resolve())}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-archive-root", type=Path, required=True)
    parser.add_argument("--sample-archive", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--old-cir-run-root", type=Path, required=True)
    parser.add_argument("--anchor-run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
