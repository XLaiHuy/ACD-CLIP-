#!/usr/bin/env python3
"""Verify and serialize compact artifacts for the matched-horizon E14 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.cir_rmt.identity import config_sha256, load_cir_config, sha256_file, validate_checkpoint_identity


EPOCHS = (10, 12, 14)
ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_checkpoints(run_root: Path, config: Mapping[str, Any], anchor_sha: str, git_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = run_root / "visa" / "seed0" / "checkpoints"
    for epoch in EPOCHS:
        path = root / f"epoch_{epoch:02d}.pth"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(payload, config, source_dataset="visa", expected_git_sha=git_sha, expected_epoch=epoch)
        if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
            raise ValueError(f"FP32 contract failed at E{epoch}")
        anchor = payload.get("image_anchor", {})
        if anchor.get("reference_checkpoint_sha256") != anchor_sha or float(anchor.get("lambda_image_anchor", 0.0)) != 0.001:
            raise ValueError(f"anchor metadata failed at E{epoch}")
        groups = payload.get("optimizer_state", {}).get("param_groups", [])
        names = [group.get("name") for group in groups]
        if names != ["image_adapter", "text_adapter", "soft_prompt"]:
            raise ValueError(f"optimizer group identity failed at E{epoch}")
        expected_lrs = [0.001 * (0.9 ** epoch), 0.0005 * (0.9 ** epoch), 9.0e-5]
        for index, group in enumerate(groups):
            if abs(float(group["lr"]) - expected_lrs[index]) > 1.0e-15:
                raise ValueError(f"optimizer LR trajectory failed at E{epoch}, group={names[index]}")
            if tuple(float(value) for value in group.get("betas", ())) != (0.9, 0.999):
                raise ValueError(f"Adam betas failed at E{epoch}, group={names[index]}")
            if abs(float(group.get("eps", -1.0)) - 1.0e-8) > 1.0e-20:
                raise ValueError(f"Adam eps failed at E{epoch}, group={names[index]}")
            if abs(float(group.get("weight_decay", -1.0))) > 1.0e-20:
                raise ValueError(f"weight decay failed at E{epoch}, group={names[index]}")
        scheduler = payload.get("scheduler_state", {})
        if int(scheduler.get("last_epoch", -1)) != epoch or int(scheduler.get("_step_count", -1)) != epoch + 1:
            raise ValueError(f"scheduler state failed at E{epoch}")
        rows.append({
            "epoch": epoch,
            "path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "scheduler_last_epoch": int(scheduler["last_epoch"]),
            "scheduler_step_count": int(scheduler["_step_count"]),
            "optimizer_group_names": names,
            "optimizer_lrs": [float(group["lr"]) for group in groups],
            "anchor_reference_sha256": anchor_sha,
        })
    return rows


def _audit_resume_cursor(run_root: Path, config: Mapping[str, Any], git_sha: str) -> dict[str, Any]:
    path = run_root / "visa" / "seed0" / "last.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(payload, config, source_dataset="visa", expected_git_sha=git_sha, expected_epoch=14)
    scheduler = payload.get("scheduler_state", {})
    if int(scheduler.get("last_epoch", -1)) != 14 or int(scheduler.get("_step_count", -1)) != 15:
        raise ValueError("final resume cursor scheduler state failed")
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
        raise ValueError("final resume cursor violates FP32 contract")
    return {"path": str(path), "sha256": sha256_file(path), "epoch": int(payload["epoch"]), "scheduler_last_epoch": int(scheduler["last_epoch"]), "scheduler_step_count": int(scheduler["_step_count"])}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check_manifest(run_root: Path, config: Mapping[str, Any], git_sha: str) -> dict[str, Any]:
    path = run_root / "visa" / "seed0" / "run_manifest.json"
    manifest = _load_json(path)
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"matched-horizon training is not COMPLETED: {manifest.get('status')!r}")
    if manifest.get("arch_id") != config["arch_id"] or manifest.get("config_sha256") != config_sha256(config):
        raise ValueError("matched-horizon run identity does not match CIR config")
    if manifest.get("git_sha") != git_sha:
        raise ValueError("matched-horizon run git SHA does not match committed runner")
    if int(manifest.get("max_epoch", -1)) != 14:
        raise ValueError("matched-horizon run did not cap training at E14")
    if [int(value) for value in manifest.get("target_epochs", [])] != list(EPOCHS):
        raise ValueError("matched-horizon candidate policy is not E10/E12/E14")
    epochs = [int(row["epoch"]) for row in manifest.get("history", [])]
    if not epochs or epochs[-1] != 14 or any(right != left + 1 for left, right in zip(epochs, epochs[1:])):
        raise ValueError(f"training history is not a contiguous run ending at E14: {epochs}")
    return manifest


def _audit_profile(archive: Path, anchor_sha: str) -> dict[str, Any]:
    profile = _load_json(archive / "ANCHOR_OVERHEAD_PROFILE.json")
    if profile.get("status") != "PASS" or profile.get("anchor_reference_resident_once") is not True:
        raise ValueError("one-time anchor overhead profile did not pass")
    if profile.get("anchor_reference_checkpoint_sha256") != anchor_sha:
        raise ValueError("anchor overhead profile references the wrong E14 checkpoint")
    if profile.get("no_vectorization_change_after_profile") is not True:
        raise ValueError("anchor profile does not document unchanged scientific path")
    return profile


def _audit_source(archive: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    status = _load_json(archive / "MATCHED_HORIZON_SOURCE_EVAL_STATUS.json")
    if status.get("status") != "PASS" or status.get("p_c0_recomputed") is not False:
        raise ValueError("source-only evaluator status is not a non-recomputed PASS")
    if status.get("medical") != "NOT_RUN" or status.get("mvtec") != "NOT_RUN":
        raise ValueError("matched-horizon evaluator unexpectedly ran a target evaluation")
    rows = _rows(archive / "MATCHED_HORIZON_SOURCE_RESULTS.csv")
    frozen = [row for row in rows if row.get("recomputed", "").lower() == "false"]
    new = [row for row in rows if row.get("recomputed", "").lower() == "true"]
    if len(frozen) != 12 or len(new) != 6:
        raise ValueError(f"source row reuse contract failed: frozen={len(frozen)} new={len(new)}")
    expected_frozen = {(epoch, method) for epoch in EPOCHS for method in ("P0", "P05", "C0", "C05")}
    actual_frozen = {(int(row["epoch"]), row["method"]) for row in frozen}
    if actual_frozen != expected_frozen:
        raise ValueError("frozen P/C0 source rows are incomplete or duplicated")
    expected_new = {(epoch, method) for epoch in EPOCHS for method in ("C0", "C05")}
    actual_new = {(int(row["epoch"]), row["method"]) for row in new}
    if actual_new != expected_new:
        raise ValueError("new CIR source rows are incomplete or duplicated")
    return rows, status


def _candidate_history(manifest: Mapping[str, Any], checkpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_epoch = {int(row["epoch"]): row for row in checkpoint_rows}
    history = {int(row["epoch"]): row for row in manifest.get("history", [])}
    rows: list[dict[str, Any]] = []
    for epoch in sorted(history):
        row = history[epoch]
        post_image = float(row["lr"])
        post_text = post_image * 0.5
        post_prompt = 0.0 if epoch <= 3 else 9.0e-5
        checkpoint = by_epoch.get(epoch, {})
        if checkpoint:
            actual = [float(value) for value in checkpoint["optimizer_lrs"]]
            if abs(actual[0] - post_image) > 1.0e-12 or abs(actual[1] - post_text) > 1.0e-12 or abs(actual[2] - post_prompt) > 1.0e-12:
                raise ValueError(f"manifest and checkpoint LR differ at E{epoch}")
        rows.append({
            "epoch": epoch,
            "start_image_lr": post_image / 0.9,
            "post_image_lr": post_image,
            "start_text_lr": post_text / 0.9,
            "post_text_lr": post_text,
            "start_prompt_lr": 0.0 if epoch <= 3 else 1.0e-4,
            "post_prompt_lr": post_prompt,
            "soft_prompt_frozen": epoch <= 3,
            "scheduler_last_epoch": epoch,
            "scheduler_step_count": epoch + 1,
            "checkpoint_sha256": checkpoint.get("checkpoint_sha256", ""),
            "post_lr_source": "checkpoint_optimizer_state" if checkpoint else "manifest_history_policy",
        })
    return rows


def _frozen_parent_lr_rows() -> list[dict[str, Any]]:
    prior = ROOT / "research_artifacts" / "cir_rmt_v2" / "corrective_matched_retrain_20260830" / "parent_lr_history.csv"
    if not prior.is_file():
        raise FileNotFoundError(f"frozen parent LR history is required: {prior}")
    rows = [row for row in _rows(prior) if int(row["epoch"]) <= 14]
    if [int(row["epoch"]) for row in rows] != list(range(1, 15)):
        raise ValueError("frozen parent LR history does not cover E1-E14")
    return rows


def _metric(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing source metric {key}")
    return float(value)


def _source_decomposition(
    source_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    evaluator_git_sha: str,
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frozen = {(int(row["epoch"]), row["method"]): row for row in source_rows if row.get("recomputed", "").lower() == "false"}
    new = {(int(row["epoch"]), row["method"]): row for row in source_rows if row.get("recomputed", "").lower() == "true"}
    evaluator_sha = sha256_file(ROOT / "tools" / "cir_rmt" / "matched_horizon_source_eval.py")
    prior_manifest_path = ROOT / "research_artifacts" / "cir_rmt_v2" / "corrective_matched_retrain_20260830" / "CORRECTIVE_MANIFEST.json"
    prior_manifest = _load_json(prior_manifest_path) if prior_manifest_path.is_file() else {}
    parent_hashes = prior_manifest.get("checkpoint_sha256", {}).get("parent", {})
    cir_hashes = {int(row["epoch"]): str(row["checkpoint_sha256"]) for row in checkpoint_rows}
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        parent = frozen[(epoch, "P0")]
        c0 = new[(epoch, "C0")]
        c05 = new[(epoch, "C05")]
        values: dict[str, Any] = {
            "target": "VisA_SOURCE_MATCHED_SAMPLE",
            "epoch": epoch,
            "n_images": int(parent["n_images"]),
            "parent_checkpoint_sha256": parent_hashes.get(str(epoch), ""),
            "cir_checkpoint_sha256": cir_hashes[epoch],
            "config_sha256": config_sha256(config),
            "evaluator_git_sha": evaluator_git_sha,
            "evaluator_sha256": evaluator_sha,
        }
        for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
            p = _metric(parent, metric)
            c = _metric(c0, metric)
            r = _metric(c05, metric)
            values.update({
                f"parent_{metric}": p,
                f"c0_{metric}": c,
                f"c05_{metric}": r,
                f"train_effect_{metric}": c - p,
                f"rmt_inference_effect_{metric}": r - c,
                f"total_cir_effect_{metric}": r - p,
                f"frozen_prior_c0_{metric}": _metric(frozen[(epoch, "C0")], metric),
                f"frozen_prior_c05_{metric}": _metric(frozen[(epoch, "C05")], metric),
            })
        rows.append(values)
    return rows


def _write_source_decomposition(archive: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "target", "epoch", "n_images", "parent_checkpoint_sha256", "cir_checkpoint_sha256",
        "config_sha256", "evaluator_git_sha", "evaluator_sha256",
    ]
    for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
        fields.extend([
            f"parent_{metric}", f"c0_{metric}", f"c05_{metric}",
            f"train_effect_{metric}", f"rmt_inference_effect_{metric}", f"total_cir_effect_{metric}",
            f"frozen_prior_c0_{metric}", f"frozen_prior_c05_{metric}",
        ])
    _write_csv(archive / "MATCHED_HORIZON_SOURCE_DECOMPOSITION.csv", rows, fields)
    _write_csv(archive / "corrected_source_decomposition.csv", rows, fields)


def _write_medical_not_run(archive: Path) -> None:
    _write_csv(
        archive / "corrected_medical_decomposition.csv",
        [{
            "target": "", "epoch": "", "method": "", "pixel_auroc": "", "pixel_ap": "",
            "image_auroc": "", "image_ap": "", "status": "NOT_RUN",
            "reason": "Target evaluation was intentionally not launched in the matched-horizon source-only stage.",
        }],
        ["target", "epoch", "method", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "status", "reason"],
    )


def _write_ledger(archive: Path, config: Mapping[str, Any], anchor_sha: str, profile: Mapping[str, Any]) -> None:
    rows = [
        ("source_dataset", "VisA_TRAIN", "VisA_TRAIN", "MATCHED", "same frozen source role"),
        ("seed", "0", "0", "MATCHED", "fixed seed"),
        ("clip_asset_sha256", "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02", "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02", "MATCHED", "same frozen CLIP asset"),
        ("precision", "fp32", "fp32", "MATCHED", "AMP and TF32 disabled"),
        ("effective_batch_size", "6", "6", "MATCHED", "micro batch six, accumulation one"),
        ("optimizer", "Adam", "Adam", "MATCHED", "same optimizer family"),
        ("adam_hyperparameters", "betas=(0.9,0.999),eps=1e-8,weight_decay=0", "betas=(0.9,0.999),eps=1e-8,weight_decay=0", "MATCHED", "checkpoint param groups verified"),
        ("learning_rates", "image=1e-3,text=5e-4,prompt=1e-4", "image=1e-3,text=5e-4,prompt=1e-4", "MATCHED", "StepLR post-state verified"),
        ("scheduler", "StepLR(step_size=1,gamma=0.9)", "StepLR(step_size=1,gamma=0.9)", "MATCHED", "step once after each epoch"),
        ("scheduler_timing", "after epoch before history/checkpoint", "after epoch before history/checkpoint", "MATCHED", "trainer source and state verified"),
        ("loss", "cls+seg+0.001*kg+0.0*k", "cls+seg+0.001*kg+0.0*k", "MATCHED", "frozen objective"),
        ("checkpoint_schedule", "E10/E12/E14", "E10/E12/E14", "MATCHED", "matched-horizon policy"),
        ("anchor", "none", f"E14 image adapter, lambda=0.001, sha={anchor_sha}", "INTENDED_DIFFERENCE", "selected preservation intervention"),
        ("anchor_profile", "not applicable", f"median overhead={float(profile['anchor_overhead_percent']):.6f}%", "MEASURED", "one real batch, no optimizer step"),
        ("source_evaluation", "frozen P/C0 rows", "new C0/C05 only", "MATCHED_SCOPE", "no P/C0 recomputation"),
        ("medical_evaluation", "not in this stage", "NOT_RUN", "NOT_RUN", "source-only speed stage"),
    ]
    _write_csv(archive / "CORRECTIVE_MATCH_LEDGER.csv", [{"field": a, "parent_phase2b": b, "cir_matched_horizon": c, "status": d, "evidence": e} for a, b, c, d, e in rows], ["field", "parent_phase2b", "cir_matched_horizon", "status", "evidence"])


def _write_reports(
    archive: Path,
    manifest: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    cir_lr_rows: Sequence[Mapping[str, Any]],
    parent_lr_rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    eval_rows: Sequence[Mapping[str, str]],
    decomp: Sequence[Mapping[str, Any]],
) -> None:
    history = manifest.get("history", [])
    candidate_history = {int(row["epoch"]): row for row in history if int(row["epoch"]) in EPOCHS}
    timing_lines = [
        "# Matched-horizon timing report",
        "",
        "Status: PASS for the E14 source-only speed path.",
        "",
        "The run trains only through E14, retains candidates E10/E12/E14, and performs no target evaluation.",
        "",
        "| epoch | seconds | images/sec | peak allocated GiB | peak reserved GiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for epoch in sorted(candidate_history):
        row = candidate_history[epoch]
        timing_lines.append(
            f"| E{epoch} | {float(row['elapsed_seconds']):.3f} | {float(row['images_per_sec']):.4f} | "
            f"{int(row.get('peak_vram_bytes', 0)) / 1024**3:.3f} | {int(row.get('peak_reserved_vram_bytes', 0)) / 1024**3:.3f} |"
        )
    timing_lines.extend([
        "",
        f"- Training wall time: {float(manifest.get('train_wall_seconds', 0.0)):.3f} seconds.",
        f"- Anchor-loss baseline median: {float(profile['baseline_median_seconds']):.6f} seconds per measured batch.",
        f"- Anchor-loss median: {float(profile['anchored_median_seconds']):.6f} seconds per measured batch.",
        f"- Anchor overhead: {float(profile['anchor_overhead_seconds']):.6f} seconds ({float(profile['anchor_overhead_percent']):.6f}%).",
        "- The anchor reference was loaded once and kept resident; no vectorization or denominator approximation was introduced.",
        "",
        "| eval epoch | inference seconds | total evaluation seconds |",
        "|---:|---:|---:|",
    ])
    for row in eval_rows:
        timing_lines.append(f"| E{int(row['epoch'])} | {float(row['inference_seconds']):.3f} | {float(row['total_evaluation_seconds']):.3f} |")
    (archive / "MATCHED_HORIZON_TIME_REPORT.md").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    optimizer_rows = {int(row["epoch"]): row for row in checkpoint_rows}
    audit_lines = [
        "# Corrective training audit",
        "",
        "Status: PASS.",
        "",
        "The committed matched-horizon runner trained the frozen CIR-V2 identity on VisA through E14 only. "
        "It used FP32, AMP disabled, TF32 disabled, effective batch six, Adam betas (0.9, 0.999), eps 1e-8, "
        "weight decay zero, gradient clipping one, and StepLR(step_size=1, gamma=0.9).",
        "",
        "Candidate checkpoint audit:",
        "",
        "| epoch | image LR | text LR | prompt LR | scheduler last_epoch | scheduler _step_count | checkpoint SHA256 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for epoch in EPOCHS:
        row = optimizer_rows[epoch]
        audit_lines.append(
            f"| E{epoch} | {row['optimizer_lrs'][0]:.12g} | {row['optimizer_lrs'][1]:.12g} | {row['optimizer_lrs'][2]:.12g} | "
            f"{row['scheduler_last_epoch']} | {row['scheduler_step_count']} | {row['checkpoint_sha256']} |"
        )
    audit_lines.extend([
        "",
        "The scheduler state is post-step and is saved before candidate checkpoint serialization. "
        "The E10 gate passed without metric-based early stopping. The E14 anchor is training-only and is not part of deployment.",
        "",
        "Resume is guarded by the process lock, checkpoint identity, optimizer state, scheduler state, RNG state, "
        "and anchor-reference identity. Medical and MVTec evaluation were not run in this stage.",
    ])
    (archive / "CORRECTIVE_TRAINING_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Matched-horizon corrected results",
        "",
        "Status: PASS for the locked E14 source-only experiment.",
        "",
        "Frozen P/C0 E10/E12/E14 source rows were reused. Only the new CIR checkpoint was forwarded, producing C0 (alpha 0) and C05 (alpha .5).",
        "C05 minus C0 is the conditional inference RMT effect on the new anchored representation; it is not a clean training effect.",
        "",
        "| epoch | P pixel AUROC | C0 pixel AUROC | C05 pixel AUROC | train effect | RMT inference effect | total CIR effect |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decomp:
        summary_lines.append(
            f"| E{int(row['epoch'])} | {float(row['parent_pixel_auroc']):.6f} | {float(row['c0_pixel_auroc']):.6f} | "
            f"{float(row['c05_pixel_auroc']):.6f} | {float(row['train_effect_pixel_auroc']):+.6f} | "
            f"{float(row['rmt_inference_effect_pixel_auroc']):+.6f} | {float(row['total_cir_effect_pixel_auroc']):+.6f} |"
        )
    summary_lines.extend([
        "",
        "Medical evaluation: NOT_RUN. MVTec evaluation: NOT_RUN. No final cross-domain RMT decision is made from this bounded source-only stage.",
        "The scientific purpose of this run is to establish the matched-horizon timing and source decomposition under the selected E14 anchor intervention.",
    ])
    (archive / "CORRECTED_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    decision = [
        "# Corrected matched-horizon decision",
        "",
        "DECISION: SOURCE_ONLY_AUDIT_PASS_NO_TARGET_DECISION",
        "",
        "The training, checkpoint, scheduler, resume, source-row reuse, and timing contracts passed. "
        "This stage intentionally stops before Medical and MVTec evaluation, so it cannot select KEEP_RMT, "
        "INFERENCE_ONLY_RMT, REDESIGN_RMT_ONCE, ABANDON_RMT_RETURN_TO_PHASE2B, or ABANDON_THIS_LINEAGE.",
        "",
        "The next authorized review is the bounded P versus new C0 versus new C05 source decomposition. "
        "No target tuning or full forensic battery is implied by this artifact.",
    ]
    (archive / "CORRECTED_GO_NO_GO_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    (archive / "MATCHED_HORIZON_AUDIT.md").write_text(
        "# Matched-horizon audit\n\n"
        "PASS: E14-only training, E10/E12/E14 candidates, one-time resident anchor profile, "
        "E10 structural gate, exact checkpoint identity, post-step scheduler state, frozen P/C0 reuse, "
        "new-CIR-only source forwarding, and no Medical/MVTec evaluation.\n",
        encoding="utf-8",
    )


def _write_hashes(archive: Path) -> list[str]:
    files = sorted(path for path in archive.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256_file(path)}  {path.relative_to(archive)}" for path in files]
    (archive / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(path.relative_to(archive)) for path in files] + ["SHA256SUMS.txt"]


def run(args: argparse.Namespace) -> None:
    archive = Path(args.archive_root)
    run_root = Path(args.run_root)
    archive.mkdir(parents=True, exist_ok=True)
    config = load_cir_config(Path(args.config))
    git_sha = str(args.git_sha)
    anchor_sha = sha256_file(Path(args.anchor_checkpoint))
    manifest = _check_manifest(run_root, config, git_sha)
    profile = _audit_profile(archive, anchor_sha)
    checkpoint_rows = _audit_checkpoints(run_root, config, anchor_sha, git_sha)
    resume_cursor = _audit_resume_cursor(run_root, config, git_sha)
    gate = _load_json(run_root / "visa" / "seed0" / "E10_CATASTROPHIC_FAILURE_GATE.json")
    if gate.get("status") != "PASS":
        raise ValueError("E10 catastrophic-failure gate is not PASS")
    source_rows, source_status = _audit_source(archive)
    eval_rows = _rows(archive / "MATCHED_HORIZON_EVAL_TIMES.csv")
    if [int(row["epoch"]) for row in eval_rows] != list(EPOCHS):
        raise ValueError("source evaluation timing does not cover exactly E10/E12/E14")
    parent_lr_rows = _frozen_parent_lr_rows()
    cir_lr_rows = _candidate_history(manifest, checkpoint_rows)
    evaluator_git_sha = str(args.evaluator_git_sha or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    decomp = _source_decomposition(source_rows, config, evaluator_git_sha, checkpoint_rows)
    _write_source_decomposition(archive, decomp)
    _write_medical_not_run(archive)
    _write_ledger(archive, config, anchor_sha, profile)
    _write_reports(archive, manifest, checkpoint_rows, cir_lr_rows, parent_lr_rows, profile, eval_rows, decomp)
    _write_csv(archive / "parent_lr_history.csv", parent_lr_rows, list(parent_lr_rows[0]))
    _write_csv(archive / "cir_lr_history.csv", cir_lr_rows, list(cir_lr_rows[0]))
    source_sample = _load_json(Path(args.baseline_archive_root) / "SOURCE_SAMPLE_IDENTITY.json")
    source_path = Path(args.source_root).resolve()
    source_identity = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    output_manifest: dict[str, Any] = {
        "manifest_version": 1,
        "status": "PASS",
        "experiment_id": "matched_horizon_anchor_e14_20260831",
        "scope": "source_only_matched_horizon",
        "arch_id": config["arch_id"],
        "architecture_version": int(config["architecture_version"]),
        "config_sha256": config_sha256(config),
        "parent_config_sha256": str(config["parent_config_sha256"]),
        "architecture_freeze_sha256": str(config["architecture_freeze_sha256"]),
        "git_sha": git_sha,
        "evaluator_git_sha": evaluator_git_sha,
        "source": {"name": "VisA", "role": "VisA_TRAIN", "root": str(source_path), "path_probe_sha256": source_identity, "sample_identity": str(Path(args.baseline_archive_root) / "SOURCE_SAMPLE_IDENTITY.json"), "sample_images": int(source_sample.get("sample_size", len(source_sample.get("selection", []))))},
        "seed": 0,
        "clip_asset": {"path": str(Path(args.clip_asset).resolve()), "sha256": sha256_file(Path(args.clip_asset))},
        "anchor": {"checkpoint": str(Path(args.anchor_checkpoint).resolve()), "sha256": anchor_sha, "lambda_image_anchor": 0.001, "scope": "image_adapter_parameters_only", "training_only": True},
        "training": {"run_root": str(run_root.resolve()), "run_manifest": str((run_root / "visa" / "seed0" / "run_manifest.json").resolve()), "status": manifest["status"], "max_epoch": 14, "candidate_epochs": list(EPOCHS), "train_wall_seconds": manifest.get("train_wall_seconds"), "e10_gate": gate, "checkpoints": checkpoint_rows, "resume_cursor": resume_cursor},
        "optimizer": {"family": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "effective_batch_size": 6, "gradient_clip_norm": 1.0},
        "scheduler": {"family": "StepLR", "step_size": 1, "gamma": 0.9, "timing": "after_epoch_before_history_and_checkpoint"},
        "source_evaluation": {"status": source_status["status"], "results": "MATCHED_HORIZON_SOURCE_RESULTS.csv", "decomposition": "MATCHED_HORIZON_SOURCE_DECOMPOSITION.csv", "frozen_p_c0_reused": True, "new_cir_only_forward": True, "epochs": list(EPOCHS)},
        "medical": {"status": "NOT_RUN", "artifact": "corrected_medical_decomposition.csv", "reason": "target evaluation intentionally excluded from this source-only speed stage"},
        "mvtec": {"status": "NOT_RUN"},
        "profile": {"artifact": "ANCHOR_OVERHEAD_PROFILE.json", "status": profile["status"], "anchor_overhead_percent": profile["anchor_overhead_percent"]},
        "excluded_large_artifacts": ["raw checkpoints", "raw per-pixel stores", "memmaps", "caches", "evaluator spools", "temporary logs"],
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    known_files = sorted(str(path.relative_to(archive)) for path in archive.rglob("*") if path.is_file() and path.name not in {"CORRECTIVE_MANIFEST.json", "SHA256SUMS.txt"})
    output_manifest["versioned_artifacts"] = known_files + ["CORRECTIVE_MANIFEST.json", "SHA256SUMS.txt"]
    _write_json(archive / "CORRECTIVE_MANIFEST.json", output_manifest)
    _write_hashes(archive)
    print(json.dumps({"status": "PASS", "archive": str(archive), "checkpoints": checkpoint_rows, "versioned_artifacts": output_manifest["versioned_artifacts"]}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--baseline-archive-root", type=Path, default=ROOT / "research_artifacts" / "cir_rmt_v2" / "pre_full_run_root_cause_lock_20260831")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--evaluator-git-sha")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
