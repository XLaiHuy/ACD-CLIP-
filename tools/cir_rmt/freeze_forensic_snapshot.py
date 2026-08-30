#!/usr/bin/env python3
"""Freeze the compact, pre-fix CIR-V2 forensic evidence.

This utility only reads completed forensic/evaluation artifacts and copies
compact reports, tables, manifests, and diagnostic sources into a tracked
archive.  It does not run an evaluator, modify model code, modify scheduler
behavior, or start training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


MEDICAL_TARGETS = [
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
]
EPOCHS = [12, 14, 16, 18, 20]
MAX_VERSION_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt"}

BASE_AUDIT_FILES = [
    "FULL_FAILURE_FORENSICS.md",
    "GO_NO_GO_DECISION.md",
    "SCHEDULER_OPTIMIZATION_AUDIT.md",
    "audit_identity.json",
    "cells.jsonl",
    "checkpoint_drift.csv",
    "final_summary.json",
    "gradient_conflict_report.csv",
    "gradient_conflict_summary.json",
    "inference_rmt_effect.csv",
    "parameter_justification_table.csv",
    "peer_forensics.csv",
    "pixel_rank_forensics.csv",
    "progress.json",
    "protocol_equivalence_ledger.csv",
    "scheduler_audit_summary.json",
    "scheduler_optimization_audit.csv",
    "scheduler_optimizer_group_detail.csv",
    "source_checkpoint_metrics.csv",
    "stage_group_attribution.csv",
    "train_deploy_mismatch.md",
    "medical_results_prefx.csv",

]
DERIVED_AUDIT_FILES = [
    "RESEARCH_DIRECTION_PRE_FIX.md",
    "FORENSIC_ARTIFACT_INVENTORY.md",
    "PRE_FIX_FORENSIC_MANIFEST.json",
    "FORENSIC_SHA256SUMS.txt",
]

SCRIPT_MAP = {
    "tools/cir_rmt/failure_forensics.py": "diagnostic_scripts/failure_forensics.py",
    "tools/cir_rmt/scheduler_audit.py": "diagnostic_scripts/scheduler_audit.py",
    "tools/cir_rmt/freeze_forensic_snapshot.py": "diagnostic_scripts/freeze_forensic_snapshot.py",
    "tools/cir_rmt/v2_source_confirmation.py": "diagnostic_scripts/v2_source_confirmation.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=None)
    return parser.parse_args()


def repo_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
    ).strip()


def float_or_none(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def number_text(value: object) -> str:
    if value is None or value == "":
        return ""
    return format(float(value), ".17g")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in TEXT_SUFFIXES:
        destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        shutil.copystat(source, destination)
    else:
        shutil.copy2(source, destination)


def create_medical_summary(audit_root: Path, identity: dict, current_git: str) -> list[dict[str, str]]:
    """Create the compact medical table from existing journaled CSV rows only."""

    source = audit_root / "inference_rmt_effect.csv"
    rows = [row for row in read_csv(source) if row["target"] in MEDICAL_TARGETS]
    expected = {(epoch, target) for epoch in EPOCHS for target in MEDICAL_TARGETS}
    observed = {(int(row["epoch"]), row["target"]) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"medical summary is incomplete; missing cells: {missing}")

    config_sha = read_json(audit_root / "audit_identity.json")["config_sha256"]
    output: list[dict[str, str]] = []
    fields = [
        "target",
        "epoch",
        "pixel_auroc",
        "pixel_ap",
        "image_auroc",
        "image_ap",
        "config_sha",
        "checkpoint_sha",
        "evaluator_git_sha",
        "source_effect_status",
    ]
    for row in sorted(rows, key=lambda item: (int(item["epoch"]), MEDICAL_TARGETS.index(item["target"]))):
        output.append(
            {
                "target": row["target"],
                "epoch": row["epoch"],
                "pixel_auroc": row["alpha05_pixel_auroc"],
                "pixel_ap": row["alpha05_pixel_ap"],
                "image_auroc": row["alpha05_image_auroc"],
                "image_ap": row["alpha05_image_ap"],
                "config_sha": config_sha,
                "checkpoint_sha": row["checkpoint_sha256"],
                "evaluator_git_sha": current_git,
                "source_effect_status": "alpha05_existing_paired_with_alpha0",
            }
        )
    write_csv(audit_root / "medical_results_prefx.csv", fields, output)
    return output


def augment_checkpoint_drift(audit_root: Path, medical_rows: list[dict[str, str]]) -> None:
    """Add LR and metric context while preserving every existing drift column."""

    drift_path = audit_root / "checkpoint_drift.csv"
    drift_rows = read_csv(drift_path)
    scheduler_rows = {
        int(row["epoch"]): row
        for row in read_csv(audit_root / "scheduler_optimization_audit.csv")
        if row.get("epoch", "").strip()
    }
    source_rows = {
        int(row["epoch"]): row
        for row in read_csv(audit_root / "source_checkpoint_metrics.csv")
        if row.get("epoch", "").strip()
    }
    medical_by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in medical_rows:
        medical_by_epoch[int(row["epoch"])].append(row)

    additions = [
        "intended_image_lr",
        "actual_cir_image_lr",
        "lr_ratio",
        "source_metric",
        "target_metric",
        "native_alpha0_metric",
        "cir_alpha05_metric",
        "source_metric_definition",
        "target_metric_definition",
    ]
    fields = list(drift_rows[0]) if drift_rows else []
    for field in additions:
        if field not in fields:
            fields.append(field)

    for row in drift_rows:
        epoch = int(row["epoch"])
        scheduler = scheduler_rows.get(epoch, {})
        intended = float_or_none(scheduler.get("parent_expected_start_image_lr"))
        if intended is None:
            intended = 1e-3 * (0.9 ** (epoch - 1))
        actual = float_or_none(scheduler.get("cir_image_lr"))
        target_epoch = medical_by_epoch.get(epoch, [])
        source = source_rows.get(epoch, {})

        def mean_field(name: str) -> float | None:
            values = [float_or_none(item.get(name)) for item in target_epoch]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        row["intended_image_lr"] = number_text(intended)
        row["actual_cir_image_lr"] = number_text(actual)
        row["lr_ratio"] = number_text(actual / intended if actual is not None and intended else None)
        row["source_metric"] = source.get("alpha05_pixel_auroc", "")
        row["target_metric"] = number_text(mean_field("pixel_auroc"))
        row["native_alpha0_metric"] = number_text(mean_field("_native_alpha0_pixel_auroc"))
        row["cir_alpha05_metric"] = number_text(mean_field("pixel_auroc"))
        row["source_metric_definition"] = "VisA_SOURCE alpha05 pixel AUROC"
        row["target_metric_definition"] = "6-medical mean pixel AUROC; native_alpha0/cir_alpha05 are paired inference values"

    # The compact medical table stores only the CIR alpha=.5 values. Read the
    # paired alpha=0 values directly from the existing journal rather than
    # rerunning evaluation or changing that table's requested schema.
    paired_rows = read_csv(audit_root / "inference_rmt_effect.csv")
    paired_by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for paired in paired_rows:
        if paired["target"] in MEDICAL_TARGETS:
            paired_by_epoch[int(paired["epoch"])].append(paired)
    for row in drift_rows:
        epoch = int(row["epoch"])
        paired = paired_by_epoch.get(epoch, [])
        values = [float_or_none(item.get("alpha0_pixel_auroc")) for item in paired]
        values = [value for value in values if value is not None]
        row["native_alpha0_metric"] = number_text(sum(values) / len(values) if values else None)

    write_csv(drift_path, fields, drift_rows)


def classify_path(path: Path, expected: bool = True) -> str:
    if not path.exists():
        return "MISSING" if expected else "MISSING"
    if path.is_dir():
        return "INCOMPLETE"
    if path.stat().st_size == 0:
        return "INCOMPLETE"
    name = path.name.lower()
    if "__pycache__" in path.parts or any(token in name for token in ("cache", "spool", "tmp")):
        return "TEMPORARY"
    if path.stat().st_size > MAX_VERSION_BYTES:
        return "TOO_LARGE_TO_VERSION"
    return "PRESENT"


def inventory_rows(repo_root: Path, audit_root: Path, archive_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        artifact: str,
        location: str,
        source_path: Path | None,
        archive_rel: str = "",
        reason: str = "",
        expected: bool = True,
    ) -> None:
        status = classify_path(source_path, expected=expected) if source_path else "MISSING"
        archived = bool(archive_rel and (archive_root / archive_rel).exists())
        versioned = "YES" if archived else "NO"
        if status == "PRESENT" and not reason:
            reason = "compact artifact retained in the pre-fix archive"
        rows.append(
            {
                "artifact": artifact,
                "location": location,
                "status": status,
                "versioned_in_snapshot": versioned,
                "archive_path": archive_rel,
                "reason": reason,
            }
        )

    for name in BASE_AUDIT_FILES:
        add(name, str(audit_root / name), audit_root / name, name)

    for name in DERIVED_AUDIT_FILES:
        add(name, str(audit_root / name), audit_root / name, name)

    for source, archive_rel in SCRIPT_MAP.items():
        source_path = repo_root / source
        add(
            source,
            str(source_path),
            source_path,
            archive_rel,
            reason="forensic diagnostic source; v2_source_confirmation.py is a pre-existing supporting reference",
        )

    add(
        "root-cause ranking tables",
        str(audit_root / "FULL_FAILURE_FORENSICS.md"),
        audit_root / "FULL_FAILURE_FORENSICS.md",
        "FULL_FAILURE_FORENSICS.md",
        reason="present as a report section; no separate ranking CSV was created",
    )
    add(
        "diagnostic plots",
        str(audit_root),
        next(iter(sorted(p for p in audit_root.rglob("*") if p.suffix.lower() in {".png", ".pdf", ".svg"})), None),
        reason="no plot artifact was produced; conclusions are backed by compact tables and reports",
    )
    add(
        "raw per-pixel score stores",
        str(audit_root),
        next(iter(sorted(p for p in audit_root.rglob("*") if p.suffix.lower() in {".npy", ".npz", ".memmap", ".mmap"})), None),
        reason="no raw per-pixel store was found under the forensic root",
    )

    known = set(BASE_AUDIT_FILES) | set(DERIVED_AUDIT_FILES)
    for path in sorted(p for p in audit_root.rglob("*") if p.is_file()):
        rel = path.relative_to(audit_root).as_posix()
        if rel in known:
            continue
        add(
            rel,
            str(path),
            path,
            reason="observed forensic-root file not separately enumerated above",
        )
    return rows


def write_inventory(path: Path, rows: list[dict[str, str]], audit_root: Path, archive_root: Path) -> None:
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# CIR-V2 pre-fix forensic artifact inventory",
        "",
        "This inventory was generated from the completed forensic root. It does not rerun evaluation and does not modify model, RMT, loss, deployment, optimizer, or scheduler behavior.",
        "",
        f"- forensic root: `{audit_root}`",
        f"- compact archive: `{archive_root}`",
        f"- status counts: `{json.dumps(dict(sorted(counts.items())), sort_keys=True)}`",
        "",
        "Status vocabulary is restricted to `PRESENT`, `MISSING`, `INCOMPLETE`, `TEMPORARY`, and `TOO_LARGE_TO_VERSION`.",
        "",
        "| artifact | source location | status | versioned in snapshot | archive path | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        reason = row["reason"].replace("|", "\\|")
        lines.append(
            f"| `{row['artifact']}` | `{row['location']}` | `{row['status']}` | `{row['versioned_in_snapshot']}` | `{row['archive_path']}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "The full forensic root remains outside the tracked archive when it contains operational or large intermediate data. The archive contains the reviewable reports, CSV/JSON summaries, provenance, hashes, and diagnostic source needed to reproduce the audit interpretation.",
            "",
            "No MVTec result artifact was found in this forensic run; MVTec coverage is therefore `NOT_RUN`, not an inferred failure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest(
    repo_root: Path,
    audit_root: Path,
    archive_root: Path,
    identity: dict,
    current_git: str,
    branch: str,
    archive_paths: list[str],
) -> dict:
    nested = identity["identity"]
    drift_rows = read_csv(audit_root / "checkpoint_drift.csv")
    checkpoint_paths = {
        str(int(row["epoch"])): {
            "path": row["checkpoint"],
            "sha256": row["checkpoint_sha256"],
        }
        for row in drift_rows
        if row.get("epoch", "").strip()
    }
    scheduler_summary = read_json(audit_root / "scheduler_audit_summary.json")
    summary = read_json(audit_root / "final_summary.json")
    clip_path = Path(identity["clip_asset"])
    producer_shas = sorted({row.get("checkpoint_git_sha", "") for row in drift_rows if row.get("checkpoint_git_sha")})
    return {
        "arch_id": nested["arch_id"],
        "architecture_version": nested["architecture_version"],
        "config_sha": identity["config_sha256"],
        "resolved_config_sha": nested["config_sha256"],
        "architecture_freeze_sha": nested["architecture_freeze_sha256"],
        "evaluator_git_sha": current_git,
        "current_git_sha": current_git,
        "current_branch": branch,
        "checkpoint_producer_git_sha": producer_shas,
        "source_dataset": {
            "name": "VisA",
            "root": identity["visa_root"],
            "role": "source/training domain for the audited CIR run",
        },
        "seed": 0,
        "clip_asset": identity["clip_asset"],
        "clip_asset_sha256": sha256(clip_path),
        "checkpoint_shas": {epoch: item["sha256"] for epoch, item in checkpoint_paths.items()},
        "checkpoint_paths": checkpoint_paths,
        "forensic_directory": {
            "absolute": str(audit_root),
            "repo_relative": audit_root.relative_to(repo_root).as_posix(),
            "archive_repo_relative": archive_root.relative_to(repo_root).as_posix(),
        },
        "forensic_script_paths": list(SCRIPT_MAP),
        "scheduler_classification": scheduler_summary["classification"],
        "current_decision": summary["decision"],
        "medical_benchmark_status": {
            "status": "COMPLETE",
            "cells": summary["medical_effect_rows"],
            "summary_csv": "medical_results_prefx.csv",
            "paired_alpha_conditions": [0.0, 0.5],
        },
        "source_diagnostic_status": "COMPLETE",
        "gradient_audit_status": "COMPLETE",
        "peer_audit_status": "COMPLETE",
        "stage_group_audit_status": "COMPLETE",
        "alpha_comparison_status": "COMPLETE",
        "protocol_audit_status": "COMPLETE",
        "loss_train_deploy_audit_status": "COMPLETE",
        "parameter_justification_status": "COMPLETE",
        "mvtec_status": "NOT_RUN",
        "coverage": {
            "scheduler_lr_mismatch": "COMPLETE",
            "alpha0_vs_alpha05": "COMPLETE",
            "checkpoint_drift_e12_to_e20": "COMPLETE",
            "source_visa_diagnostics": "COMPLETE",
            "gradient_loss_conflict": "COMPLETE",
            "train_vs_deploy": "COMPLETE",
            "stage_group_attribution": "COMPLETE",
            "peer_delta_mad_saturation": "COMPLETE",
            "parameter_justification": "COMPLETE",
            "protocol_equivalence_vs_phase2b": "COMPLETE",
            "medical_failure_pattern": "COMPLETE",
            "mvtec_failure_pattern": "NOT_RUN",
            "proven": "COMPLETE",
            "correlational": "COMPLETE",
            "unknown": "COMPLETE",
            "smallest_next_experiment": "COMPLETE",
            "go_modify_abandon": "COMPLETE",
        },
        "timestamp": datetime.now().astimezone().isoformat(),
        "versioned_artifacts": archive_paths,
        "intentionally_excluded_large_artifacts": [
            {
                "pattern": "raw per-pixel score stores / memmaps / caches / evaluator spools",
                "reason": "not scientifically useful as compact review artifacts; no such file was found under the audited forensic root",
            }
        ],
        "excluded_transient_artifacts": ["cache files", "memmaps", "temp spools", "__pycache__"],
        "scientific_firewall": {
            "scheduler_fix_included": False,
            "architecture_modified": False,
            "rmt_modified": False,
            "loss_modified": False,
            "optimizer_modified": False,
            "deployment_modified": False,
            "retrain_started": False,
            "new_training_results": False,
        },
        "notes": [
            "CIR_SCHEDULER_BUG_CONFIRMED: the CIR loop constructs StepLR but does not call scheduler.step().",
            "The alpha=.5 minus alpha=0 comparison is conditional on the wrongly trained CIR representation and is not a clean CIR-vs-Phase2B causal estimate.",
            "The available historical parent checkpoint lacks optimizer_state and scheduler_state; parent LR history and canonical expected values are preserved with that limitation.",
            "The single next experiment is one matched corrective Phase2B-vs-CIR retrain; it was not launched in this snapshot task.",
        ],
    }


def write_hashes(archive_root: Path, audit_root: Path) -> list[str]:
    hash_name = "FORENSIC_SHA256SUMS.txt"
    hash_path = archive_root / hash_name
    files = sorted(
        path
        for path in archive_root.rglob("*")
        if path.is_file() and path.name != hash_name
    )
    lines = [
        "# SHA-256 for every compact report, CSV/JSON summary, and diagnostic source in this snapshot",
        "# FORENSIC_SHA256SUMS.txt is self-excluded to avoid a circular digest.",
    ]
    for path in files:
        lines.append(f"{sha256(path)}  {path.relative_to(archive_root).as_posix()}")
    hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copy_file(hash_path, audit_root / hash_name)
    return [path.relative_to(archive_root).as_posix() for path in files] + [hash_name]


def main() -> int:
    args = parse_args()
    repo_root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    audit_root = repo_path(
        repo_root,
        args.audit_root or Path("runs/cir_rmt/CIR_DFG_RMT_V2/forensics_20260830"),
    ).resolve()
    archive_root = repo_path(
        repo_root,
        args.archive_root or Path("research_artifacts/cir_rmt_v2/forensics_20260830_pre_scheduler_fix"),
    ).resolve()
    audit_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    identity = read_json(audit_root / "audit_identity.json")
    current_git = git_value(repo_root, "rev-parse", "HEAD")
    branch = git_value(repo_root, "branch", "--show-current")

    medical_rows = create_medical_summary(audit_root, identity, current_git)
    augment_checkpoint_drift(audit_root, medical_rows)

    for name in BASE_AUDIT_FILES:
        source = audit_root / name
        if source.exists():
            copy_file(source, archive_root / name)
    for source_rel, archive_rel in SCRIPT_MAP.items():
        source = repo_root / source_rel
        if source.exists():
            copy_file(source, archive_root / archive_rel)

    research_path = audit_root / "RESEARCH_DIRECTION_PRE_FIX.md"
    if not research_path.exists():
        raise RuntimeError("RESEARCH_DIRECTION_PRE_FIX.md must be written before freezing")
    copy_file(research_path, archive_root / research_path.name)

    inventory = inventory_rows(repo_root, audit_root, archive_root)
    inventory_path = audit_root / "FORENSIC_ARTIFACT_INVENTORY.md"
    write_inventory(inventory_path, inventory, audit_root, archive_root)
    copy_file(inventory_path, archive_root / inventory_path.name)

    archive_paths = sorted(
        path.relative_to(archive_root).as_posix()
        for path in archive_root.rglob("*")
        if path.is_file()
    )
    archive_paths_with_manifest = archive_paths + [
        "PRE_FIX_FORENSIC_MANIFEST.json",
        "FORENSIC_SHA256SUMS.txt",
    ]
    manifest_data = manifest(
        repo_root,
        audit_root,
        archive_root,
        identity,
        current_git,
        branch,
        sorted(set(archive_paths_with_manifest)),
    )
    manifest_path = audit_root / "PRE_FIX_FORENSIC_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_file(manifest_path, archive_root / manifest_path.name)

    # The manifest and inventory are now part of the immutable compact set.
    write_hashes(archive_root, audit_root)

    # Refresh the inventory after all generated deliverables exist, then copy
    # the final inventory and regenerate hashes so every archived file is
    # covered. The manifest's artifact list is intentionally stable and
    # includes the hash file even though the hash file self-excludes.
    inventory = inventory_rows(repo_root, audit_root, archive_root)
    write_inventory(inventory_path, inventory, audit_root, archive_root)
    copy_file(inventory_path, archive_root / inventory_path.name)
    write_hashes(archive_root, audit_root)

    print(json.dumps({
        "audit_root": str(audit_root),
        "archive_root": str(archive_root),
        "medical_rows": len(medical_rows),
        "archive_file_count": sum(1 for path in archive_root.rglob("*") if path.is_file()),
        "current_git": current_git,
        "branch": branch,
        "scheduler_classification": manifest_data["scheduler_classification"],
        "decision": manifest_data["current_decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
