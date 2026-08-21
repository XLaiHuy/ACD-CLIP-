#!/usr/bin/env python3
"""Resumable, test-only medical validation sweep for LAB-20E H6 checkpoints.

This launcher intentionally contains no model or metric implementation.  Each
job delegates inference and all scientific scoring to ``test.py --checkpoint``;
that evaluator restores the checkpoint's saved Phase4/H6 configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pty
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from model.checkpoint_loader import checkpoint_identity
from model.checkpoint_utils import validate_p1_v83_checkpoint_contract


METHOD_ID = "SABRA_H6_N4_EXPLORATORY"
EXPECTED_SOURCE_SHA = "643d2f0335f7bd8c3475656a39c549034b4c00b8"
DEFAULT_RUN_ROOT = Path("/home/ai4/caohuy/acdclip_runs/lab20e_b1_accum6_seed0")
DEFAULT_MEDICAL_ROOT = Path("/home/ai4/caohuy/data")
DEFAULT_MANIFEST_ROOT = Path("/home/ai4/caohuy/acdclip_runs/protocol/medical_seed0_val030")
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "validation/h6_n4_exploratory_medical_val_even_epochs"
DEFAULT_PYTHON = Path("/home/ai4/caohuy/ACD-CLIP-lab20e/.venv-gpu128/bin/python")
DEFAULT_EPOCHS = (10, 12, 14, 16, 18, 20)

PIXEL_DATASETS = (
    ("Colon_colonDB", "ColonDB"),
    ("Colon_clinicDB", "ClinicDB"),
    ("Colon_Kvasir", "Kvasir"),
    ("Brain", "BrainMRI"),
    ("Liver", "Liver CT"),
    ("Retina", "Retina OCT"),
)
IMAGE_DATASET_KEYS = ("Brain", "Liver", "Retina")
DATA_ROOTS = {
    "Brain": ("MedAD", "Brain_AD", "valid"),
    "Liver": ("MedAD", "Liver_AD", "valid"),
    "Retina": ("MedAD", "Retina_RESC_AD", "val"),
    "Colon_clinicDB": ("Colon", "CVC-ClinicDB"),
    "Colon_colonDB": ("Colon", "CVC-ColonDB"),
    "Colon_Kvasir": ("Colon", "Kvasir"),
}
DISPLAY_BY_KEY = dict(PIXEL_DATASETS)
ACTIVE_CHILD: subprocess.Popen[bytes] | None = None
ACTIVE_JOB: "Job" | None = None


@dataclass(frozen=True)
class CheckpointInfo:
    epoch: int
    path: Path
    sha256: str
    h6_version: str
    routing: str
    rho: Any
    img_size: int
    n_groups: int
    dfg_mode: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ManifestInfo:
    dataset_key: str
    display: str
    path: Path
    sha256: str
    n_images: int
    n_normal: int
    n_anomaly: int


@dataclass(frozen=True)
class Job:
    checkpoint: CheckpointInfo
    manifest: ManifestInfo

    @property
    def epoch_label(self) -> str:
        return f"E{self.checkpoint.epoch}"

    @property
    def directory_name(self) -> Path:
        return Path("jobs") / self.epoch_label / self.manifest.display.replace(" ", "_")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"SOURCE_SHA_MISMATCH: git rev-parse HEAD failed: {completed.stderr.strip()}")
    actual = completed.stdout.strip()
    if actual != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"SOURCE_SHA_MISMATCH: expected={EXPECTED_SOURCE_SHA} actual={actual}")
    return actual


def method_warning() -> str:
    return (
        "============================================================\n"
        "EXPERIMENTAL METHOD:\n"
        "Current checkpoints contain H6 state (N=4).\n"
        "Results are exploratory and are NOT the clean Phase2B-only\n"
        "SABRA primary experiment.\n"
        "============================================================"
    )


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.run_root = args.run_root.expanduser().resolve()
    args.medical_root = args.medical_root.expanduser().resolve()
    args.medical_manifest_root = args.medical_manifest_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.epochs = tuple(args.checkpoint_epochs)
    if tuple(args.epochs) != tuple(sorted(set(args.epochs))):
        raise ValueError("--checkpoint-epochs must be unique and ascending")
    if not args.epochs:
        raise ValueError("--checkpoint-epochs must not be empty")
    if tuple(args.epochs) != DEFAULT_EPOCHS:
        raise ValueError(f"this fixed sweep requires exactly these epochs: {DEFAULT_EPOCHS}")
    if args.output_root == args.run_root or args.output_root == args.run_root / "checkpoints":
        raise ValueError("--output-root must not be the run root or checkpoint root")
    return args


def checkpoint_preflight(checkpoint_root: Path, epochs: tuple[int, ...]) -> list[CheckpointInfo]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint root is missing: {checkpoint_root}")
    infos: list[CheckpointInfo] = []
    for epoch in epochs:
        path = checkpoint_root / f"adapter_{epoch}.pth"
        identity = checkpoint_identity(path, expected_epoch=epoch, require_final_contract=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "h6_state_dict" not in checkpoint or not isinstance(checkpoint["h6_state_dict"], dict):
            raise ValueError(f"{path}: H6 state is required but missing")
        h6 = checkpoint.get("h6_config")
        if not isinstance(h6, dict):
            raise ValueError(f"{path}: h6_config is required but missing")
        if int(checkpoint.get("phase4_progress", h6.get("progress", 0))) != 1:
            raise ValueError(f"{path}: expected h6_progress=1")
        if int(h6.get("num_factors", -1)) != 4:
            raise ValueError(f"{path}: expected h6_num_factors=4, got {h6.get('num_factors')!r}")
        validate_p1_v83_checkpoint_contract(checkpoint)
        base = checkpoint.get("phase2b_config") or {}
        gate_values = checkpoint.get("gate_values") or {}
        infos.append(CheckpointInfo(
            epoch=epoch,
            path=path.resolve(),
            sha256=str(identity["checkpoint_sha256"]),
            h6_version=str(h6.get("progress_version")),
            routing=str(h6.get("prediction_routing")),
            rho=gate_values.get("rho", h6.get("rho")),
            img_size=int(checkpoint.get("img_size", base.get("img_size", -1))),
            n_groups=int(checkpoint.get("n_groups", base.get("n_groups", -1))),
            dfg_mode=str(checkpoint.get("dfg_mode", base.get("dfg_mode", ""))),
            config=h6,
        ))
    return infos


def read_manifest(path: Path, dataset_key: str, display: str, medical_root: Path) -> ManifestInfo:
    rows: list[dict[str, Any]] = []
    data_root = medical_root.joinpath(*DATA_ROOTS[dataset_key])
    if not data_root.is_dir():
        raise FileNotFoundError(f"{dataset_key}: validation data root is missing: {data_root}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("class_name") != dataset_key:
                raise ValueError(f"{path}:{line_number}: class_name does not match {dataset_key}")
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number}: invalid binary label {label!r}")
            image_path = data_root / str(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: missing validation image {image_path}")
            if label:
                mask_path = data_root / str(row.get("mask_path", ""))
                if not mask_path.is_file():
                    raise FileNotFoundError(f"{path}:{line_number}: missing validation mask {mask_path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: validation manifest is empty")
    return ManifestInfo(
        dataset_key=dataset_key,
        display=display,
        path=path.resolve(),
        sha256=sha256_file(path),
        n_images=len(rows),
        n_normal=sum(int(row["label"]) == 0 for row in rows),
        n_anomaly=sum(int(row["label"]) == 1 for row in rows),
    )


def manifest_preflight(manifest_root: Path, medical_root: Path) -> dict[str, ManifestInfo]:
    if not medical_root.is_dir():
        raise FileNotFoundError(f"medical root is missing: {medical_root}")
    if not manifest_root.is_dir():
        raise FileNotFoundError(f"medical manifest root is missing: {manifest_root}")
    protocol = manifest_root / "medical_protocol_manifest.json"
    if not protocol.is_file():
        raise FileNotFoundError(f"frozen protocol manifest is missing: {protocol}")
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    if protocol_payload.get("protocol") != "phase4_medical_val_test_v1" or protocol_payload.get("seed") != 0:
        raise ValueError(f"unexpected frozen medical protocol: {protocol}")
    manifests: dict[str, ManifestInfo] = {}
    for key, display in PIXEL_DATASETS:
        path = manifest_root / f"{key}_val.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"required validation manifest is missing: {path}")
        manifests[key] = read_manifest(path, key, display, medical_root)
    return manifests


def verify_cuda(python: Path, device: int) -> None:
    if device < 0:
        raise ValueError("--device must be non-negative")
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        f"assert {device} < torch.cuda.device_count(), "
        "f'cuda device out of range: requested=" + str(device) + " count={torch.cuda.device_count()}'; "
        f"print(torch.cuda.get_device_name({device}))"
    )
    completed = subprocess.run([str(python), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise RuntimeError(f"CUDA_PREFLIGHT_FAILED: {completed.stderr.strip() or completed.stdout.strip()}")
    print(f"CUDA device {device}: {completed.stdout.strip()}")


def ensure_output_writable(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / ".write_probe.tmp"
    with probe.open("w", encoding="utf-8") as handle:
        handle.write("ok\n")
        handle.flush()
        os.fsync(handle.fileno())
    probe.unlink()


def job_fingerprint(job: Job, source: str) -> str:
    return canonical_fingerprint({
        "method_id": METHOD_ID,
        "source_sha": source,
        "epoch": job.checkpoint.epoch,
        "checkpoint": str(job.checkpoint.path),
        "checkpoint_sha256": job.checkpoint.sha256,
        "h6_num_factors": 4,
        "dataset_key": job.manifest.dataset_key,
        "dataset_display": job.manifest.display,
        "medical_split": "val",
        "manifest_path": str(job.manifest.path),
        "manifest_sha256": job.manifest.sha256,
        "exact_pixel_metrics": True,
        "pixel_stride": 1,
        "batch_size": 1,
        "evaluator": "test.py --checkpoint (checkpoint-restored configuration)",
    })


def result_matches(path: Path, job: Job, source: str) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all((
        result.get("status") == "COMPLETE",
        result.get("method_id") == METHOD_ID,
        result.get("source_sha") == source,
        result.get("epoch") == job.checkpoint.epoch,
        result.get("checkpoint") == str(job.checkpoint.path),
        result.get("checkpoint_sha256") == job.checkpoint.sha256,
        result.get("dataset_key") == job.manifest.dataset_key,
        result.get("dataset_display") == job.manifest.display,
        result.get("split") == "val",
        result.get("manifest_sha256") == job.manifest.sha256,
        result.get("evaluation_config_fingerprint") == job_fingerprint(job, source),
        result.get("exact_pixel_metrics") is True,
        result.get("pixel_stride") == 1,
        result.get("h6_num_factors") == 4,
    ))


def evaluator_command(args: argparse.Namespace, job: Job, repo_root: Path, job_dir: Path) -> list[str]:
    # Do not pass model/H6/DFG/prompt flags: test.py restores all of them from
    # the explicit checkpoint, which is the authoritative configuration.
    return [
        str(args.python), str(repo_root / "test.py"),
        "--checkpoint", str(job.checkpoint.path),
        "--dataset", job.manifest.dataset_key,
        "--medical_split", "val",
        "--medical_manifest_root", str(args.medical_manifest_root),
        "--save_path", str(job_dir),
        "--batch_size", "1",
        "--cuda_device", str(args.device),
        "--num_workers", "0",
        "--external_exact_pixel_metrics",
        "--pixel_stride", "1",
    ]


def terminate_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_child_live(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    global ACTIVE_CHILD
    master_fd, slave_fd = pty.openpty()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
        log.flush()
        child = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=slave_fd, stderr=slave_fd, start_new_session=True,
        )
        ACTIVE_CHILD = child
        os.close(slave_fd)
        try:
            while True:
                ready, _, _ = select.select([master_fd], [], [], 0.25)
                if ready:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        data = b""
                    if data:
                        log.write(data)
                        log.flush()
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    elif child.poll() is not None:
                        break
                if child.poll() is not None:
                    while True:
                        try:
                            data = os.read(master_fd, 65536)
                        except OSError:
                            data = b""
                        if not data:
                            break
                        log.write(data)
                        sys.stdout.buffer.write(data)
                    log.flush()
                    sys.stdout.buffer.flush()
                    break
            return int(child.wait())
        except KeyboardInterrupt:
            terminate_child(child)
            raise
        finally:
            ACTIVE_CHILD = None
            os.close(master_fd)


def numeric_or_none(value: str | None) -> float | None:
    if value is None or value.strip().upper() in {"", "N/A", "NA"}:
        return None
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"metric is outside percentage scale [0,100]: {number}")
    return number


def parse_evaluator_csv(job_dir: Path, job: Job) -> dict[str, float | None]:
    csv_path = job_dir / f"exact_results_{job.manifest.dataset_key}_val_epoch_{job.checkpoint.epoch}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"evaluator did not write expected result CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    average = next((row for row in reversed(rows) if row.get("class name") == "Average"), None)
    if average is None:
        raise ValueError(f"result CSV lacks Average row: {csv_path}")
    return {
        "pixel_auroc": numeric_or_none(average.get("pixel AUC")),
        "pixel_ap": numeric_or_none(average.get("pixel AP")),
        "image_auroc": numeric_or_none(average.get("image AUC")),
        "image_ap": numeric_or_none(average.get("image AP")),
    }


def write_job_provenance(job_dir: Path, job: Job, source: str, command: list[str]) -> None:
    atomic_json(job_dir / "provenance.json", {
        "method_id": METHOD_ID,
        "warning": "CURRENT CHECKPOINTS CONTAIN H6 STATE. THESE RESULTS ARE EXPLORATORY AND ARE NOT THE CLEAN PHASE2B-ONLY SABRA PRIMARY EXPERIMENT.",
        "source_sha": source,
        "checkpoint": str(job.checkpoint.path),
        "checkpoint_sha256": job.checkpoint.sha256,
        "epoch": job.checkpoint.epoch,
        "h6_num_factors": 4,
        "h6_version": job.checkpoint.h6_version,
        "dataset_key": job.manifest.dataset_key,
        "dataset_display": job.manifest.display,
        "split": "val",
        "manifest_path": str(job.manifest.path),
        "manifest_sha256": job.manifest.sha256,
        "exact_pixel_metrics": True,
        "pixel_stride": 1,
        "command": command,
    })


def execute_job(args: argparse.Namespace, repo_root: Path, source: str, job: Job) -> str:
    global ACTIVE_JOB
    job_dir = args.output_root / job.directory_name
    result_path = job_dir / "RESULT.json"
    if result_path.is_file() and result_matches(result_path, job, source):
        return "cached"
    job_dir.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(args, job, repo_root, job_dir)
    write_job_provenance(job_dir, job, source, command)
    ACTIVE_JOB = job
    started = time.monotonic()
    env = os.environ.copy()
    env["MEDICAL_ROOT"] = str(args.medical_root)
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        return_code = run_child_live(command, repo_root, env, job_dir / "evaluation.log")
    except KeyboardInterrupt:
        atomic_json(job_dir / "FAILED.json", {
            "status": "INTERRUPTED", "epoch": job.checkpoint.epoch,
            "dataset_key": job.manifest.dataset_key, "return_code": 130,
            "ended_at_utc": utc_now(), "method_id": METHOD_ID,
        })
        raise
    finally:
        ACTIVE_JOB = None
    runtime_sec = time.monotonic() - started
    if return_code != 0:
        atomic_json(job_dir / "FAILED.json", {
            "status": "FAILED", "method_id": METHOD_ID, "epoch": job.checkpoint.epoch,
            "dataset_key": job.manifest.dataset_key, "dataset_display": job.manifest.display,
            "return_code": return_code, "runtime_sec": runtime_sec, "ended_at_utc": utc_now(),
        })
        return "failed"
    try:
        metrics = parse_evaluator_csv(job_dir, job)
        payload = {
            "status": "COMPLETE", "method_id": METHOD_ID, "source_sha": source,
            "epoch": job.checkpoint.epoch, "checkpoint": str(job.checkpoint.path),
            "checkpoint_sha256": job.checkpoint.sha256, "h6_num_factors": 4,
            "dataset_key": job.manifest.dataset_key, "dataset_display": job.manifest.display,
            "split": "val", "manifest_path": str(job.manifest.path),
            "manifest_sha256": job.manifest.sha256,
            "evaluation_config_fingerprint": job_fingerprint(job, source),
            "exact_pixel_metrics": True, "pixel_stride": 1,
            "n_images": job.manifest.n_images, "n_normal": job.manifest.n_normal,
            "n_anomaly": job.manifest.n_anomaly, "runtime_sec": runtime_sec,
            "completed_at_utc": utc_now(), **metrics,
        }
        atomic_json(result_path, payload)
        (job_dir / "FAILED.json").unlink(missing_ok=True)
        return "done"
    except Exception as error:
        atomic_json(job_dir / "FAILED.json", {
            "status": "FAILED_RESULT_PARSE", "method_id": METHOD_ID,
            "epoch": job.checkpoint.epoch, "dataset_key": job.manifest.dataset_key,
            "return_code": 0, "error": str(error), "ended_at_utc": utc_now(),
        })
        return "failed"


def collect_results(args: argparse.Namespace, jobs: list[Job], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        path = args.output_root / job.directory_name / "RESULT.json"
        if path.is_file() and result_matches(path, job, source):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda row: (int(row["epoch"]), PIXEL_DATASETS.index((row["dataset_key"], row["dataset_display"]))))


def cell(auc: float | None, ap: float | None) -> str:
    return "NA" if auc is None or ap is None else f"({auc:.1f}, {ap:.1f})"


def ranks(values: list[float | None]) -> dict[int, str]:
    distinct = sorted({value for value in values if value is not None}, reverse=True)
    labels: dict[int, str] = {}
    for index, value in enumerate(values):
        if value is None:
            continue
        if distinct and value == distinct[0]:
            labels[index] = "blue"
        elif len(distinct) > 1 and value == distinct[1]:
            labels[index] = "red"
    return labels


def latex_cell(auc: float | None, ap: float | None, auc_rank: str | None, ap_rank: str | None) -> str:
    if auc is None or ap is None:
        return "NA"
    a = f"{auc:.1f}" if auc_rank is None else f"\\textcolor{{{auc_rank}}}{{{auc:.1f}}}"
    p = f"{ap:.1f}" if ap_rank is None else f"\\textcolor{{{ap_rank}}}{{{ap:.1f}}}"
    return f"({a}, {p})"


def write_table_files(args: argparse.Namespace, rows: list[dict[str, Any]], epochs: tuple[int, ...], keys: tuple[str, ...], name: str, metric_label: str) -> None:
    by_key_epoch = {(row["dataset_key"], int(row["epoch"])): row for row in rows}
    csv_rows: list[dict[str, str]] = []
    markdown = [
        f"# Medical {metric_label} validation table",
        "", f"**METHOD_ID:** `{METHOD_ID}`  ",
        "**Warning:** Current checkpoints contain H6 state. These results are exploratory and are not the clean Phase2B-only SABRA primary experiment.",
        "", "| Domain (Metric) | Dataset | " + " | ".join(f"E{epoch}" for epoch in epochs) + " |",
        "|---|---|" + "|".join("---" for _ in epochs) + "|",
    ]
    latex = [
        "% METHOD_ID = SABRA_H6_N4_EXPLORATORY",
        "% CURRENT CHECKPOINTS CONTAIN H6 STATE. EXPLORATORY ONLY; NOT CLEAN PHASE2B-ONLY SABRA.",
        "% Requires \\usepackage{xcolor}",
        "\\begin{tabular}{ll" + "c" * len(epochs) + "}",
        "\\toprule",
        "Domain (Metric) & Dataset & " + " & ".join(f"E{epoch}" for epoch in epochs) + " \\\\",
        "\\midrule",
    ]
    for index, key in enumerate(keys):
        display = DISPLAY_BY_KEY[key]
        values = [by_key_epoch.get((key, epoch)) for epoch in epochs]
        aucs = [None if value is None else value.get("pixel_auroc" if metric_label == "Pixel-level" else "image_auroc") for value in values]
        aps = [None if value is None else value.get("pixel_ap" if metric_label == "Pixel-level" else "image_ap") for value in values]
        cells = [cell(auc, ap) for auc, ap in zip(aucs, aps)]
        csv_rows.append({"Domain (Metric)": f"Medical ({metric_label})" if index == 0 else "", "Dataset": display, **{f"E{epoch}": value for epoch, value in zip(epochs, cells)}})
        markdown.append(f"| {'Medical (' + metric_label + ')' if index == 0 else ''} | {display} | " + " | ".join(cells) + " |")
        auc_ranks, ap_ranks = ranks(aucs), ranks(aps)
        latex_cells = [latex_cell(auc, ap, auc_ranks.get(i), ap_ranks.get(i)) for i, (auc, ap) in enumerate(zip(aucs, aps))]
        latex.append(f"{'Medical (' + metric_label + ')' if index == 0 else ''} & {display} & " + " & ".join(latex_cells) + " \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    headers = ["Domain (Metric)", "Dataset", *[f"E{epoch}" for epoch in epochs]]
    atomic_csv(args.output_root / f"TABLE_MEDICAL_{name}.csv", headers, csv_rows)
    (args.output_root / f"TABLE_MEDICAL_{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (args.output_root / f"TABLE_MEDICAL_{name}.tex").write_text("\n".join(latex), encoding="utf-8")


def aggregate(args: argparse.Namespace, jobs: list[Job], source: str) -> None:
    rows = collect_results(args, jobs, source)
    long_headers = ["method_id", "source_sha", "epoch", "checkpoint", "checkpoint_sha256", "h6_num_factors", "dataset_key", "dataset_display", "split", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "n_images", "n_normal", "n_anomaly", "runtime_sec", "status"]
    atomic_csv(args.output_root / "RESULTS_LONG.csv", long_headers, rows)
    epochs = tuple(args.epochs)
    write_table_files(args, rows, epochs, tuple(key for key, _ in PIXEL_DATASETS), "PIXEL", "Pixel-level")
    write_table_files(args, rows, epochs, IMAGE_DATASET_KEYS, "IMAGE", "Image-level")
    by_epoch = {epoch: [row for row in rows if int(row["epoch"]) == epoch] for epoch in epochs}
    summary: list[dict[str, Any]] = []
    for epoch in epochs:
        lookup = {row["dataset_key"]: row for row in by_epoch[epoch]}
        def average_metric(dataset_keys: tuple[str, ...], metric: str) -> float | None:
            values = [lookup[key].get(metric) for key in dataset_keys if lookup.get(key, {}).get(metric) is not None]
            return None if len(values) != len(dataset_keys) else sum(float(value) for value in values) / len(values)
        summary.append({
            "method_id": METHOD_ID, "source_sha": source, "epoch": f"E{epoch}",
            "macro_pixel_auroc_6": average_metric(tuple(key for key, _ in PIXEL_DATASETS), "pixel_auroc"),
            "macro_pixel_ap_6": average_metric(tuple(key for key, _ in PIXEL_DATASETS), "pixel_ap"),
            "macro_image_auroc_3": average_metric(IMAGE_DATASET_KEYS, "image_auroc"),
            "macro_image_ap_3": average_metric(IMAGE_DATASET_KEYS, "image_ap"),
        })
    atomic_csv(args.output_root / "EPOCH_SUMMARY.csv", list(summary[0]), summary)
    atomic_json(args.output_root / "AGGREGATE_PROVENANCE.json", {
        "method_id": METHOD_ID,
        "warning": "CURRENT CHECKPOINTS CONTAIN H6 STATE. THESE RESULTS ARE EXPLORATORY AND ARE NOT THE CLEAN PHASE2B-ONLY SABRA PRIMARY EXPERIMENT.",
        "source_sha": source, "split": "val", "exact_pixel_metrics": True, "pixel_stride": 1,
        "epochs": list(epochs), "pixel_datasets": [dict(key=key, display=display) for key, display in PIXEL_DATASETS],
        "image_datasets": list(IMAGE_DATASET_KEYS), "generated_at_utc": utc_now(),
    })


def print_checkpoint_summary(infos: list[CheckpointInfo]) -> None:
    for info in infos:
        print(
            f"E{info.epoch}: sha256={info.sha256} epoch={info.epoch} h6_present=true "
            f"h6_num_factors=4 h6_version={info.h6_version} routing={info.routing} "
            f"rho={info.rho} img_size={info.img_size} n_groups={info.n_groups} dfg_mode={info.dfg_mode}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    parser.add_argument("--medical-root", type=Path, default=DEFAULT_MEDICAL_ROOT)
    parser.add_argument("--medical-manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    global ACTIVE_CHILD, ACTIVE_JOB
    args = resolve_args(build_parser().parse_args(argv))
    print(method_warning())
    repo_root = Path(__file__).resolve().parents[2]
    source = source_sha(repo_root)
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"required Python interpreter is not executable: {args.python}")
    checkpoints = checkpoint_preflight(args.run_root / "checkpoints", args.epochs)
    manifests = manifest_preflight(args.medical_manifest_root, args.medical_root)
    ensure_output_writable(args.output_root)
    verify_cuda(args.python, args.device)
    print_checkpoint_summary(checkpoints)
    jobs = [Job(checkpoint, manifests[key]) for checkpoint in checkpoints for key, _ in PIXEL_DATASETS]
    atomic_json(args.output_root / "SWEEP_PROVENANCE.json", {
        "method_id": METHOD_ID,
        "warning": "CURRENT CHECKPOINTS CONTAIN H6 STATE. THESE RESULTS ARE EXPLORATORY AND ARE NOT THE CLEAN PHASE2B-ONLY SABRA PRIMARY EXPERIMENT.",
        "source_sha": source, "run_root": str(args.run_root), "checkpoint_root": str(args.run_root / "checkpoints"),
        "medical_root": str(args.medical_root), "medical_manifest_root": str(args.medical_manifest_root),
        "split": "val", "exact_pixel_metrics": True, "pixel_stride": 1, "batch_size": 1,
        "epochs": list(args.epochs), "total_jobs": len(jobs), "created_at_utc": utc_now(),
    })
    done = cached = failed = 0
    try:
        with tqdm(total=len(jobs), desc="SABRA H6-N4 Medical val", unit="job", dynamic_ncols=True) as bar:
            for job in jobs:
                state = execute_job(args, repo_root, source, job)
                done += state == "done"
                cached += state == "cached"
                failed += state == "failed"
                bar.update(1)
                bar.set_postfix(epoch=job.epoch_label, dataset=job.manifest.display, done=done, cached=cached, failed=failed)
    except KeyboardInterrupt:
        if ACTIVE_CHILD is not None:
            terminate_child(ACTIVE_CHILD)
        if ACTIVE_JOB is not None:
            job_dir = args.output_root / ACTIVE_JOB.directory_name
            atomic_json(job_dir / "FAILED.json", {
                "status": "INTERRUPTED", "method_id": METHOD_ID, "epoch": ACTIVE_JOB.checkpoint.epoch,
                "dataset_key": ACTIVE_JOB.manifest.dataset_key, "return_code": 130, "ended_at_utc": utc_now(),
            })
        aggregate(args, jobs, source)
        print("Interrupted safely; completed jobs remain cacheable and the partial job is not COMPLETE.", file=sys.stderr)
        return 130
    aggregate(args, jobs, source)
    if failed:
        failed_jobs = [
            f"E{job.checkpoint.epoch}×{job.manifest.display}"
            for job in jobs
            if (args.output_root / job.directory_name / "FAILED.json").is_file()
        ]
        print("FAILED JOBS: " + ", ".join(failed_jobs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
