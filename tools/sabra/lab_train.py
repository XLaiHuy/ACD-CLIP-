"""Portable lab launcher for the frozen 20e P1-v8.3 contract."""
from __future__ import annotations
import argparse
import errno
import hashlib
import json
import os
import pty
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "runs/phase5/sabra/LAB_20E_READY_V2"
RUNS_DEFAULT = ROOT.parent / "acdclip_lab_runs"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def replace_or_append(args: list[str], flag: str, value: str) -> list[str]:
    result = list(args)
    if flag in result:
        result[result.index(flag) + 1] = value
    else:
        result.extend([flag, value])
    return result

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def make_run_root(run_id: str, requested: Path | None) -> Path:
    if not run_id or any(char in run_id for char in "/\\"):
        raise ValueError("run-id must be a simple unique name")
    base = (requested or Path(os.environ.get("ACDCLIP_RUN_ROOT", str(RUNS_DEFAULT)))).expanduser().resolve()
    root = base / run_id
    if root.exists():
        raise FileExistsError(f"RUN_ID_COLLISION: {root}")
    for name in ("logs", "checkpoints", "validation", "final"):
        (root / name).mkdir(parents=True, exist_ok=False)
    return root


def resolve_resume_root(
    run_id: str,
    requested: Path | None,
    checkpoint: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate an existing run boundary and its epoch-boundary checkpoint.

    Unlike ``make_run_root``, this never creates, deletes, or rewrites the
    original run directory. It is limited to launcher metadata and checkpoint
    validation; model construction remains in train.py.
    """
    if not run_id or any(char in run_id for char in "/\\"):
        raise ValueError("run-id must be a simple unique name")
    base = (requested or Path(os.environ.get("ACDCLIP_RUN_ROOT", str(RUNS_DEFAULT)))).expanduser().resolve()
    root = base / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"RESUME_RUN_ROOT_MISSING: {root}")
    required_root_entries = (
        "RESOLVED_CONFIG.json", "GIT_PROVENANCE.json", "ENVIRONMENT.json",
        "DATASET_ROLE_CONTRACT.json", "ASSET_HASHES.json", "checkpoints",
    )
    missing_root_entries = [name for name in required_root_entries if not (root / name).exists()]
    if missing_root_entries:
        raise RuntimeError(f"RESUME_RUN_PROVENANCE_MISSING: {missing_root_entries}")

    checkpoint_path = checkpoint.expanduser().resolve()
    checkpoint_dir = (root / "checkpoints").resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"RESUME_CHECKPOINT_DIRECTORY_INVALID: {checkpoint_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RESUME_CHECKPOINT_MISSING: {checkpoint_path}")
    if checkpoint_path.parent != checkpoint_dir:
        raise ValueError(
            f"RESUME_CHECKPOINT_ROOT_MISMATCH: checkpoint={checkpoint_path} expected_parent={checkpoint_dir}"
        )

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_checkpoint_keys = (
        "epoch", "seed", "batch_size", "grad_accum_steps", "image_adapter",
        "text_adapter", "soft_prompt", "h6_state_dict", "optimizer_state",
        "scheduler_state", "python_random_state", "numpy_random_state",
        "torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state",
        "package_config_sha256", "dataset_role_contract_sha256", "phase2b_config",
    )
    missing_checkpoint_keys = [
        key for key in required_checkpoint_keys if key not in payload or payload[key] is None
    ]
    if missing_checkpoint_keys:
        raise RuntimeError(f"RESUME_CHECKPOINT_STATE_MISSING: {missing_checkpoint_keys}")

    args = [str(value) for value in config["training"]["canonical_cli_args"]]

    def option_value(flag: str) -> str:
        if flag not in args or args.index(flag) + 1 >= len(args):
            raise RuntimeError(f"RESUME_CONFIG_ARGUMENT_MISSING: {flag}")
        return args[args.index(flag) + 1]

    expected_batch = int(config["training"]["batch_size"])
    expected_accum = int(config["training"]["grad_accum_steps"])
    expected_seed = int(option_value("--seed"))
    expected_routing = option_value("--h6_prediction_routing")
    if int(payload["batch_size"]) != expected_batch or int(payload["grad_accum_steps"]) != expected_accum:
        raise RuntimeError("RESUME_BATCH_GEOMETRY_MISMATCH")
    if int(payload["seed"]) != expected_seed:
        raise RuntimeError("RESUME_SEED_MISMATCH")
    checkpoint_routing = payload["phase2b_config"].get("h6_prediction_routing")
    if checkpoint_routing != expected_routing:
        raise RuntimeError(
            f"RESUME_PREDICTION_ROUTING_MISMATCH: checkpoint={checkpoint_routing} config={expected_routing}"
        )

    package_hash = sha256_file(PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    dataset_role_hash = sha256_file(PACKAGE / "DATASET_ROLE_CONTRACT.json")
    if payload["package_config_sha256"] != package_hash:
        raise RuntimeError("RESUME_PACKAGE_CONFIG_HASH_MISMATCH")
    if payload["dataset_role_contract_sha256"] != dataset_role_hash:
        raise RuntimeError("RESUME_DATASET_ROLE_HASH_MISMATCH")
    original_provenance = json.loads((root / "GIT_PROVENANCE.json").read_text(encoding="utf-8"))
    if original_provenance.get("package_config_sha256") != package_hash:
        raise RuntimeError("RESUME_ORIGINAL_PROVENANCE_CONFIG_HASH_MISMATCH")
    if original_provenance.get("dataset_role_contract_sha256") != dataset_role_hash:
        raise RuntimeError("RESUME_ORIGINAL_PROVENANCE_DATASET_HASH_MISMATCH")

    epoch = int(payload["epoch"])
    scheduler_state = payload["scheduler_state"]
    if int(scheduler_state.get("last_epoch", -1)) != epoch:
        raise RuntimeError("RESUME_SCHEDULER_EPOCH_MISMATCH")
    optimizer_groups = payload["optimizer_state"].get("param_groups", [])
    if not optimizer_groups:
        raise RuntimeError("RESUME_OPTIMIZER_GROUPS_MISSING")
    if epoch >= int(option_value("--epoch")):
        raise RuntimeError("RESUME_CHECKPOINT_ALREADY_COMPLETE")
    return {
        "root": root,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_payload": payload,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "batch_size": expected_batch,
        "grad_accum_steps": expected_accum,
        "seed": expected_seed,
        "prediction_routing": expected_routing,
        "expected_primary_lr": float(optimizer_groups[0]["lr"]),
        "scheduler_summary": {
            "last_epoch": int(scheduler_state["last_epoch"]),
            "step_size": scheduler_state.get("step_size"),
            "gamma": scheduler_state.get("gamma"),
            "last_lr": scheduler_state.get("_last_lr"),
        },
        "original_training_git_sha": payload.get("git_sha") or original_provenance.get("git_sha"),
        "package_config_sha256": package_hash,
        "dataset_role_contract_sha256": dataset_role_hash,
    }


def resume_provenance_path(root: Path, epoch: int) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        root / "resume" / f"RESUME_EPOCH{epoch}_{stamp}.json",
        root / "logs" / f"resume_from_epoch_{epoch}_{stamp}.log",
    )


def record_resume_provenance(
    resume_info: dict[str, Any],
    *,
    config_path: Path,
    command: list[str],
    provenance_path: Path,
    log_path: Path,
) -> None:
    train_diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--", "train.py"], cwd=ROOT)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    train_blob = subprocess.check_output(["git", "hash-object", "train.py"], cwd=ROOT, text=True).strip()
    payload = {
        "resumed_from_epoch": resume_info["epoch"],
        "next_epoch": resume_info["next_epoch"],
        "checkpoint": str(resume_info["checkpoint"]),
        "checkpoint_sha256": resume_info["checkpoint_sha256"],
        "original_training_git_sha": resume_info["original_training_git_sha"],
        "current_git_head": current_head,
        "corrective_train_blob_sha": train_blob,
        "corrective_train_diff_sha256": hashlib.sha256(train_diff).hexdigest(),
        "correction_classification": "CONTROL_FLOW_ONLY_GATE_FIX",
        "scientific_formula_change": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "package_config_sha256": resume_info["package_config_sha256"],
        "dataset_role_contract_sha256": resume_info["dataset_role_contract_sha256"],
        "scheduler": resume_info["scheduler_summary"],
        "expected_primary_lr_next_epoch": resume_info["expected_primary_lr"],
        "batch_size": resume_info["batch_size"],
        "grad_accum_steps": resume_info["grad_accum_steps"],
        "seed": resume_info["seed"],
        "prediction_routing": resume_info["prediction_routing"],
        "command": command,
        "log_path": str(log_path),
    }
    # The resume directory may contain earlier additive records. Never
    # overwrite one: only the parent may pre-exist.
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    if provenance_path.exists():
        raise FileExistsError(f"RESUME_PROVENANCE_COLLISION: {provenance_path}")
    write_json(provenance_path, payload)


def prepare_run_root(root: Path, config: dict[str, Any], run_id: str) -> None:
    write_json(root / "RESOLVED_CONFIG.json", {**config, "run_id": run_id, "run_root": str(root), "resolved_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(root / "GIT_PROVENANCE.json", {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "package_config_sha256": sha256_file(PACKAGE / "TRAIN20E_FINAL_CONFIG.json"),
        "dataset_role_contract_sha256": sha256_file(PACKAGE / "DATASET_ROLE_CONTRACT.json"),
        "asset_hashes_sha256": sha256_file(PACKAGE / "ASSET_HASHES.json"),
        "training_started": False,
    })
    shutil.copy2(PACKAGE / "DATASET_ROLE_CONTRACT.json", root / "DATASET_ROLE_CONTRACT.json")
    shutil.copy2(PACKAGE / "ASSET_HASHES.json", root / "ASSET_HASHES.json")
    write_json(root / "ENVIRONMENT.json", {"python": sys.version, "cuda_device": os.environ.get("CUDA_DEVICE", "0"), "medical_reads": 0})

def repository_env() -> dict[str, str]:
    """Make subprocess imports deterministic from a clean lab shell."""
    env = os.environ.copy()
    entries = [str(ROOT), str(ROOT / "tools")]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _write_live_output(data: bytes) -> None:
    """Forward PTY bytes without changing tqdm's carriage-return rendering."""
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(data)
        stream.flush()
        return
    sys.stdout.write(data.decode("utf-8", errors="replace"))
    sys.stdout.flush()


def run_process(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    live_progress: bool = False,
) -> int:
    """Run train.py, optionally teeing its PTY byte stream to terminal and log."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        if not live_progress:
            return int(subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode)

        master_fd, slave_fd = pty.openpty()
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=None,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
        )
        os.close(slave_fd)
        interrupted = False
        try:
            while True:
                try:
                    data = os.read(master_fd, 64 * 1024)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not data:
                    break
                log.buffer.write(data)
                log.flush()
                _write_live_output(data)

        except KeyboardInterrupt:
            interrupted = True
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        finally:
            os.close(master_fd)

        child.wait()
        return 130 if interrupted else int(child.returncode)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["training"]["dataset"] != "VisA":
        raise RuntimeError("final training contract must use VisA")
    return config

def apply_training_overrides(
    config: dict[str, Any],
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
) -> dict[str, Any]:
    """Apply explicit runtime geometry without mutating the frozen package config."""
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if grad_accum_steps is not None and grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    training = config["training"]
    resolved_batch = batch_size if batch_size is not None else int(training["batch_size"])
    resolved_accum = grad_accum_steps if grad_accum_steps is not None else int(training["grad_accum_steps"])
    training["batch_size"] = resolved_batch
    training["grad_accum_steps"] = resolved_accum
    training["effective_batch_size"] = resolved_batch * resolved_accum
    args = [str(value) for value in training["canonical_cli_args"]]
    args = replace_or_append(args, "--batch_size", str(resolved_batch))
    args = replace_or_append(args, "--grad_accum_steps", str(resolved_accum))
    training["canonical_cli_args"] = args
    training["runtime_overrides"] = {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
    }
    return config

def train(
    config_path: Path,
    run_id: str,
    run_root: Path | None,
    resume: Path | None = None,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
    live_progress: bool = False,
) -> int:
    if not os.environ.get("VISA_ROOT"):
        raise RuntimeError("VISA_ROOT is required; training never discovers MVTec or Medical")
    config = apply_training_overrides(load_config(config_path), batch_size, grad_accum_steps)
    resume_info = None
    if resume is None:
        root = make_run_root(run_id, run_root)
        prepare_run_root(root, config, run_id)
    else:
        resume_info = resolve_resume_root(run_id, run_root, resume, config)
        root = resume_info["root"]
    args = [str(value) for value in config["training"]["canonical_cli_args"]]
    args = replace_or_append(args, "--save_path", str(root / "checkpoints"))
    if resume_info is not None:
        args = replace_or_append(args, "--resume", str(resume_info["checkpoint"]))
    env = repository_env()
    env["ACDCLIP_PACKAGE_CONFIG_SHA256"] = sha256_file(PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    env["ACDCLIP_DATASET_ROLE_SHA256"] = sha256_file(PACKAGE / "DATASET_ROLE_CONTRACT.json")
    env.setdefault("CUDA_DEVICE", "0")
    env.setdefault("NUM_WORKERS", "0")
    command = [sys.executable, "train.py", *args]
    if resume_info is None:
        log_path = root / "logs" / "train_wrapper.log"
    else:
        provenance_path, log_path = resume_provenance_path(root, resume_info["epoch"])
        record_resume_provenance(
            resume_info,
            config_path=config_path,
            command=command,
            provenance_path=provenance_path,
            log_path=log_path,
        )
    return run_process(command, log_path, env, live_progress=live_progress)

def preflight(
    config_path: Path,
    output_root: Path | None,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
) -> int:
    config = apply_training_overrides(load_config(config_path), batch_size, grad_accum_steps)
    root = (output_root or Path(tempfile.mkdtemp(prefix="acdclip_lab_precheck_"))).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"preflight output must be new and empty: {root}")
    (root / "checkpoints").mkdir(parents=True, exist_ok=False)
    args = [str(value) for value in config["training"]["canonical_cli_args"]]
    args = replace_or_append(args, "--save_path", str(root / "checkpoints"))
    args = replace_or_append(args, "--epoch", "1")
    args = replace_or_append(args, "--h6_smoke_max_batches", "1")
    env = repository_env()
    env["ACDCLIP_PACKAGE_CONFIG_SHA256"] = sha256_file(PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    env["ACDCLIP_DATASET_ROLE_SHA256"] = sha256_file(PACKAGE / "DATASET_ROLE_CONTRACT.json")
    env.setdefault("CUDA_DEVICE", "0")
    env.setdefault("NUM_WORKERS", "0")
    code = run_process([sys.executable, "train.py", *args], root / "preflight.log", env)
    checkpoint_read = False
    checkpoint_fields: list[str] = []
    checkpoint_path = root / "checkpoints" / "adapter_1.pth"
    if code == 0:
        import torch
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_fields = sorted(loaded)
        required = {
            "epoch", "global_step", "image_adapter", "text_adapter", "soft_prompt",
            "h6_state_dict", "optimizer_state", "scheduler_state", "amp_scaler_state",
            "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
            "torch_cuda_rng_state_all", "dataloader_generator_state", "phase2b_config",
            "h6_config", "git_sha", "package_config_sha256",
            "dataset_role_contract_sha256",
        }
        missing = sorted(required - set(loaded))
        if missing:
            raise RuntimeError(f"preflight checkpoint contract missing fields: {missing}")
        checkpoint_read = True
    write_json(root / "PRECHECK_RESULT.json", {"returncode": code, "bounded_one_batch": True, "checkpoint_read": checkpoint_read, "checkpoint_fields": checkpoint_fields, "training_started": False, "medical_reads": 0, "output_root": str(root)})
    return code

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--run-root", type=Path, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--grad-accum-steps", type=int, default=None)
    train_parser.add_argument("--live-progress", action="store_true", help="stream existing train.py tqdm through a PTY while logging it")
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--config", type=Path, default=PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--run-root", type=Path, default=None)
    resume_parser.add_argument("--batch-size", type=int, default=None)
    resume_parser.add_argument("--grad-accum-steps", type=int, default=None)
    resume_parser.add_argument("--live-progress", action="store_true", help="stream existing train.py tqdm through a PTY while logging it")
    pre_parser = sub.add_parser("preflight")
    pre_parser.add_argument("--config", type=Path, default=PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    pre_parser.add_argument("--output-root", type=Path, default=None)
    pre_parser.add_argument("--batch-size", type=int, default=None)
    pre_parser.add_argument("--grad-accum-steps", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "train":
        raise SystemExit(train(args.config, args.run_id, args.run_root, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, live_progress=args.live_progress))
    if args.mode == "resume":
        raise SystemExit(train(args.config, args.run_id, args.run_root, resume=args.checkpoint, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, live_progress=args.live_progress))
    raise SystemExit(preflight(args.config, args.output_root, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps))

if __name__ == "__main__":
    main()
