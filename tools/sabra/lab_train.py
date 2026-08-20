"""Portable lab launcher for the frozen 20e P1-v8.3 contract."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "runs/phase5/sabra/LAB_20E_READY"
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

def run_process(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        return int(subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode)

def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["training"]["dataset"] != "VisA":
        raise RuntimeError("final training contract must use VisA")
    return config

def train(config_path: Path, run_id: str, run_root: Path | None, resume: Path | None = None) -> int:
    if not os.environ.get("VISA_ROOT"):
        raise RuntimeError("VISA_ROOT is required; training never discovers MVTec or Medical")
    config = load_config(config_path)
    root = make_run_root(run_id, run_root)
    prepare_run_root(root, config, run_id)
    args = [str(value) for value in config["training"]["canonical_cli_args"]]
    args = replace_or_append(args, "--save_path", str(root / "checkpoints"))
    if resume is not None:
        args = replace_or_append(args, "--resume", str(resume))
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE", "0")
    env.setdefault("NUM_WORKERS", "0")
    return run_process([sys.executable, "train.py", *args], root / "logs" / "train_wrapper.log", env)

def preflight(config_path: Path, output_root: Path | None) -> int:
    config = load_config(config_path)
    root = (output_root or Path(tempfile.mkdtemp(prefix="acdclip_lab_precheck_"))).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"preflight output must be new and empty: {root}")
    (root / "checkpoints").mkdir(parents=True, exist_ok=False)
    args = [str(value) for value in config["training"]["canonical_cli_args"]]
    args = replace_or_append(args, "--save_path", str(root / "checkpoints"))
    args = replace_or_append(args, "--epoch", "1")
    args = replace_or_append(args, "--h6_smoke_max_batches", "1")
    env = os.environ.copy()
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
        required = {"epoch", "global_step", "optimizer_state", "scheduler_state", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "dataloader_generator_state"}
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
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--config", type=Path, default=PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--run-root", type=Path, default=None)
    pre_parser = sub.add_parser("preflight")
    pre_parser.add_argument("--config", type=Path, default=PACKAGE / "TRAIN20E_FINAL_CONFIG.json")
    pre_parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "train":
        raise SystemExit(train(args.config, args.run_id, args.run_root))
    if args.mode == "resume":
        raise SystemExit(train(args.config, args.run_id, args.run_root, resume=args.checkpoint))
    raise SystemExit(preflight(args.config, args.output_root))

if __name__ == "__main__":
    main()
