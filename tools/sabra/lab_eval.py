"""Lab-facing checkpoint-aware post-training evaluation interface.

The evaluator is deliberately separate from training. Every post-training
command requires an explicit checkpoint, and the Medical subcommand requires
an explicit authorization flag plus the existing run-local manifest contract.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from model.checkpoint_loader import checkpoint_identity  # noqa: E402

MEDICAL_DATASETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")


def repository_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(ROOT), str(ROOT / "tools")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if extra:
        env.update(extra)
    return env


def ensure_new_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"evaluation output collision: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_command(command: list[str], output_root: Path, env: dict[str, str], *, precreate_output: bool = True) -> int:
    if precreate_output:
        write_json(output_root / "EVALUATION_PROVENANCE.json", {
        "command": command,
        "cwd": str(ROOT),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "medical_reads": 0,
        })
    if precreate_output:
        log = output_root / "evaluation.log"
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"evaluation output collision: {output_root}")
        log = output_root.parent / f"{output_root.name}.launcher.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def require_final_checkpoint(path: Path) -> dict[str, Any]:
    return checkpoint_identity(path, expected_epoch=20, require_final_contract=True)


def run_visa(args: argparse.Namespace) -> int:
    identity = require_final_checkpoint(args.checkpoint)
    root = ensure_new_output(args.output_root)
    write_json(root / "CHECKPOINT_IDENTITY.json", identity)
    command = [
        sys.executable, str(ROOT / "test.py"),
        "--dataset", "VisA",
        "--checkpoint", str(Path(identity["checkpoint_path"])),
        "--save_path", str(root),
        "--external_exact_pixel_metrics",
    ]
    env = repository_env({"VISA_ROOT": str(args.data_root.expanduser().resolve())})
    return run_command(command, root, env)


def run_mvtec(args: argparse.Namespace) -> int:
    identity = require_final_checkpoint(args.checkpoint)
    root = args.output_root.expanduser().resolve()
    command = [
        sys.executable, str(ROOT / "tools/sabra/trust_v2/mvtec_external.py"),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--checkpoint", str(Path(identity["checkpoint_path"])),
        "--output-root", str(root),
        "--backend", "fast",
    ]
    code = run_command(
        command,
        root,
        repository_env({"MVTEC_ROOT": str(args.data_root.expanduser().resolve())}),
        precreate_output=False,
    )
    if code == 0:
        write_json(root / "EVALUATION_PROVENANCE.json", {
            "command": command, "checkpoint_identity": identity, "medical_reads": 0,
            "evaluation_role": "POST_TRAINING_PREVIOUSLY_OBSERVED_EXTERNAL_BENCHMARK",
        })
    return code


def run_medical(args: argparse.Namespace) -> int:
    if not args.allow_medical_evaluation:
        raise RuntimeError("MEDICAL_SEALED: --allow-medical-evaluation is required after final checkpoint freeze")
    data_root = args.data_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    identity = require_final_checkpoint(args.checkpoint)
    root = ensure_new_output(args.output_root)
    write_json(root / "CHECKPOINT_IDENTITY.json", identity)
    aggregate = {
        "state": "AUTHORIZED_EXECUTION_REQUESTED",
        "checkpoint_identity": identity,
        "dataset_root": str(data_root),
        "manifest_root": str(manifest_root),
        "datasets": list(MEDICAL_DATASETS),
        "medical_reads": 0,
        "evaluation_only": True,
        "checkpoint_selection": "disabled; adapter_20 is fixed before evaluation",
    }
    write_json(root / "MEDICAL_EVALUATION_PROVENANCE.json", aggregate)
    for dataset in MEDICAL_DATASETS:
        dataset_root = root / dataset
        dataset_root.mkdir(parents=False, exist_ok=False)
        command = [
            sys.executable, str(ROOT / "test.py"),
            "--dataset", dataset,
            "--medical_split", "test",
            "--medical_manifest_root", str(manifest_root),
            "--checkpoint", str(Path(identity["checkpoint_path"])),
            "--save_path", str(dataset_root),
            "--external_exact_pixel_metrics",
        ]
        code = run_command(command, dataset_root, repository_env({"MEDICAL_ROOT": str(data_root)}))
        if code:
            raise SystemExit(code)
    aggregate["state"] = "MEDICAL_EVALUATION_COMPLETED"
    aggregate["medical_reads"] = "authorized evaluation reads only"
    write_json(root / "MEDICAL_EVALUATION_PROVENANCE.json", aggregate)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="mode", required=True)
    visa = sub.add_parser("visa", help="evaluate the fixed adapter_20 checkpoint on VisA")
    visa.add_argument("--checkpoint", type=Path, required=True)
    visa.add_argument("--data-root", type=Path, required=True)
    visa.add_argument("--output-root", type=Path, required=True)
    visa.set_defaults(handler=run_visa)
    mvtec = sub.add_parser("mvtec", help="run the post-training frozen MVTec benchmark")
    mvtec.add_argument("--checkpoint", type=Path, required=True)
    mvtec.add_argument("--data-root", type=Path, required=True)
    mvtec.add_argument("--output-root", type=Path, required=True)
    mvtec.set_defaults(handler=run_mvtec)
    medical = sub.add_parser("medical", help="run the sealed final Medical evaluation")
    medical.add_argument("--checkpoint", type=Path, required=True)
    medical.add_argument("--data-root", type=Path, required=True)
    medical.add_argument("--manifest-root", type=Path, required=True)
    medical.add_argument("--output-root", type=Path, required=True)
    medical.add_argument("--allow-medical-evaluation", action="store_true")
    medical.set_defaults(handler=run_medical)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
