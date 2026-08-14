#!/usr/bin/env python3
"""Integrity audit for the published Phase4-V V1 bounded evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    output = ROOT / "runs/phase4v/v1_5/V1_5_PREFLIGHT.json"
    evidence = ROOT / "runs/phase4v/short64_diagnostic/ABLATION_METRICS.json"
    manifest = ROOT / "runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"
    config = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
    data = json.loads(evidence.read_text())
    checkpoints = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
        for path in (ROOT / "runs/phase4v").rglob("*")
        if path.suffix in {".pth", ".pt", ".ckpt", ".bin"}
    ]
    metric_definitions = {
        "mean_relative_correction": "mean correction/base norm on final deterministic evaluation images",
        "max_relative_correction": "maximum correction/base norm among active training microbatches",
        "resolution": "cross-population statistics; max is not required to exceed the evaluation mean. Labels are clarified here without changing historical V1 evidence.",
    }
    report = {
        "decision": "V1_DIAGNOSTIC_REPAIRED",
        "remote_head_expected": "6c2ce038562826d40800e33576987ae7579bde47",
        "local_head": git("rev-parse", "HEAD"),
        "source_ancestor_actual": git("rev-parse", "a12cd36"),
        "source_ancestor_in_instruction": "a12cd36816cfc6479515b7f567b4f4f668e2ecb2",
        "evidence": {"path": str(evidence.relative_to(ROOT)), "sha256": sha(evidence)},
        "config": {"path": str(config.relative_to(ROOT)), "sha256": sha(config), "seed": 0, "precision": "fp32"},
        "audit_manifest": {"path": str(manifest.relative_to(ROOT)), "sha256": sha(manifest), "samples": len(json.loads(manifest.read_text())["samples"])},
        "activation_schedule": "8 epochs x 8 microbatches; Phase4-V off through epoch 7; 8 active microbatches in epoch 8",
        "metric_definition_repair": metric_definitions,
        "checkpoint_inventory": checkpoints,
        "same_checkpoint_attribution_possible": bool(checkpoints),
        "reason_same_checkpoint_not_run": None if checkpoints else "V1 runner intentionally saved compact metrics only; no V1 variant checkpoint exists locally.",
        "v1_variants": {name: {key: values[key] for key in ("normal_bce", "anomaly_bce", "pixel_ap_macro", "pixel_auc_macro", "mean_relative_correction", "max_relative_correction", "all_finite")} for name, values in data.items()},
        "authorized_next": "Branch E: one fresh shared OpenAI-CLIP warmup paired BASE/current-V1 run with 32 active microbatches, retaining checkpoints for ON/OFF/ZERO_DELTA attribution.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("decision", "same_checkpoint_attribution_possible", "reason_same_checkpoint_not_run", "metric_definition_repair")}, indent=2))


if __name__ == "__main__":
    main()
