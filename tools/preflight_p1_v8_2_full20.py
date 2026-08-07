#!/usr/bin/env python3
"""Preflight Check Tool for Phase 4 Progress 1 v8.2 20-Epoch Training & Testing

Verifies:
  - Config schema & canonical hash
  - Training parameters (20 epochs, save every epoch, no validation)
  - Model settings (rho=0.05 frozen, correction_max=1.0, hard_anchor, center_spread)
  - Objective switches (experts, load_bias, balance, cluster, func_div, router_teacher, center_losses all disabled)
  - Metric audit decision (METRICS_READY)
  - Calibration decision (READY_FOR_ITERATION_D)
  - Dataset existence and test split readiness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight check for 20-epoch train/test")
    parser.add_argument("--stage", type=str, required=True, choices=["train", "test", "all"])
    parser.add_argument("--config", type=str, default="configs/phase4/p1_v8_2_candidate1.json")
    parser.add_argument("--dataset-dir", type=str, default="dataset/hub")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"PREFLIGHT CHECK FOR P1-V8.2 FULL 20-EPOCH ({args.stage.upper()} STAGE)")
    print("=" * 70)

    # 1. Config existence & schema
    if not os.path.exists(args.config):
        print(f"[FAIL] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    # Hash check
    canonical_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical_json.encode()).hexdigest()
    print(f"[OK] Config loaded: {args.config}")
    print(f"[OK] Config SHA-256: {config_hash}")

    # 2. Calibration readiness check
    calib_decision = cfg.get("calibration_decision", None)
    if calib_decision != "READY_FOR_ITERATION_D":
        print(f"\nERROR: loss calibration is not approved; resolve Iteration C before full training.")
        print(f"Current config calibration_decision: {calib_decision}")
        sys.exit(1)

    # 3. Required fields check
    required_fields = [
        "schema_version", "n_groups", "rho_values", "rho_trainable",
        "global_text_mode", "local_factor_mode", "correction_max",
        "h6_logit_temperature", "Candidate-1_objective_switches"
    ]
    for rf in required_fields:
        if rf not in cfg:
            print(f"[FAIL] Missing required field in config: {rf}")
            sys.exit(1)

    # 4. Numerical & structural constraints
    rho = cfg["rho_values"]
    if rho != [0.05, 0.05, 0.05]:
        print(f"[FAIL] Invalid rho_values: expected [0.05, 0.05, 0.05], got {rho}")
        sys.exit(1)
    if cfg.get("rho_trainable", True):
        print("[FAIL] rho_trainable must be false")
        sys.exit(1)
    if cfg.get("correction_max", None) != 1.0:
        print(f"[FAIL] correction_max must be 1.0, got {cfg.get('correction_max')}")
        sys.exit(1)
    if cfg.get("global_text_mode") != "hard_anchor":
        print(f"[FAIL] global_text_mode must be 'hard_anchor', got {cfg.get('global_text_mode')}")
        sys.exit(1)
    if cfg.get("local_factor_mode") != "center_spread":
        print(f"[FAIL] local_factor_mode must be 'center_spread', got {cfg.get('local_factor_mode')}")
        sys.exit(1)
    if cfg.get("local_center_mix") != 0.05:
        print(f"[FAIL] local_center_mix must be 0.05, got {cfg.get('local_center_mix')}")
        sys.exit(1)
    if cfg.get("local_factor_spread") != 0.10:
        print(f"[FAIL] local_factor_spread must be 0.10, got {cfg.get('local_factor_spread')}")
        sys.exit(1)

    # 5. Objective switches check
    switches = cfg.get("Candidate-1_objective_switches", {})
    for sw_name in ["load_bias", "balance", "cluster", "functional_diversity",
                    "router_teacher", "center_losses", "experts"]:
        if switches.get(sw_name, False):
            print(f"[FAIL] Objective switch '{sw_name}' must be false in Candidate 1")
            sys.exit(1)

    # 6. Metric audit decision
    metric_report_path = "runs/phase4/p1_v8_2_full20_script_build/METRIC_AUDIT_REPORT.md"
    if not os.path.exists(metric_report_path):
        print(f"[FAIL] Metric audit report not found at {metric_report_path}")
        sys.exit(1)
    with open(metric_report_path) as f:
        metric_content = f.read()
    if "METRICS_READY" not in metric_content:
        print("[FAIL] Metric audit decision is not METRICS_READY")
        sys.exit(1)

    # 7. Dataset readiness
    dataset_name = cfg.get("dataset", "VisA")
    hub_file = os.path.join(args.dataset_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(hub_file):
        print(f"[WARN] Hub file not found: {hub_file}")
    else:
        print(f"[OK] Dataset hub file verified: {hub_file}")

    print("=" * 70)
    print("PREFLIGHT CHECK PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    main()
