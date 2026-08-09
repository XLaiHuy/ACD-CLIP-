#!/usr/bin/env python3
"""Fail-closed audit for the frozen P1-v8.3 epoch-20 medical checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from model.checkpoint_utils import validate_p1_v83_checkpoint_contract
from test import combine_image_score, image_auc_ap_or_none


CANONICAL_MEDICAL_DATASETS = (
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
)


def validate_final_checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    validate_p1_v83_checkpoint_contract(checkpoint)
    h6_config = checkpoint["h6_config"]
    loss_weights = checkpoint.get("loss_weights", {})
    expected = {
        "epoch": (checkpoint.get("epoch"), 20),
        "n_groups": (checkpoint.get("n_groups"), 3),
        "dfg_mode": (checkpoint.get("dfg_mode"), "attn"),
        "dfg_attn_dim": (checkpoint.get("dfg_attn_dim"), 256),
        "dfg_attn_tau": (checkpoint.get("dfg_attn_tau"), 8.0),
        "use_ss2d_dfg": (checkpoint.get("use_ss2d_dfg"), True),
        "dfg_ss2d_fusion": (checkpoint.get("dfg_ss2d_fusion"), "weight_residual"),
        "dfg_beta": (checkpoint.get("dfg_beta"), 0.10),
        "dfg_beta_schedule": (checkpoint.get("dfg_beta_schedule"), "warmup010"),
        "dfg_beta_target": (checkpoint.get("dfg_beta_target"), 0.10),
        "h6_n_groups": (h6_config.get("n_groups"), 3),
        "factor_effective_beta": (h6_config.get("utility_factor_effective_beta"), 0.999),
        "router_support_normalized": (
            h6_config.get("utility_router_support_normalized"), True
        ),
        "pcgrad_main_factor": (h6_config.get("pcgrad_main_factor"), True),
        "lambda_factor": (h6_config.get("utility_factor_weight"), 0.03),
        "lambda_router": (h6_config.get("utility_router_weight"), 0.10),
    }
    mismatches = {
        name: {"actual": actual, "required": required}
        for name, (actual, required) in expected.items()
        if actual != required
    }
    disabled_weights = (
        "balance", "center", "orth", "functional_factor_diversity",
        "router_teacher", "cluster_loss_weight",
    )
    for name in disabled_weights:
        actual = loss_weights.get(name)
        if actual is None or float(actual) != 0.0:
            mismatches[f"loss_weights.{name}"] = {"actual": actual, "required": 0.0}
    rho = checkpoint.get("gate_values", {}).get("rho")
    if rho is None or len(rho) != 3 or any(abs(float(value) - 0.05) > 1e-8 for value in rho):
        mismatches["gate_values.rho"] = {"actual": rho, "required": [0.05, 0.05, 0.05]}
    if mismatches:
        raise ValueError(f"P1-v8.3 final checkpoint mismatch: {mismatches}")

    cls = torch.tensor([0.2, 0.8])
    pmax = torch.tensor([0.6, 0.4])
    expected_score = torch.tensor([0.4, 0.6])
    if not torch.equal(combine_image_score(cls, pmax, "Medical"), expected_score):
        raise ValueError("medical image score is not exactly 0.5 * cls + 0.5 * pmax")
    if image_auc_ap_or_none(torch.ones(2, dtype=torch.int64), cls) != (None, None):
        raise ValueError("one-class image metrics must be reported as N/A")

    return {
        "status": "PASS",
        "checkpoint_epoch": 20,
        "datasets": list(CANONICAL_MEDICAL_DATASETS),
        "dataset_count": len(CANONICAL_MEDICAL_DATASETS),
        "medical_split": "test",
        "pixel_stride": 1,
        "external_exact_pixel_metrics": True,
        "one_class_image_metrics": "N/A",
        "image_score": "0.5 * cls + 0.5 * pmax",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"canonical epoch-20 checkpoint missing: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    report = validate_final_checkpoint_payload(checkpoint)
    report["checkpoint"] = str(args.checkpoint.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
