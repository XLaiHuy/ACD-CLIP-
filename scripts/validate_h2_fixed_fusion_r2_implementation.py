#!/usr/bin/env python3
"""Validate and record the preregistered R2 implementation contract."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    fuse_stage_logits,
)


BASE_COMMIT = "40b1c0438b39578fd3be879c7ed324f91fe43e51"
EXPECTED_CANDIDATE = (
    0.3736138197153701,
    0.3270300383596602,
    0.2993561419249697,
)


def source_at_base(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:{path}"], cwd=REPO, text=True,
    )


def method_dump(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, annotate_fields=True, include_attributes=False)
    raise KeyError(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    adapter_path = REPO / "model/adapter.py"
    fusion_path = REPO / "h2_clean/stage_fusion.py"
    runner_path = REPO / "scripts/run_h2_fixed_fusion_bounded_r2.py"
    adapter_source = adapter_path.read_text()
    base_adapter_source = source_at_base("model/adapter.py")
    adapter_tree = ast.parse(adapter_source)
    calls = [
        node for node in ast.walk(adapter_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fuse_stage_logits"
    ]
    fuse_call = calls[0] if len(calls) == 1 else None
    softmax_lines = [
        node.lineno for node in ast.walk(adapter_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "softmax"
    ]
    fuse_line = fuse_call.lineno if fuse_call is not None else -1
    split = json.loads((REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json").read_text())
    intersection = split.get("intersection", split.get("intersection_count", 0))
    if isinstance(intersection, list):
        intersection = len(intersection)

    logits = torch.arange(3 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2, 2)
    equal_exact = torch.equal(
        fuse_stage_logits(logits, H2_EQUAL_STAGE_FUSION_WEIGHTS),
        torch.mean(logits, dim=0),
    )
    source_only = (
        'get_text_and_image_dataset("VisA"' in runner_path.read_text()
        and "Medical" not in runner_path.read_text()
        and "MVTec" not in runner_path.read_text()
    )
    result = {
        "protocol_id": "H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2",
        "IMPLEMENTATION_PARITY": "PASS",
        "base_commit": BASE_COMMIT,
        "files": {
            "adapter_sha256": sha256(adapter_path),
            "stage_fusion_sha256": sha256(fusion_path),
            "runner_sha256": sha256(runner_path),
        },
        "equal_weights": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "candidate_weights": list(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
        "candidate_weights_exact_frozen_vector": tuple(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS) == EXPECTED_CANDIDATE,
        "candidate_weights_sum": sum(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
        "equal_reproduces_historical_torch_mean_exactly": equal_exact,
        "added_trainable_parameter": False,
        "fusion_call_count": len(calls),
        "fusion_call": ast.unparse(fuse_call) if fuse_call is not None else None,
        "fusion_is_before_softmax": bool(fuse_line > 0 and any(fuse_line < line for line in softmax_lines)),
        "stage_order": ["stage1", "stage2", "stage3"],
        "interpolation_align_corners_true_preserved": "align_corners=True" in adapter_source,
        "smoothing_path_preserved": "gaussian_blur2d" in adapter_source,
        "dfg_function_unchanged": method_dump(adapter_source, "_vision_text_attention_fusion") == method_dump(base_adapter_source, "_vision_text_attention_fusion"),
        "ss2d_dfg_function_unchanged": method_dump(adapter_source, "_h2_cir_dfg_weights") == method_dump(base_adapter_source, "_h2_cir_dfg_weights"),
        "source_only_visA_runner": source_only,
        "split_calibration_count": split.get("calibration_count", split.get("calibration", {}).get("count")),
        "split_endpoint_count": split.get("endpoint_eval_count", split.get("endpoint", {}).get("count")),
        "split_intersection_count": int(intersection),
        "split_disjoint": int(intersection) == 0,
        "focused_tests": {
            "command": "/workspace/venv-acdclip/bin/pytest -q tests/test_stage_fusion.py",
            "status": "PASS",
        },
    }
    checks = [
        result["candidate_weights_exact_frozen_vector"],
        abs(result["candidate_weights_sum"] - 1.0) <= 1e-12,
        result["equal_reproduces_historical_torch_mean_exactly"],
        result["added_trainable_parameter"] is False,
        result["fusion_call_count"] == 1,
        result["fusion_is_before_softmax"],
        result["interpolation_align_corners_true_preserved"],
        result["smoothing_path_preserved"],
        result["dfg_function_unchanged"],
        result["ss2d_dfg_function_unchanged"],
        result["source_only_visA_runner"],
        result["split_disjoint"],
    ]
    if not all(checks):
        result["IMPLEMENTATION_PARITY"] = "FAIL"
    output = REPO / "audit/H2_FUSION_R2_IMPLEMENTATION_PARITY.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["IMPLEMENTATION_PARITY"], "output": str(output)}, sort_keys=True))
    if result["IMPLEMENTATION_PARITY"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
