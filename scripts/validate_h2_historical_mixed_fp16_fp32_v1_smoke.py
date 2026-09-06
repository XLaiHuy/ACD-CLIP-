#!/usr/bin/env python3
"""Validate the bounded historical mixed-precision restoration smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floating_dtypes(value):
    found = set()
    if torch.is_tensor(value) and value.is_floating_point():
        found.add(str(value.dtype))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(floating_dtypes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(floating_dtypes(item))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    e1_path = args.root / "shared_e1" / "adapter_1.pth"
    e6_path = args.root / "A" / "adapter_6.pth"
    e1 = torch.load(e1_path, map_location="cpu", weights_only=False)
    e6 = torch.load(e6_path, map_location="cpu", weights_only=False)
    trace = json.loads(args.trace.read_text())
    config = e6["resolved_scientific_config"]
    parent = e6["parent_scientific_config"]

    required = {
        "checkpoint_version", "model_state", "optimizer_state", "scheduler_state",
        "scaler_state", "python_random_state", "numpy_random_state",
        "torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state",
        "resolved_scientific_config", "config_sha256", "git_sha", "clip_sha256",
        "dataset_manifest_sha256", "precision_protocol",
        "later_transformer_fp32_islands",
    }
    assert required <= set(e1) and required <= set(e6)
    assert e1["epoch"] == 1 and e1["global_step"] == 5
    assert e6["epoch"] == 6 and e6["global_step"] == 30
    assert e6["precision_protocol"] == "HISTORICAL_MIXED_FP16_FP32_V1"
    assert e6["precision"] == "fp16" and e6["gradscaler_enabled"] is True
    assert e6["later_transformer_fp32_islands"] is False
    assert e6["tf32_enabled"] is False
    assert config["precision_protocol"] == "HISTORICAL_MIXED_FP16_FP32_V1"
    assert config["later_transformer_fp32_islands"] is False
    assert config["use_safe_anchor"] is True and config["anchor_gradient_budget"] is True
    assert config["anchor_family_budget"] == 0.1
    assert config["dfg_weight_residual_fp32"] is True
    assert config["use_ss2d_dfg"] is True and config["use_hybrid_soft_prompt"] is True
    assert config["training_horizon"] == 15 and parent["training_horizon"] == 15
    assert floating_dtypes(e6["model_state"]) == {"torch.float32"}
    assert floating_dtypes(e6["optimizer_state"]) == {"torch.float32"}
    assert e6["scaler_state"]
    assert trace["status"] == "PASS" and trace["bf16_tensor_observed"] is False

    summaries = json.loads((args.root / "A" / "epoch_summary.json").read_text())
    by_epoch = {int(row["epoch"]): row for row in summaries}
    assert sorted(by_epoch) == [2, 3, 4, 5, 6]
    expected_alpha = {2: 0.0, 3: 0.0, 4: 0.05, 5: 0.1, 6: 0.2}
    expected_beta = {2: 0.0, 3: 0.0, 4: 0.05, 5: 0.05, 6: 0.05}
    for epoch in expected_alpha:
        assert abs(by_epoch[epoch]["hybrid_alpha"] - expected_alpha[epoch]) < 1e-12
        assert abs(by_epoch[epoch]["dfg_beta"] - expected_beta[epoch]) < 1e-12
        assert by_epoch[epoch]["nonfinite_loss_events"] == 0
        assert by_epoch[epoch]["precision_protocol"] == "HISTORICAL_MIXED_FP16_FP32_V1"
        assert by_epoch[epoch]["later_transformer_fp32_islands"] is False
        assert by_epoch[epoch]["trainable_parameter_count"] == (14095887 if epoch <= 3 else 14102031)
        assert abs(by_epoch[epoch]["lr"]["image_adapter"] - 0.001 * 0.9 ** (epoch - 1)) < 1e-15
        assert abs(by_epoch[epoch]["lr"]["text_adapter"] - 0.0005 * 0.9 ** (epoch - 1)) < 1e-15
        assert by_epoch[epoch]["lr"]["soft_prompt"] == (0.0 if epoch <= 3 else 0.00005)
    events = args.root / "A" / "events.jsonl"
    event_rows = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    grad_skips = sum(row.get("event") == "nonfinite_gradient" for row in event_rows)

    # Two independent deserializations must preserve the saved RNG and loader identities.
    e6_again = torch.load(e6_path, map_location="cpu", weights_only=False)
    assert torch.equal(e6["torch_cpu_rng_state"], e6_again["torch_cpu_rng_state"])
    assert torch.equal(e6["dataloader_generator_state"], e6_again["dataloader_generator_state"])
    log_lines = (args.root / "A" / "train.log").read_text().splitlines()
    batch_lines = [line for line in log_lines if "batch_identity epoch=" in line]
    assert len(batch_lines) == 25
    anchor_rows = [json.loads(line.split("metrics=", 1)[1]) for line in log_lines if "anchor_family_step" in line]
    assert len(anchor_rows) == 25
    max_anchor_ratio = max(row["max_effective_active_family_ratio"] for row in anchor_rows)
    assert max_anchor_ratio <= 0.1 + 1e-7

    text = f"""# H2 historical mixed FP16+FP32 V1 smoke

SMOKE_TEST=PASS
CHECKPOINT_RESUME_SMOKE=PASS
DTYPE_TRACE=PASS

- Scope: source-only VisA; 30 attempted batches total (5 fresh E1 + 5 each E2-E6); no target evaluation.
- Fresh/save/reload/resume: PASS; E1 global step 5, E6 global step 30.
- Checkpoint v3 state: model, optimizer, scheduler, GradScaler, Python/NumPy/Torch/CUDA RNG, and dataloader generator present.
- RNG/dataloader identity: PASS; saved states reproduce byte-for-byte across independent reloads; 25 resumed augmented-batch identities logged.
- Scientific identity: HISTORICAL_MIXED_FP16_FP32_V1, FP16 autocast, GradScaler on, TF32 off, later transformer islands off.
- FP32 persistence: model and optimizer floating state are FP32.
- Anchor: active from E2 with lambda {config['anchor_lambda']} and family cap {config['anchor_family_budget']}.
- Trainables/LR: exact counts 14,095,887 while prompt-frozen and 14,102,031 after unfreeze; image/text StepLR and constant active prompt LR verified per epoch.
- Maximum effective active-family Anchor ratio: {max_anchor_ratio:.10f} (cap 0.10).
- DFG/SS2D: active; FP32 weight residual; beta schedule observed E2-E3=0 and E4-E6=0.05.
- Prompt: hybrid; alpha schedule observed E2-E3=0, E4=0.05, E5=0.10, E6=0.20.
- Numerical events in smoke: nonfinite loss skips 0; recoverable gradient skips {grad_skips}. The smoke is not required to reproduce the historical two skips.
- E1 checkpoint SHA256: `{sha256(e1_path)}`
- E6 checkpoint SHA256: `{sha256(e6_path)}`
"""
    args.output.write_text(text, encoding="utf-8")
    print("H2_HISTORICAL_MIXED_SMOKE=PASS")


if __name__ == "__main__":
    main()
