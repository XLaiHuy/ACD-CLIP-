#!/usr/bin/env python3
"""V0 runtime interface audit for factorized semantic-spatial visual adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import (
    DeterministicVisATrainDataset,
    _build_model,
    _git_sha,
    _sha256,
)
from utils import configure_canonical_fp32, get_phase2b_global_text_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4v/V0_DESIGN.json"))
    args = parser.parse_args()
    run_dir, manifest_path, output_path = args.run_dir.resolve(), args.manifest.resolve(), args.output.resolve()
    config_path, checkpoint_path = run_dir / "config.json", run_dir / "adapter_1.pth"
    config, checkpoint = json.loads(config_path.read_text()), torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    if config.get("h6_progress_version") != "P4-CSF-K1" or not torch.cuda.is_available():
        raise RuntimeError("V0 requires the published K1 interface checkpoint and CUDA.")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    model = _build_model(config, checkpoint, device)
    model.requires_grad_(False)
    core = model.h6.conditional_semantic_core
    if core is None:
        raise RuntimeError("ConditionalSemanticCore is unavailable.")
    core.requires_grad_(True)
    loader = DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=2, shuffle=False, num_workers=0)
    raw = next(iter(loader))
    image = raw["image"].to(device=device, dtype=torch.float32)
    class_names = list(raw["class_name"])
    with torch.no_grad():
        visual = model(image, return_phase4_features=True)
        base_text = get_phase2b_global_text_features(
            model, "VisA", class_names, device, use_hybrid_soft_prompt=True, use_soft_prompt=False
        ).float()
        _, base_group_logits, _ = model.vision_text_fusion_gate_seg(
            torch.stack(visual["seg_tokens"], dim=0), base_text, img_size=config["img_size"], return_details=True
        )
    core_out = core(
        visual["seg_tokens_pre_l2"], visual["cls24"], model.soft_prompt.ctx_normal, model.soft_prompt.ctx_abnormal
    )
    semantic_code = (core_out["prototype_abnormal"] - core_out["prototype_normal"]).squeeze(1)
    where = torch.softmax(base_group_logits.float(), dim=-1)[..., 1].detach()
    source_paths = [
        REPO_ROOT / "model/adapter.py", REPO_ROOT / "model/h6/conditional_semantics.py",
        REPO_ROOT / "model/h6/model.py", Path(__file__).resolve(),
    ]
    report = {
        "decision": "PHASE4V_INTERFACE_DESIGN_PASS",
        "provenance": {
            "repo_sha": _git_sha(), "script_sha256": _sha256(Path(__file__).resolve()),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
            "k1_checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)), "k1_checkpoint_sha256": _sha256(checkpoint_path),
            "config_sha256": _sha256(config_path), "manifest_sha256": _sha256(manifest_path),
            "precision": "strict FP32; TF32 off; AMP off", "main_model_optimizer_steps": 0,
        },
        "runtime_probe": {"batch_size": len(class_names), "class_names": class_names},
        "where": {
            "tensor": "stopgrad(softmax(base_group_logits, dim=-1)[..., abnormal])",
            "source": "ACDCLIP.vision_text_fusion_gate_seg on original seg_tokens with unchanged Phase2B text and DFG",
            "shape": list(where.shape), "range": [float(where.min()), float(where.max())],
            "requires_grad": bool(where.requires_grad), "role": "per-patch local anomaly probability; WHERE only",
        },
        "what": {
            "tensor": "prototype_abnormal - prototype_normal",
            "source": "ConditionalSemanticCore over multi-level seg_tokens_pre_l2 with separate normal/abnormal prototype queries",
            "shape": list(semantic_code.shape), "requires_grad": bool(semantic_code.requires_grad),
            "role": "one image-level CoPS normal-vs-abnormal state contrast; no base posterior or patch gate enters this code",
        },
        "how": {
            "tensor": "delta_v = W_up(GELU((1 + gamma(c_I))*W_down(v) + beta(c_I)))",
            "input_shape": list(visual["seg_tokens"][0].shape), "planned_bottleneck": 64,
            "controller_output_shape": [len(class_names), 2, 64], "lambda_fixed": 0.05,
            "role": "one shared lightweight visual residual; no attention, patch classifier, Router, or experts",
        },
        "predictor_contract": {
            "base_text_shape": list(base_text.shape), "base_group_logits_shape": list(base_group_logits.shape),
            "unchanged": "Phase2B text features, DFG weights/logic, normal-vs-abnormal predictor, and loss remain unchanged; only its visual-token input may become v_prime after activation.",
            "forbidden": ["dynamic text logits", "A0/A1 text selection", "Router", "Factor Bank", "responsibility", "ACT", "OT"],
        },
        "exact_noop": {
            "equation": "correction=lambda*g*delta_v; v_prime = v exactly if Phase4V is off or correction is exactly zero, else normalize(v+correction)",
            "implementation_rule": "Use an explicit identity branch/torch.where for exact-zero correction; never renormalize v in a no-op case, so lambda=0, g=0, and delta_v=0 reproduce the bit-identical Phase2B feature.",
            "conditions": ["Phase4V_OFF -> v_prime=v", "lambda=0 -> v_prime=v", "g=0 -> v_prime=v", "delta_v=0 -> v_prime=v"],
        },
        "gradient_ownership": {
            "main_task_to_phase2b": "allowed through unchanged predictor and v_prime",
            "main_task_to_conditional_core": "allowed after activation through c_I -> FiLM controller -> delta_v",
            "main_task_to_visual_adapter": "allowed after activation through delta_v",
            "spatial_gate": "detached before multiplication; Phase4V cannot alter g",
            "dynamic_text": "not constructed or consumed by the Phase4V path",
            "legacy_router_factor_act": "frozen/not executed/absent in the Phase4V path",
        },
        "residual_monitoring": {
            "required": ["mean_norm_delta_v", "mean_norm_lambda_g_delta_v", "mean_relative_correction_to_base"],
            "safety_rule": "Stop before training if correction norm becomes comparable to base feature norm; no lambda sweep is permitted.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "where_shape": report["where"]["shape"], "what_shape": report["what"]["shape"], "how_input_shape": report["how"]["input_shape"], "where_detached": not where.requires_grad, "what_reachable": semantic_code.requires_grad}, indent=2))


if __name__ == "__main__":
    main()
