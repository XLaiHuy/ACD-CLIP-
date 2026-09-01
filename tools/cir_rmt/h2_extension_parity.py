#!/usr/bin/env python3
"""Fixed-input parity and K-regularization audit for the exact H2 extension.

This is a pre-training test.  It imports the historical H2 model/data code
from the isolated detached worktree, loads only the common E0 model state,
and compares the historical native DFG/deployment path with the extension's
explicit native reconstruction.  It also checks the recovered historical
K-space regularizer's gradient path and the frozen V2 transport direction.

No checkpoint is rewritten and no training/evaluation result is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.cir_rmt.train_h2_anchor_cir import (
    DEFAULT_CONFIG,
    build_model,
    h2_dfg_weights,
    h2_native_weights_logits,
    logits_to_deployment_probability,
    logits_to_training_probability,
    current_git_sha,
    _load_current_cir_primitives,
    _load_h2_modules,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H2_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max().item())


def _load_e0(model: Any, e0_path: Path) -> Mapping[str, Any]:
    payload = torch.load(e0_path, map_location="cpu", weights_only=False)
    if payload.get("snapshot_kind") != "exact_h2_initialization_only":
        raise ValueError("parity requires the common exact-H2 E0 snapshot")
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.dfg_beta = float(payload.get("dfg_beta_current", model.dfg_beta))
    return payload


def _make_fixed_batch(modules: Mapping[str, Any], cfg: Mapping[str, Any], h2_repo: Path, device: torch.device) -> tuple[torch.Tensor, list[str], list[str]]:
    # BaseSingleClassDataset has the historical test preprocessing but no
    # stochastic train augmentation.  Two existing candle records provide a
    # deterministic, compact, same-class batch for the forward comparison.
    import dataset as h2_dataset  # type: ignore

    dataset = h2_dataset.BaseSingleClassDataset(
        str(Path(cfg["source_root"]).resolve()),
        str((h2_repo / "dataset" / "hub" / "VisA.jsonl").resolve()),
        int(cfg["img_size"]),
        "candle",
    )
    if len(dataset) < 2:
        raise ValueError("frozen VisA candle class does not contain two records")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch["image"].to(device), [str(v) for v in batch["class_name"]], [str(v) for v in batch["file_name"]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    h2_repo = Path(cfg.get("h2_repo_path", DEFAULT_H2_REPO)).resolve()
    modules = _load_h2_modules(h2_repo)
    current_cir = _load_current_cir_primitives(h2_repo)
    device = torch.device(args.device)
    model = build_model(cfg, modules, device)
    e0_payload = _load_e0(model, args.e0.resolve())
    model.eval()
    model.set_dfg_beta(float(cfg["dfg_beta"]))
    model.hybrid_alpha_current = 0.0
    image, class_names, file_names = _make_fixed_batch(modules, cfg, h2_repo, device)

    get_text = modules["get_hybrid_soft_prompt_single_class_text_embedding"]
    with torch.no_grad():
        text_by_class = {
            name: get_text(model, "VisA", name, device, return_kg=True, return_components=True)[0]
            for name in sorted(set(class_names))
        }
        text = torch.stack([text_by_class[name] for name in class_names], dim=0).permute(1, 0, 2, 3)
        with torch.cuda.amp.autocast(enabled=bool(cfg["amp"])):
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            det_features = torch.stack(det_tokens, dim=0)
            historical_train = model.vision_text_fusion_gate_seg(seg_features, text)
            historical_deployed = model.vision_text_fusion_gate_seg(
                seg_features, text, test_mode=True, domain="Industrial"
            )
            extension_weights, extension_logits = h2_native_weights_logits(model, seg_features, text)
            extension_train = logits_to_training_probability(extension_logits, int(cfg["img_size"]))[:, 1]
            extension_deployed = logits_to_deployment_probability(
                extension_logits, int(cfg["img_size"]), "Industrial"
            )[:, 1]
            historical_logits: list[torch.Tensor] = []
            direct_weights: list[torch.Tensor] = []
            group_text = text.permute(1, 0, 2, 3)
            for stage in range(int(seg_features.shape[0])):
                direct_weights.append(torch.stack(h2_dfg_weights(model, seg_features[stage], group_text, stage), dim=-1))
                fused = model._vision_text_attention_fusion(seg_features[stage], group_text, stage)
                historical_logits.append(torch.matmul(10.0 * seg_features[stage], fused))
            historical_logits_tensor = torch.stack(historical_logits, dim=0)
            direct_weights_tensor = torch.stack(direct_weights, dim=0)
            historical_cls = torch.stack(
                [torch.matmul(det_features[i].unsqueeze(1), text[i]).squeeze(1) for i in range(det_features.shape[0])],
                dim=0,
            ).mean(dim=0)

    weight_diff = _max_abs(direct_weights_tensor, extension_weights)
    native_logit_diff = _max_abs(historical_logits_tensor, extension_logits)
    train_prob_diff = _max_abs(historical_train[:, 1], extension_train)
    deployed_prob_diff = _max_abs(historical_deployed, extension_deployed)
    finite_forward = all(
        bool(torch.isfinite(value).all().item())
        for value in (image, text, seg_features, det_features, historical_logits_tensor, extension_logits, historical_train, extension_train, historical_deployed, extension_deployed, historical_cls)
    )

    # Actual E0 peer geometry: this is a GT-free validity check only.
    peer_delta_from_native_margins = current_cir["peer_delta_from_native_margins"]
    from tools.cir_rmt.core import transport_pair

    visual = F.normalize(seg_features.float(), dim=-1)
    prompts = F.normalize(group_text.float(), dim=-2)
    group_margins = torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)[..., 1] - torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)[..., 0]
    delta, delta_stats = peer_delta_from_native_margins(
        seg_features.detach().float(),
        group_margins.detach().float(),
        peer_count=int(cfg["rmt_peer_count"]),
        spatial_radius=int(cfg["rmt_spatial_radius"]),
        eps=float(cfg["rmt_eps"]),
        mad_constant=float(cfg["rmt_mad_constant"]),
    )
    direction_native = torch.full((1, 2), 0.5)
    direction_delta = torch.tensor([[0.4, -0.4]])
    direction_normal, direction_abnormal = transport_pair(
        direction_native, direction_native, direction_delta, 0.5, str(cfg["rmt_transport_direction"])
    )
    direction_pass = bool(
        direction_normal[0, 0] > direction_native[0, 0]
        and direction_abnormal[0, 0] < direction_native[0, 0]
        and direction_normal[0, 1] < direction_native[0, 1]
        and direction_abnormal[0, 1] > direction_native[0, 1]
    )

    # Recover the historical K-reg gradient contract: W_K is detached by
    # design, while the soft prompt is the intended differentiable input.
    model.zero_grad(set_to_none=True)
    model.soft_prompt.requires_grad_(True)
    model.hybrid_alpha_current = float(cfg["hybrid_alpha_max"]) * 0.25
    _, _, _, components = get_text(model, "VisA", class_names[0], device, return_kg=True, return_components=True)
    k_loss, k_stats = modules["compute_hybrid_k_regularization"](
        model, components["hard_text"], components["soft_text"], model.hybrid_alpha_current
    )
    weighted_k_loss = float(cfg["lambda_k"]) * k_loss
    weighted_k_loss.backward()
    soft_grad_norm = float(sum(
        parameter.grad.detach().float().norm().item()
        for parameter in model.soft_prompt.parameters()
        if parameter.grad is not None
    ))
    wk_grad_values = [
        parameter.grad for name, parameter in model.image_adapter.named_parameters()
        if "vision_text_k" in name and parameter.grad is not None
    ]
    wk_grad_norm = float(sum(value.detach().float().norm().item() for value in wk_grad_values)) if wk_grad_values else 0.0
    kreg_pass = bool(
        torch.isfinite(k_loss).all().item()
        and float(k_loss.detach().item()) > 0.0
        and torch.isfinite(weighted_k_loss).all().item()
        and soft_grad_norm > 0.0
        and wk_grad_norm == 0.0
        and abs(float(cfg["lambda_k"]) - 0.002) <= 1e-12
    )

    tolerance = float(args.tolerance)
    parity_pass = bool(
        finite_forward
        and weight_diff <= tolerance
        and native_logit_diff <= tolerance
        and train_prob_diff <= tolerance
        and deployed_prob_diff <= tolerance
        and bool(delta_stats["valid"].all().item())
        and direction_pass
        and kreg_pass
    )
    result: dict[str, Any] = {
        "status": "PASS" if parity_pass else "FAIL",
        "scope": "fixed_input_historical_h2_vs_extension_and_kreg",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_sha": current_git_sha(),
        "h2_repo": str(h2_repo),
        "h2_commit": cfg["h2_repo_commit"],
        "e0": str(args.e0.resolve()),
        "e0_sha256": sha256_file(args.e0.resolve()),
        "device": str(device),
        "fixed_batch": {"class_names": class_names, "file_names": file_names, "batch_size": len(file_names)},
        "tolerance": tolerance,
        "forward_finite": finite_forward,
        "max_abs_differences": {
            "dfg_weights": weight_diff,
            "native_logits": native_logit_diff,
            "training_probability": train_prob_diff,
            "deployed_probability": deployed_prob_diff,
        },
        "native_shapes": {
            "seg_features": list(seg_features.shape),
            "det_features": list(det_features.shape),
            "text_features": list(text.shape),
            "native_weights": list(extension_weights.shape),
            "native_logits": list(extension_logits.shape),
        },
        "peer_audit": {
            "peer_count": int(cfg["rmt_peer_count"]),
            "spatial_radius": int(cfg["rmt_spatial_radius"]),
            "valid_fraction": float(delta_stats["valid"].float().mean().item()),
            "candidate_count_min": int(delta_stats["candidate_count"].min().item()),
            "delta_abs_mean": float(delta.float().abs().mean().item()),
        },
        "transport_direction_audit": {
            "configured": cfg["rmt_transport_direction"],
            "positive_delta_increases_normal_group0": bool(direction_normal[0, 0] > direction_native[0, 0]),
            "positive_delta_decreases_abnormal_group0": bool(direction_abnormal[0, 0] < direction_native[0, 0]),
            "pass": direction_pass,
        },
        "k_regularization_audit": {
            "lambda_k": float(cfg["lambda_k"]),
            "alpha_probe": float(model.hybrid_alpha_current),
            "loss": float(k_loss.detach().item()),
            "weighted_loss": float(weighted_k_loss.detach().item()),
            "soft_prompt_grad_norm": soft_grad_norm,
            "vision_text_k_grad_norm": wk_grad_norm,
            "stats": k_stats,
            "expected": "soft-prompt path receives K-reg gradient; detached W_K receives none",
            "pass": kreg_pass,
        },
        "e0_payload_epoch": int(e0_payload.get("epoch", -1)),
        "weights_modified": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    md = output.with_suffix(".md")
    md.write_text(
        "# H2 Extension Parity\n\n"
        f"Status: `{result['status']}`\n\n"
        "This fixed-input test compares the historical H2 native DFG and deployment path with the extension reconstruction using the common E0 model state. It is a portability/implementation test, not a training or target-selection result.\n\n"
        f"- H2 commit: `{cfg['h2_repo_commit']}`\n"
        f"- E0 SHA256: `{result['e0_sha256']}`\n"
        f"- Fixed batch: `{', '.join(file_names)}`\n"
        f"- Tolerance: `{tolerance}`\n"
        f"- DFG weight max abs diff: `{weight_diff:.9g}`\n"
        f"- Native-logit max abs diff: `{native_logit_diff:.9g}`\n"
        f"- Training-probability max abs diff: `{train_prob_diff:.9g}`\n"
        f"- Deployment-probability max abs diff: `{deployed_prob_diff:.9g}`\n"
        f"- Peer validity fraction: `{result['peer_audit']['valid_fraction']:.6f}`\n"
        f"- K-reg loss: `{result['k_regularization_audit']['loss']:.9g}`\n"
        f"- Soft-prompt K-reg gradient norm: `{soft_grad_norm:.9g}`\n"
        f"- Detached W_K gradient norm: `{wk_grad_norm:.9g}`\n\n"
        "The K-reg check expects gradients through the soft-prompt input and no gradient through detached W_K, matching the historical implementation. The V2 direction check is synthetic and only verifies the configured abnormal-minus/normal-plus sign.\n",
        encoding="utf-8",
    )
    print(f"H2_EXTENSION_PARITY={result['status']}")
    print(f"OUTPUT={output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
