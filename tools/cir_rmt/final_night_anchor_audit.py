#!/usr/bin/env python3
"""Bounded final-night audit of the H2 image-parameter Anchor.

This script reads existing H2/R/RA checkpoints, computes the exact current
per-tensor Anchor terms, and probes raw/weighted Anchor and native H2 task
gradients on one fixed VisA training batch.  It never updates model weights
and never launches a training loop.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
H2_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901")
H2_E1 = Path(
    "/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/"
    "phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_1.pth"
)
E0 = ROOT / "runs/h2_anchor_cir_master_20260901/common/e0.pth"
SNAPSHOTS = {
    "H2_E1": H2_E1,
    "E0": E0,
    "RA_E10": ROOT / "runs/h2_anchor_cir_master_20260901/RA/adapter_10.pth",
    "RA_E16": ROOT / "runs/h2_anchor_cir_master_20260901/RA/adapter_16.pth",
}
TARGET_BATCH_SIZE = 6
ANCHOR_LAMBDA = 1.0e-3


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(path: Path) -> Mapping[str, Any]:
    return torch.load(path.resolve(), map_location="cpu", weights_only=False)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _contributions(reference: Mapping[str, torch.Tensor], snapshots: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    names = list(reference)
    for snapshot, payload in snapshots.items():
        current = payload["image_adapter"]
        terms: list[float] = []
        zero_names: list[str] = []
        for name in names:
            ref = reference[name].detach().float().cpu()
            value = current[name].detach().float().cpu()
            ref_norm = float(ref.norm().item())
            denominator = max(float(ref.square().sum().item()), 1.0e-12)
            difference = value - ref
            diff_sq = float(difference.square().sum().item())
            term = diff_sq / denominator
            terms.append(term)
            if ref_norm <= 1.0e-6:
                zero_names.append(name)
            rows.append({
                "snapshot": snapshot,
                "parameter": name,
                "family": name.split(".", 1)[0],
                "numel": int(value.numel()),
                "reference_norm": ref_norm,
                "denominator": denominator,
                "denominator_clamped": bool(ref.square().sum().item() < 1.0e-12),
                "reference_zero_or_near_zero": bool(ref_norm <= 1.0e-6),
                "current_norm": float(value.norm().item()),
                "distance_from_reference": float(difference.norm().item()),
                "difference_squared": diff_sq,
                "raw_anchor_term": term,
                "percent_total_unweighted_anchor_loss": 100.0 * term / max(sum(terms), 1.0e-30),
            })
        total = sum(terms)
        # Percentages are calculated after the full term vector exists.
        offset = len(rows) - len(names)
        for row, term in zip(rows[offset:], terms):
            row["percent_total_unweighted_anchor_loss"] = 100.0 * term / max(total, 1.0e-30)
        summaries[snapshot] = {
            "n_parameters": len(names),
            "mean_anchor_loss": total / max(len(terms), 1),
            "weighted_mean_anchor_loss": ANCHOR_LAMBDA * total / max(len(terms), 1),
            "zero_or_near_zero_reference_count": len(zero_names),
            "zero_or_near_zero_reference_parameters": zero_names,
            "top_parameters_by_raw_term": [
                {"parameter": row["parameter"], "raw_anchor_term": row["raw_anchor_term"], "percent": row["percent_total_unweighted_anchor_loss"]}
                for row in sorted(rows[offset:], key=lambda item: float(item["raw_anchor_term"]), reverse=True)[:12]
            ],
        }
    return rows, summaries


def _set_payload(model: Any, payload: Mapping[str, Any], epoch: int) -> None:
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = True
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.soft_prompt.requires_grad_(epoch > 3)
    model.text_adapter.requires_grad_(True)
    model.image_adapter.requires_grad_(True)
    if hasattr(model, "set_dfg_beta"):
        model.set_dfg_beta(float(payload.get("dfg_beta_current", 0.0)))
    else:
        model.dfg_beta = float(payload.get("dfg_beta_current", 0.0))
    model.eval()


def _native_task_loss(model: Any, batch: Mapping[str, Any], modules: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=device.type == "cuda")
    mask = batch["mask"].to(device, non_blocking=device.type == "cuda")
    label = batch["label"].to(device, non_blocking=device.type == "cuda")
    class_names = [str(value) for value in batch["class_name"]]
    text_by_class: dict[str, torch.Tensor] = {}
    kg_losses: list[torch.Tensor] = []
    k_losses: list[torch.Tensor] = []
    for class_name in list(set(class_names)):
        text_levels, kg_loss, _stats, components = modules["get_hybrid_soft_prompt_single_class_text_embedding"](
            model, "VisA", class_name, device, return_kg=True, return_components=True
        )
        k_loss, _k_stats = modules["compute_hybrid_k_regularization"](
            model, components["hard_text"], components["soft_text"], float(model.hybrid_alpha_current)
        )
        text_by_class[class_name] = text_levels
        kg_losses.append(kg_loss)
        k_losses.append(k_loss)
    text_features = torch.stack([text_by_class[class_name] for class_name in class_names], dim=0).permute(1, 0, 2, 3)
    kg_loss = torch.stack(kg_losses).mean() if kg_losses else torch.zeros((), device=device)
    k_loss = torch.stack(k_losses).mean() if k_losses else torch.zeros((), device=device)
    with torch.cuda.amp.autocast(enabled=bool(cfg["amp"])):
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        det_features = torch.stack(det_tokens, dim=0)
        cls_pred = torch.stack([
            torch.matmul(det_features[index].unsqueeze(1), text_features[index]).squeeze(1)
            for index in range(det_features.shape[0])
        ], dim=0).mean(dim=0)
        cls_loss = torch.nn.functional.cross_entropy(cls_pred, label)
        seg_pred = model.vision_text_fusion_gate_seg(seg_features, text_features)
        seg_loss = modules["calculate_seg_loss"](seg_pred, mask)
        base_loss = cls_loss + seg_loss + float(cfg["lambda_kg"]) * kg_loss + float(cfg["lambda_k"]) * k_loss
    del image, mask, label, text_by_class, text_features, kg_loss, k_loss, seg_tokens, det_tokens, seg_features, det_features, cls_pred, seg_pred, cls_loss, seg_loss
    return base_loss


def _gradient_rows(model: Any, batch: Mapping[str, Any], modules: Mapping[str, Any], cfg: Mapping[str, Any], anchor: Any, payload: Mapping[str, Any], epoch: int, device: torch.device) -> list[dict[str, Any]]:
    params = list(model.image_adapter.named_parameters())
    model.zero_grad(set_to_none=True)
    anchor_loss = anchor.loss(model.image_adapter)
    anchor_grads = torch.autograd.grad(anchor_loss, [parameter for _, parameter in params], allow_unused=True)
    anchor_vectors = [gradient.detach().float().reshape(-1) if gradient is not None else torch.zeros(parameter.numel(), device=device) for (_, parameter), gradient in zip(params, anchor_grads)]
    model.zero_grad(set_to_none=True)
    base_loss = _native_task_loss(model, batch, modules, cfg, device)
    base_grads = torch.autograd.grad(base_loss, [parameter for _, parameter in params], allow_unused=True)
    base_vectors = [gradient.detach().float().reshape(-1) if gradient is not None else torch.zeros(parameter.numel(), device=device) for (_, parameter), gradient in zip(params, base_grads)]
    all_anchor = torch.cat(anchor_vectors)
    all_base = torch.cat(base_vectors)
    family_indices: dict[str, list[int]] = {}
    for index, (name, _parameter) in enumerate(params):
        family_indices.setdefault(name.split(".", 1)[0], []).append(index)

    def vector_stats(indices: Sequence[int]) -> tuple[float, float, float, float | None]:
        anchor_vec = torch.cat([anchor_vectors[index] for index in indices])
        base_vec = torch.cat([base_vectors[index] for index in indices])
        anchor_norm = float(anchor_vec.norm().item())
        base_norm = float(base_vec.norm().item())
        weighted = ANCHOR_LAMBDA * anchor_norm
        ratio = weighted / base_norm if base_norm > 0 else None
        cosine = float(torch.dot(anchor_vec, base_vec).item() / (anchor_norm * base_norm)) if anchor_norm > 0 and base_norm > 0 else None
        return anchor_norm, base_norm, ratio, cosine

    rows: list[dict[str, Any]] = []
    for name, parameter in params:
        index = next(index for index, (candidate, _) in enumerate(params) if candidate == name)
        anchor_norm, base_norm, ratio, cosine = vector_stats([index])
        rows.append({
            "snapshot": str(payload.get("_audit_snapshot", "")),
            "epoch": epoch,
            "row_type": "parameter",
            "family": name.split(".", 1)[0],
            "parameter": name,
            "numel": int(parameter.numel()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "base_task_loss": float(base_loss.detach().float().item()),
            "lambda_anchor": ANCHOR_LAMBDA,
            "anchor_grad_l2": anchor_norm,
            "base_task_grad_l2": base_norm,
            "weighted_anchor_grad_l2": ANCHOR_LAMBDA * anchor_norm,
            "weighted_anchor_to_base_ratio": ratio,
            "gradient_cosine": cosine,
            "batch_protocol": "one fixed VisA train batch; batch_size=6; shuffle=False; num_workers=0; seed=12345; native segmentation; AMP enabled",
        })
    global_anchor, global_base, global_ratio, global_cosine = vector_stats(list(range(len(params))))
    rows.append({
        "snapshot": str(payload.get("_audit_snapshot", "")), "epoch": epoch, "row_type": "global_image_adapter", "family": "__all__", "parameter": "__all__", "numel": int(sum(parameter.numel() for _, parameter in params)),
        "anchor_loss": float(anchor_loss.detach().float().item()), "base_task_loss": float(base_loss.detach().float().item()), "lambda_anchor": ANCHOR_LAMBDA,
        "anchor_grad_l2": global_anchor, "base_task_grad_l2": global_base, "weighted_anchor_grad_l2": ANCHOR_LAMBDA * global_anchor,
        "weighted_anchor_to_base_ratio": global_ratio, "gradient_cosine": global_cosine,
        "batch_protocol": "one fixed VisA train batch; batch_size=6; shuffle=False; num_workers=0; seed=12345; native segmentation; AMP enabled",
    })
    for family, indices in sorted(family_indices.items()):
        family_anchor, family_base, family_ratio, family_cosine = vector_stats(indices)
        rows.append({
            "snapshot": str(payload.get("_audit_snapshot", "")), "epoch": epoch, "row_type": "family", "family": family, "parameter": "__family__", "numel": int(sum(params[index][1].numel() for index in indices)),
            "anchor_loss": float(anchor_loss.detach().float().item()), "base_task_loss": float(base_loss.detach().float().item()), "lambda_anchor": ANCHOR_LAMBDA,
            "anchor_grad_l2": family_anchor, "base_task_grad_l2": family_base, "weighted_anchor_grad_l2": ANCHOR_LAMBDA * family_anchor,
            "weighted_anchor_to_base_ratio": family_ratio, "gradient_cosine": family_cosine,
            "batch_protocol": "one fixed VisA train batch; batch_size=6; shuffle=False; num_workers=0; seed=12345; native segmentation; AMP enabled",
        })
    del anchor_loss, base_loss, anchor_grads, base_grads, anchor_vectors, base_vectors, all_anchor, all_base
    model.zero_grad(set_to_none=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exact_h2_anchor_cir_master_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    from scripts.cir_rmt import train_h2_anchor_cir as runner

    modules = runner._load_h2_modules(Path(cfg.get("h2_repo_path", H2_REPO)))
    current = runner._load_current_cir_primitives(Path(cfg.get("h2_repo_path", H2_REPO)))
    anchor_class = current["ImageParameterAnchor"]
    device = torch.device(args.device)
    payloads = {snapshot: _load(path) for snapshot, path in SNAPSHOTS.items()}
    reference = payloads["H2_E1"]["image_adapter"]
    contribution_rows, summaries = _contributions(reference, payloads)
    _write_csv(
        output / "ANCHOR_PARAMETER_CONTRIBUTIONS.csv",
        ["snapshot", "parameter", "family", "numel", "reference_norm", "denominator", "denominator_clamped", "reference_zero_or_near_zero", "current_norm", "distance_from_reference", "difference_squared", "raw_anchor_term", "percent_total_unweighted_anchor_loss"],
        contribution_rows,
    )
    # Fetch the exact same deterministic bounded batch for every state.
    runner.seed_everything(12345)
    dataset = modules["TextAndImageDataset"](
        str(Path(cfg["source_root"]).resolve()),
        str((Path(cfg["h2_repo_path"]).resolve() / cfg["manifest_path"]).resolve()),
        int(cfg["img_size"]),
    )
    loader = DataLoader(dataset, batch_size=TARGET_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    model = runner.build_model(cfg, modules, device)
    anchor = anchor_class(reference, checkpoint_sha256=runner.sha256_file(H2_E1), epoch=1, config_sha256=None, device=device)
    gradient_rows: list[dict[str, Any]] = []
    for snapshot in ("H2_E1", "E0", "RA_E10", "RA_E16"):
        payload = payloads[snapshot]
        epoch = int(payload.get("epoch", 0))
        payload = dict(payload)
        payload["_audit_snapshot"] = snapshot
        _set_payload(model, payload, epoch)
        gradient_rows.extend(_gradient_rows(model, batch, modules, cfg, anchor, payload, epoch, device))
        print(f"completed gradient probe {snapshot}", flush=True)
    gradient_fields = ["snapshot", "epoch", "row_type", "family", "parameter", "numel", "anchor_loss", "base_task_loss", "lambda_anchor", "anchor_grad_l2", "base_task_grad_l2", "weighted_anchor_grad_l2", "weighted_anchor_to_base_ratio", "gradient_cosine", "batch_protocol"]
    _write_csv(output / "ANCHOR_GRADIENT_DECOMPOSITION.csv", gradient_fields, gradient_rows)
    summary = {
        "status": "COMPLETE",
        "reference": str(H2_E1.resolve()),
        "reference_sha256": runner.sha256_file(H2_E1),
        "config_sha256": runner.sha256_file(args.config.resolve()),
        "snapshots": {snapshot: {"path": str(path.resolve()), "sha256": runner.sha256_file(path)} for snapshot, path in SNAPSHOTS.items()},
        "gradient_batch": {"batch_size": TARGET_BATCH_SIZE, "shuffle": False, "num_workers": 0, "seed": 12345, "source": "VisA training manifest first batch", "amp": bool(cfg["amp"])},
        "anchor_lambda": ANCHOR_LAMBDA,
        "summaries": summaries,
        "missing_requested_snapshots": ["RA_E1", "RA_E4"],
        "missing_requested_snapshot_reason": "The RA run saved only candidate checkpoints E10/E12/E14/E16/E18/E20; no E1 or E4 model-state checkpoint exists.",
    }
    (output / "ANCHOR_SCALE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({snapshot: {"mean_anchor_loss": value["mean_anchor_loss"], "zero_or_near_zero_reference_count": value["zero_or_near_zero_reference_count"]} for snapshot, value in summaries.items()}, sort_keys=True), flush=True)
    print("ANCHOR_AUDIT_COMPLETE", flush=True)
    del model, loader, dataset, batch, anchor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
