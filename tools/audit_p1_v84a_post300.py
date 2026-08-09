#!/usr/bin/env python3
"""Forward-only root-cause audit for the completed P1-v8.4-A 300B state.

This tool intentionally constructs no optimizer, executes no backward pass,
and verifies that the adapter/H6 state is bitwise unchanged by the replay.
It recovers decision metrics that were not persisted by the training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import build_patch_targets
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


EXPECTED_GIT_HEAD = "1b88c1e45896a2eb25b2b84264152c7cffff4004"
EXPECTED_OPENAI_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        sample["dataset_index"] = torch.tensor(index, dtype=torch.int64)
        return sample


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for module_name, module in (
        ("image_adapter", model.image_adapter),
        ("text_adapter", model.text_adapter),
        ("soft_prompt", model.soft_prompt),
        ("h6", model.h6),
    ):
        for name, value in sorted(module.state_dict().items()):
            cpu = value.detach().contiguous().cpu()
            digest.update(module_name.encode())
            digest.update(name.encode())
            digest.update(str(cpu.dtype).encode())
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _stats(values: torch.Tensor, quantiles: Iterable[float]) -> dict[str, float | int]:
    values = values.detach().float().flatten()
    result: dict[str, float | int] = {"count": int(values.numel())}
    if not values.numel():
        result.update({"mean": 0.0, "std": 0.0})
        result.update({f"p{int(q * 100):02d}": 0.0 for q in quantiles})
        return result
    result.update({
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    })
    result.update({
        f"p{int(q * 100):02d}": float(torch.quantile(values, q).item())
        for q in quantiles
    })
    return result


def _effective_rank(functions: torch.Tensor) -> float:
    rows = functions.float().T
    singular = torch.linalg.svdvals(rows)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()).item())


def _functional_correlation(functions: torch.Tensor) -> dict[str, Any]:
    rows = functions.float().T
    centered = rows - rows.mean(dim=1, keepdim=True)
    unit = F.normalize(centered, dim=1)
    matrix = unit @ unit.T
    upper = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]
    return {
        "matrix": matrix.tolist(),
        "pairwise_mean": float(upper.mean().item()),
        "pairwise_min": float(upper.min().item()),
        "pairwise_max": float(upper.max().item()),
    }


def _winner_shares(winners: torch.Tensor, region: torch.Tensor, factors: int) -> list[float]:
    count = int(region.sum().item())
    return [
        float(((winners == factor) & region).sum().item() / max(count, 1))
        for factor in range(factors)
    ]


def _region_utility(
    region: torch.Tensor,
    base_loss: torch.Tensor,
    per_factor_loss: torch.Tensor,
    gain_rel: torch.Tensor,
    dense: torch.Tensor,
    targets: torch.Tensor,
    z0: torch.Tensor,
) -> dict[str, Any]:
    count = int(region.sum().item())
    if not count:
        raise RuntimeError("decision region unexpectedly has no valid patches")
    factor_means = per_factor_loss[region].mean(dim=0)
    best_single = factor_means.min()
    oracle_patch = per_factor_loss.min(dim=-1).values
    oracle = oracle_patch[region].mean()
    uniform_logits = z0 + 0.05 * gain_rel.new_zeros(gain_rel.shape[:-1])
    # Recover residual evidence exactly from candidate logits/loss inputs below.
    # This placeholder is overwritten by the caller-provided tensor attribute.
    residual = _region_utility.residual
    uniform_logits = z0 + 0.05 * residual.mean(dim=-1)
    soft_logits = z0 + 0.05 * (dense * residual).sum(dim=-1)
    hard_index = dense.argmax(dim=-1, keepdim=True)
    hard_logits = z0 + 0.05 * residual.gather(-1, hard_index).squeeze(-1)
    uniform_loss = F.binary_cross_entropy_with_logits(uniform_logits, targets, reduction="none")
    soft_loss = F.binary_cross_entropy_with_logits(soft_logits, targets, reduction="none")
    hard_loss = F.binary_cross_entropy_with_logits(hard_logits, targets, reduction="none")
    base = base_loss[region].mean()
    winners = gain_rel.argmax(dim=-1)
    best_gain = gain_rel.max(dim=-1).values
    noop_winner = torch.cat((base_loss.unsqueeze(-1), per_factor_loss), dim=-1).argmin(dim=-1)
    noop_shares = [
        float(((noop_winner == index) & region).sum().item() / count)
        for index in range(per_factor_loss.shape[-1] + 1)
    ]
    return {
        "patch_count": count,
        "Base": float(base.item()),
        "ResidualBestSingle": float(best_single.item()),
        "ResidualOracleMulti": float(oracle.item()),
        "SoftRouted": float(soft_loss[region].mean().item()),
        "HardRouted": float(hard_loss[region].mean().item()),
        "Uniform": float(uniform_loss[region].mean().item()),
        "Oracle_gain_vs_Base": float(((base - oracle) / base.clamp_min(1e-12)).item()),
        "BestSingle_gain_vs_Base": float(((base - best_single) / base.clamp_min(1e-12)).item()),
        "SoftRouted_gain_vs_Base": float(((base - soft_loss[region].mean()) / base.clamp_min(1e-12)).item()),
        "OracleMulti_minus_BestSingle": float((oracle - best_single).item()),
        "G_multi": float(((best_single - oracle) / base.clamp_min(1e-12)).item()),
        "all_harm_fraction": float((best_gain[region] <= 0.0).float().mean().item()),
        "no_op_selected_fraction": noop_shares[0],
        "no_op_and_factor_winner_shares": noop_shares,
        "factor_winner_shares": _winner_shares(winners, region, per_factor_loss.shape[-1]),
    }


def _support_row(best_gain: torch.Tensor, targets: torch.Tensor, threshold: float, label: str) -> dict[str, Any]:
    def split(region: torch.Tensor) -> dict[str, float | int]:
        values = best_gain[region]
        positive = values > threshold
        negative = values <= 0.0
        ambiguous = (values > 0.0) & (values <= threshold)
        return {
            "patch_count": int(values.numel()),
            "positive_fraction": float(positive.float().mean().item()),
            "negative_fraction": float(negative.float().mean().item()),
            "ambiguous_fraction": float(ambiguous.float().mean().item()),
        }

    return {
        "source": label,
        "threshold": float(threshold),
        "overall": split(torch.ones_like(targets, dtype=torch.bool)),
        "normal": split(targets < 0.5),
        "anomaly": split(targets >= 0.5),
    }


def _act_gate_audit(best_gain: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    positive = best_gain[best_gain > 0.0]
    quantiles = [("positive_p25", 0.25), ("positive_p50", 0.50),
                 ("positive_p75", 0.75), ("positive_p90", 0.90),
                 ("positive_p95", 0.95)]
    candidates: list[tuple[str, float]] = [("current", 0.02), ("zero_boundary", 0.0)]
    candidates.extend((name, float(torch.quantile(positive, q).item())) for name, q in quantiles)
    rows = []
    seen: set[float] = set()
    for label, threshold in candidates:
        if threshold in seen:
            continue
        seen.add(threshold)
        rows.append(_support_row(best_gain, targets, threshold, label))
    return {
        "current_threshold": 0.02,
        "negative_boundary": 0.0,
        "positive_best_gain_distribution": _stats(positive, (0.25, 0.50, 0.75, 0.90, 0.95)),
        "threshold_table": rows,
    }


def _teacher_audit(gain_rel: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    best_gain, winners = gain_rel.max(dim=-1)
    positive = best_gain > 0.0
    current_act_positive = best_gain > 0.02
    top_two = gain_rel.topk(2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]
    positive_margin = margins[positive]
    relative_margin = positive_margin / best_gain[positive].clamp_min(1e-12)
    rows = []
    for tau in (0.05, 0.03, 0.02):
        q = F.softmax(gain_rel / tau, dim=-1)
        entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1) / math.log(q.shape[-1])
        max_probability = q.max(dim=-1).values
        for threshold in (0.98, 0.99, 0.995):
            entropy_pass = entropy < threshold
            zero_joint = positive & entropy_pass
            current_joint = current_act_positive & entropy_pass
            rows.append({
                "tau_utility": tau,
                "entropy_threshold": threshold,
                "eligible_positive_patch_count": int(positive.sum().item()),
                "eligible_positive_fraction": float(positive.float().mean().item()),
                "entropy_passing_positive_patch_count": int(zero_joint.sum().item()),
                "entropy_passing_positive_fraction": float(zero_joint.sum().item() / max(int(positive.sum().item()), 1)),
                "joint_zero_boundary_patch_count": int(zero_joint.sum().item()),
                "joint_zero_boundary_support_fraction": float(zero_joint.float().mean().item()),
                "joint_current_act_gate_patch_count": int(current_joint.sum().item()),
                "joint_current_act_gate_support_fraction": float(current_joint.float().mean().item()),
                "teacher_entropy_positive_mean": float(entropy[positive].mean().item()),
                "teacher_max_probability_positive_mean": float(max_probability[positive].mean().item()),
            })
    normal = targets < 0.5
    anomaly = targets >= 0.5
    canonical = next(row for row in rows if row["tau_utility"] == 0.05 and row["entropy_threshold"] == 0.98)
    return {
        "positive_patch_count": int(positive.sum().item()),
        "positive_patch_fraction": float(positive.float().mean().item()),
        "best_vs_second_gain_margin_distribution": _stats(positive_margin, (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)),
        "best_vs_second_relative_margin_distribution": _stats(relative_margin, (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)),
        "material_factor_choice_fraction": {
            "definition": "best-vs-second gain margin exceeds 10% of positive best gain",
            "overall_positive": float((relative_margin > 0.10).float().mean().item()),
            "normal_positive": float(((margins > 0.10 * best_gain) & positive & normal).sum().item() / max(int((positive & normal).sum().item()), 1)),
            "anomaly_positive": float(((margins > 0.10 * best_gain) & positive & anomaly).sum().item() / max(int((positive & anomaly).sum().item()), 1)),
        },
        "winner_shares": {
            "overall": _winner_shares(winners, torch.ones_like(positive), gain_rel.shape[-1]),
            "positive": _winner_shares(winners, positive, gain_rel.shape[-1]),
            "normal_positive": _winner_shares(winners, positive & normal, gain_rel.shape[-1]),
            "anomaly_positive": _winner_shares(winners, positive & anomaly, gain_rel.shape[-1]),
        },
        "canonical": canonical,
        "sensitivity_matrix": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attempt-summary", type=Path, required=True)
    parser.add_argument("--openai-checkpoint", type=Path, default=Path("model/ViT-L-14-336px.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed audit: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded forward-only replay")
    if args.max_batches != 300:
        raise ValueError("this audit is locked to the completed 300-batch protocol")
    if _git_head() != EXPECTED_GIT_HEAD:
        raise RuntimeError("unexpected base Git HEAD")
    if _sha256(args.openai_checkpoint) != EXPECTED_OPENAI_SHA256:
        raise RuntimeError("OpenAI checkpoint SHA256 mismatch")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed(args.seed)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("h6_config", {})
    required = {
        "checkpoint_version": checkpoint.get("checkpoint_version") == 9,
        "checkpoint_git_sha": checkpoint.get("git_sha") == EXPECTED_GIT_HEAD,
        "progress_version": config.get("progress_version") == "P1-v8.4-A",
        "seed": checkpoint.get("seed") == args.seed,
        "img_size": checkpoint.get("img_size") == 518,
        "batch_size": checkpoint.get("batch_size") == 1,
        "grad_accum_steps": checkpoint.get("grad_accum_steps") == 6,
        "precision": checkpoint.get("precision") == "fp32",
        "tf32_off": checkpoint.get("tf32_enabled") is False,
        "amp_off": checkpoint.get("amp_enabled") is False,
        "rho_fixed": config.get("rho_fixed") is True,
        "residual_act": config.get("local_correction_semantics") == "act_times_routed_true_residual",
    }
    if not all(required.values()):
        raise RuntimeError(f"checkpoint contract failure: {[k for k, v in required.items() if not v]}")

    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    state_hash_before = _state_hash(model)
    all_grad_none_before = all(parameter.grad is None for parameter in model.parameters())

    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", 518, "train"))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed),
    )

    z0_records, target_records, residual_records = [], [], []
    dense_records, act_records = [], []
    dataset_indices: list[int] = []
    residual_definition_max_error = 0.0
    routed_reconstruction_max_error = 0.0
    started = time.monotonic()

    for batch_number, sample in enumerate(loader, start=1):
        if batch_number > args.max_batches:
            break
        dataset_indices.append(int(sample["dataset_index"].item()))
        image = sample["image"].to(device, non_blocking=True)
        mask = sample["mask"].to(device, non_blocking=True)
        local_valid = sample["local_mask_valid"].to(device, non_blocking=True)
        class_names = list(sample["class_name"])
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, "VisA", class_names, visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]),
                update_load_bias=False,
            )
            seg_features = torch.stack(visual["seg_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model, "VisA", class_names, device,
                use_hybrid_soft_prompt=True, use_soft_prompt=False,
            ).to(dtype=seg_features.dtype)
            _, _, z0 = model.vision_text_fusion_gate_seg(
                seg_features, text_global, img_size=518,
                h6_patch_logits=h6_batch["h6_logits"], return_details=True,
            )
            patch_count = int(h6_batch["factor_residual_logits"].shape[2])
            y_patch, valid_patch = build_patch_targets(mask, patch_count, local_valid)
            valid = valid_patch.unsqueeze(0).expand_as(z0)
            targets = y_patch.unsqueeze(0).expand_as(z0).float()
            residual = h6_batch["factor_residual_logits"].float()
            definition = (
                h6_batch["factor_patch_logits"].float()
                - h6_batch["noop_reference_logit"].float().unsqueeze(-1)
            )
            residual_definition_max_error = max(
                residual_definition_max_error,
                float((residual - definition).abs().max().item()),
            )
            reconstructed = h6_batch["act_probability"].float() * (
                h6_batch["prediction_probabilities"].float() * residual
            ).sum(dim=-1)
            routed_reconstruction_max_error = max(
                routed_reconstruction_max_error,
                float((h6_batch["h6_logits"].float() - reconstructed).abs().max().item()),
            )
            z0_records.append(z0.float()[valid].cpu())
            target_records.append(targets[valid].cpu())
            residual_records.append(residual[valid].cpu())
            dense_records.append(h6_batch["dense_probabilities"].float()[valid].cpu())
            act_records.append(h6_batch["act_probability"].float()[valid].cpu())
        if batch_number % args.progress_every == 0:
            print(json.dumps({"batches": batch_number, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)

    if len(dataset_indices) != args.max_batches:
        raise RuntimeError(f"replay ended after {len(dataset_indices)} batches")
    z0 = torch.cat(z0_records)
    targets = torch.cat(target_records)
    residual = torch.cat(residual_records)
    dense = torch.cat(dense_records)
    act = torch.cat(act_records)
    base_loss = F.binary_cross_entropy_with_logits(z0, targets, reduction="none")
    candidate_logits = z0.unsqueeze(-1) + 0.05 * residual
    per_factor_loss = F.binary_cross_entropy_with_logits(
        candidate_logits, targets.unsqueeze(-1).expand_as(candidate_logits), reduction="none"
    )
    gain_rel = (base_loss.unsqueeze(-1) - per_factor_loss) / base_loss.unsqueeze(-1).clamp_min(0.1)
    _region_utility.residual = residual
    regions = {
        "overall": _region_utility(torch.ones_like(targets, dtype=torch.bool), base_loss, per_factor_loss, gain_rel, dense, targets, z0),
        "normal": _region_utility(targets < 0.5, base_loss, per_factor_loss, gain_rel, dense, targets, z0),
        "anomaly": _region_utility(targets >= 0.5, base_loss, per_factor_loss, gain_rel, dense, targets, z0),
    }
    best_gain = gain_rel.max(dim=-1).values
    act_audit = _act_gate_audit(best_gain, targets)
    teacher_audit = _teacher_audit(gain_rel, targets)
    state_hash_after = _state_hash(model)
    all_grad_none_after = all(parameter.grad is None for parameter in model.parameters())
    with args.attempt_summary.open() as handle:
        saved = json.load(handle)
    saved_cumulative = saved["final_cumulative"]

    output = {
        "status": "PASS",
        "audit_kind": "FORWARD_ONLY_REPLAY",
        "source_provenance": {
            "saved_artifact_inspection": {
                "artifact": str(args.attempt_summary.resolve()),
                "exact_regional_triplets_available": False,
                "available_region_fields": sorted(saved_cumulative["normal_anomaly_breakdown"]["normal"]),
            },
            "regional_utility": "FORWARD_ONLY_REPLAY",
            "per_patch_gain_evidence": "FORWARD_ONLY_REPLAY",
        },
        "contract": {
            "checks": required,
            "git_head": _git_head(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "openai_checkpoint_sha256": EXPECTED_OPENAI_SHA256,
            "seed": args.seed,
            "dataset": "VisA/train",
            "batches": len(dataset_indices),
            "batch_size": 1,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True,
            "rho": model.h6.rho_values().detach().cpu().tolist(),
            "fp32": True,
            "tf32": False,
            "optimizer_constructed": False,
            "backward_executed": False,
            "optimizer_steps": 0,
            "all_grad_none_before": all_grad_none_before,
            "all_grad_none_after": all_grad_none_after,
            "model_state_hash_before": state_hash_before,
            "model_state_hash_after": state_hash_after,
            "model_state_unchanged": state_hash_before == state_hash_after,
            "dataset_indices": dataset_indices,
            "residual_definition_max_error": residual_definition_max_error,
            "routed_reconstruction_max_error": routed_reconstruction_max_error,
        },
        "attempt1_saved_reference": {
            key: saved_cumulative[key] for key in (
                "Base", "BestSingle", "OracleMulti", "SoftRouted", "HardRouted",
                "Uniform", "G_local", "G_multi", "winner_shares"
            )
        },
        "regional_utility": regions,
        "act_support": act_audit,
        "act_probability": {
            "overall_mean": float(act.mean().item()),
            "normal_mean": float(act[targets < 0.5].mean().item()),
            "anomaly_mean": float(act[targets >= 0.5].mean().item()),
        },
        "router_teacher": teacher_audit,
        "factor_specialization": {
            "G_local": regions["overall"]["Oracle_gain_vs_Base"],
            "G_multi": regions["overall"]["G_multi"],
            "BestSingle": regions["overall"]["ResidualBestSingle"],
            "OracleMulti": regions["overall"]["ResidualOracleMulti"],
            "residual_effective_rank": _effective_rank(residual),
            "residual_functional_correlation": _functional_correlation(residual),
            "best_second_utility_margin": teacher_audit["best_vs_second_gain_margin_distribution"],
            "material_factor_choice_fraction": teacher_audit["material_factor_choice_fraction"],
            "winner_shares": teacher_audit["winner_shares"],
        },
        "runtime_seconds": time.monotonic() - started,
    }
    invariants = output["contract"]
    if not (
        invariants["model_state_unchanged"]
        and invariants["all_grad_none_before"]
        and invariants["all_grad_none_after"]
        and invariants["residual_definition_max_error"] == 0.0
        and invariants["routed_reconstruction_max_error"] == 0.0
    ):
        output["status"] = "FAIL"
    _write_json_atomic(args.output, output)
    print(json.dumps({
        "status": output["status"],
        "output": str(args.output),
        "batches": len(dataset_indices),
        "valid_group_patches": int(targets.numel()),
        "runtime_seconds": round(output["runtime_seconds"], 3),
    }), flush=True)
    if output["status"] != "PASS":
        raise RuntimeError("forward-only audit invariant failed")


if __name__ == "__main__":
    main()
