#!/usr/bin/env python3
"""One forward-only, frozen-checkpoint Router target-sharpness audit.

The audit intentionally replays the canonical 300 VisA/train microbatches
without constructing an optimizer or retaining model outputs.  It persists
only margin-selected q-router aggregates for the predeclared tau values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import build_patch_targets
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from tools.audit_p1_v84a_post300 import (
    _IndexedDataset,
    _git_head,
    _seed,
    _sha256,
    _write_json_atomic,
)
from tools.audit_p1_v84a_teacher_semantics import (
    CHECKPOINT_ORIGIN,
    EXPECTED_CHECKPOINT_GIT_SHA,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_OPENAI_SHA256,
    _full_state_hash,
)
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


TAUS = (0.05, 0.03, 0.02)
MARGIN_REL_THRESHOLD = 0.10
ENTROPY_REFERENCE = 0.98
EPSILON = 1e-12


def _stats(values: list[torch.Tensor]) -> dict[str, float | int]:
    tensor = torch.cat(values).float() if values else torch.empty(0, dtype=torch.float32)
    result: dict[str, float | int] = {"count": int(tensor.numel())}
    if not tensor.numel():
        result.update({"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0})
        return result
    return {
        **result,
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def _winner_stats(values: list[torch.Tensor]) -> dict[str, float | int]:
    stats = _stats(values)
    return {key: stats[key] for key in ("count", "mean", "p50")}


class _SharpnessAccumulator:
    """CPU aggregate that retains only scalar target diagnostics for quantiles."""

    def __init__(self) -> None:
        self.valid_count = {name: 0 for name in ("overall", "normal", "anomaly")}
        self.selected_count = {name: 0 for name in ("overall", "normal", "anomaly")}
        self.condition_count = {
            region: {
                name: 0
                for name in (
                    "valid", "best_gain_gt_0", "margin_rel_gt_0_10",
                    "valid_and_best_gain_gt_0", "valid_and_margin_rel_gt_0_10",
                    "eligible",
                )
            }
            for region in ("overall", "normal", "anomaly")
        }
        self.winner_count = {
            region: {f"F{factor + 1}": 0 for factor in range(4)}
            for region in ("overall", "normal", "anomaly")
        }
        self.values: dict[float, dict[str, dict[str, list[torch.Tensor]]]] = {
            tau: {
                region: {metric: [] for metric in ("entropy", "max_probability", "kl_uniform")}
                for region in ("overall", "normal", "anomaly")
            }
            for tau in TAUS
        }
        self.winners: dict[float, dict[str, dict[str, list[torch.Tensor]]]] = {
            tau: {
                f"F{factor + 1}": {metric: [] for metric in ("entropy", "max_probability")}
                for factor in range(4)
            }
            for tau in TAUS
        }
        self.entropy_below_reference = {
            tau: {region: 0 for region in ("overall", "normal", "anomaly")}
            for tau in TAUS
        }
        self.q_argmax_agreement = {tau: [0, 0] for tau in TAUS}
        self.finite = True
        self.raw_values = {
            region: {metric: [] for metric in ("best_gain", "second_gain", "margin_rel")}
            for region in ("overall", "normal", "anomaly")
        }
        self.per_factor_gain = {f"F{factor + 1}": [] for factor in range(4)}
        self.valid_winner_count = {
            region: {f"F{factor + 1}": 0 for factor in range(4)}
            for region in ("overall", "normal", "anomaly")
        }

    def add(self, gain: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        """Add one [G,B,P,M] gain batch without retaining model tensors."""
        if gain.shape[:-1] != targets.shape or valid.shape != targets.shape:
            raise ValueError("gain, targets, and valid shapes are inconsistent")
        best, winners = gain.max(dim=-1)
        second = gain.topk(2, dim=-1).values[..., 1]
        margin_rel = (best - second) / best.abs().clamp_min(EPSILON)
        selected = valid & (best > 0.0) & (margin_rel > MARGIN_REL_THRESHOLD)
        conditions = {
            "valid": valid,
            "best_gain_gt_0": best > 0.0,
            "margin_rel_gt_0_10": margin_rel > MARGIN_REL_THRESHOLD,
            "valid_and_best_gain_gt_0": valid & (best > 0.0),
            "valid_and_margin_rel_gt_0_10": valid & (margin_rel > MARGIN_REL_THRESHOLD),
            "eligible": selected,
        }
        regions = {
            "overall": torch.ones_like(valid, dtype=torch.bool),
            "normal": targets < 0.5,
            "anomaly": targets >= 0.5,
        }
        for name, region in regions.items():
            self.valid_count[name] += int((valid & region).sum().item())
            self.selected_count[name] += int((selected & region).sum().item())
            for condition_name, condition in conditions.items():
                self.condition_count[name][condition_name] += int((condition & region).sum().item())
            for factor in range(4):
                self.valid_winner_count[name][f"F{factor + 1}"] += int(
                    ((winners == factor) & valid & region).sum().item()
                )
                self.winner_count[name][f"F{factor + 1}"] += int(
                    ((winners == factor) & selected & region).sum().item()
                )
            raw_mask = valid & region
            for metric, tensor in (("best_gain", best), ("second_gain", second), ("margin_rel", margin_rel)):
                self.raw_values[name][metric].append(tensor[raw_mask].detach().float().cpu())
        for factor in range(4):
            self.per_factor_gain[f"F{factor + 1}"].append(gain[..., factor][valid].detach().float().cpu())

        for tau in TAUS:
            q = F.softmax(gain / tau, dim=-1)
            entropy = -(q * q.clamp_min(EPSILON).log()).sum(dim=-1) / math.log(4.0)
            max_probability = q.max(dim=-1).values
            kl_uniform = (q * (q.clamp_min(EPSILON).log() + math.log(4.0))).sum(dim=-1)
            self.finite = self.finite and bool(torch.isfinite(q[selected]).all().item())
            q_winners = q.argmax(dim=-1)
            self.q_argmax_agreement[tau][0] += int(((q_winners == winners) & selected).sum().item())
            self.q_argmax_agreement[tau][1] += int(selected.sum().item())
            for name, region in regions.items():
                mask = selected & region
                for metric, tensor in (
                    ("entropy", entropy),
                    ("max_probability", max_probability),
                    ("kl_uniform", kl_uniform),
                ):
                    self.values[tau][name][metric].append(tensor[mask].detach().float().cpu())
                self.entropy_below_reference[tau][name] += int(
                    ((entropy < ENTROPY_REFERENCE) & mask).sum().item()
                )
            for factor in range(4):
                mask = selected & (winners == factor)
                name = f"F{factor + 1}"
                self.winners[tau][name]["entropy"].append(entropy[mask].detach().float().cpu())
                self.winners[tau][name]["max_probability"].append(
                    max_probability[mask].detach().float().cpu()
                )

    def result(self) -> dict[str, Any]:
        support = {
            region: {
                "valid_count": self.valid_count[region],
                "selected_count": self.selected_count[region],
                "selected_fraction": self.selected_count[region] / max(self.valid_count[region], 1),
            }
            for region in ("overall", "normal", "anomaly")
        }
        winner = {
            name: {
                "count": count,
                "share": count / max(self.selected_count["overall"], 1),
            }
            for name, count in self.winner_count["overall"].items()
        }
        tau_results: dict[str, Any] = {}
        for tau in TAUS:
            regions: dict[str, Any] = {}
            for region in ("overall", "normal", "anomaly"):
                count = self.selected_count[region]
                regions[region] = {
                    "selected_count": count,
                    "normalized_entropy": _stats(self.values[tau][region]["entropy"]),
                    "max_q_probability": _stats(self.values[tau][region]["max_probability"]),
                    "kl_q_to_uniform": _stats(self.values[tau][region]["kl_uniform"]),
                    "fraction_normalized_entropy_below_0_98": (
                        self.entropy_below_reference[tau][region] / max(count, 1)
                    ),
                }
            tau_results[f"{tau:.2f}"] = {
                "router_tau_utility": tau,
                "regions": regions,
                "by_winner": {
                    winner_name: {
                        "selected_count": self.winner_count["overall"][winner_name],
                        "normalized_entropy": _winner_stats(metrics["entropy"]),
                        "max_q_probability": _winner_stats(metrics["max_probability"]),
                    }
                    for winner_name, metrics in self.winners[tau].items()
                },
                "q_argmax_matches_raw_winner": (
                    self.q_argmax_agreement[tau][0] == self.q_argmax_agreement[tau][1]
                ),
                "q_argmax_agreement_count": self.q_argmax_agreement[tau][0],
            }
        return {
            "definition": "valid and best_gain > 0 and margin_rel > 0.10",
            "margin_rel": "(best_gain - second_gain) / max(abs(best_gain), 1e-12)",
            "margin_rel_threshold": MARGIN_REL_THRESHOLD,
            "support": support,
            "condition_counts": self.condition_count,
            "winner_shares": winner,
            "winner_counts_by_region": self.winner_count,
            "valid_winner_counts_by_region": self.valid_winner_count,
            "raw_gain_diagnostics": {
                region: {
                    metric: _stats(values) for metric, values in metrics.items()
                }
                for region, metrics in self.raw_values.items()
            },
            "per_factor_gain_rel": {
                factor: _stats(values) for factor, values in self.per_factor_gain.items()
            },
            "anomaly_non_f1_coverage": {
                name: {"count": self.winner_count["anomaly"][name]}
                for name in ("F2", "F3", "F4")
            },
            "finite_q_router": self.finite,
            "tau": tau_results,
        }


def _checkpoint_checks(checkpoint: dict[str, Any], seed: int) -> dict[str, bool]:
    config = checkpoint.get("h6_config", {})
    return {
        "checkpoint_version": checkpoint.get("checkpoint_version") == 9,
        "checkpoint_git_sha": checkpoint.get("git_sha") == EXPECTED_CHECKPOINT_GIT_SHA,
        "progress_version": config.get("progress_version") == "P1-v8.4-A",
        "seed": checkpoint.get("seed") == seed,
        "img_size": checkpoint.get("img_size") == 518,
        "batch_size": checkpoint.get("batch_size") == 1,
        "grad_accum_steps": checkpoint.get("grad_accum_steps") == 6,
        "precision_fp32": checkpoint.get("precision") == "fp32",
        "tf32_off": checkpoint.get("tf32_enabled") is False,
        "amp_off": checkpoint.get("amp_enabled") is False,
        "rho_fixed": config.get("rho_fixed") is True,
        "residual_act_semantics": config.get("local_correction_semantics") == "act_times_routed_true_residual",
    }


def _provenance_checks(provenance: dict[str, Any], checkpoint_sha: str) -> dict[str, bool]:
    validation = provenance.get("validation", {})
    return {
        "checkpoint_origin": provenance.get("checkpoint_origin") == CHECKPOINT_ORIGIN,
        "historical_original_checkpoint_lost": provenance.get("original_checkpoint_lost") is True,
        "checkpoint_sha256": checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
        "provenance_checkpoint_sha256": provenance.get("canonical_checkpoint", {}).get("sha256") == EXPECTED_CHECKPOINT_SHA256,
        "tier_a_protocol_pass": validation.get("tier_a_protocol_pass") is True,
        "tier_b_numerical_pass": validation.get("tier_b_numerical_pass") is True,
        "tier_c_scientific_pass": validation.get("tier_c_scientific_pass") is True,
        "safe_for_forward_only_teacher_audit": provenance.get("safe_for_forward_only_teacher_audit") is True,
    }


def _tau_usable(audit: dict[str, Any], tau: str) -> bool:
    entry = audit["tau"][tau]
    anomaly_winners = audit["anomaly_non_f1_coverage"]
    return bool(
        audit["finite_q_router"]
        and entry["q_argmax_matches_raw_winner"]
        and audit["support"]["normal"]["selected_count"] > 0
        and audit["support"]["anomaly"]["selected_count"] > 0
        and all(anomaly_winners[name]["count"] > 0 for name in ("F2", "F3", "F4"))
        and entry["regions"]["overall"]["normalized_entropy"]["p50"] < ENTROPY_REFERENCE
        and entry["regions"]["anomaly"]["normalized_entropy"]["p50"] < ENTROPY_REFERENCE
    )


def _baseline_accepted(sharpness: dict[str, Any], invariants: dict[str, bool]) -> bool:
    support = sharpness["support"]
    anomaly_winners = sharpness["winner_counts_by_region"]["anomaly"]
    return bool(
        all(invariants.values())
        and support["overall"]["selected_fraction"] >= 0.90
        and support["normal"]["selected_fraction"] >= 0.90
        and support["anomaly"]["selected_fraction"] >= 0.50
        and sum(value > 0 for value in anomaly_winners.values()) >= 2
        and sharpness["raw_gain_diagnostics"]["normal"]["best_gain"]["p50"] > 0.0
        and sharpness["raw_gain_diagnostics"]["anomaly"]["best_gain"]["p50"] > 0.0
    )


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/adapter_1.pth"))
    parser.add_argument("--provenance", type=Path, default=Path("runs/p1_v84a_gpu/regenerated_300b_checkpoint_provenance.json"))
    parser.add_argument("--openai-checkpoint", type=Path, default=Path("model/ViT-L-14-336px.pt"))
    parser.add_argument("--historical-audit", type=Path, default=Path("runs/p1_v84a_gpu/post300_teacher_semantics_audit.json"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/p1_v84a_gpu/router_margin_fingerprinted_baseline"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    output_paths = {
        "manifest": args.output_dir / "input_manifest.json",
        "fingerprint": args.output_dir / "fingerprint_summary.json",
        "conditions": args.output_dir / "condition_counts.json",
        "sharpness": args.output_dir / "q_sharpness.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("refusing to overwrite a fingerprinted Router baseline artifact")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded forward-only replay")
    if args.max_batches != 300 or args.progress_every != 50:
        raise ValueError("audit is locked to 300 batches with milestones every 50")

    checkpoint_sha = _sha256(args.checkpoint)
    openai_sha = _sha256(args.openai_checkpoint)
    provenance = json.loads(args.provenance.read_text())
    provenance_checks = _provenance_checks(provenance, checkpoint_sha)
    if not all(provenance_checks.values()):
        raise RuntimeError(f"regeneration provenance failure: {[k for k, v in provenance_checks.items() if not v]}")
    if openai_sha != EXPECTED_OPENAI_SHA256:
        raise RuntimeError("OpenAI checkpoint SHA256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_checks = _checkpoint_checks(checkpoint, args.seed)
    if not all(checkpoint_checks.values()):
        raise RuntimeError(f"checkpoint contract failure: {[k for k, v in checkpoint_checks.items() if not v]}")

    historical = json.loads(args.historical_audit.read_text())
    historical_margin = next(
        row for row in historical["router"]["margin_support"]["rows"]
        if row["margin_rel_threshold"] == MARGIN_REL_THRESHOLD
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed(args.seed)
    device = torch.device("cuda:0")
    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    all_grads_none_before = all(parameter.grad is None for parameter in model.parameters())
    state_hash_before = _full_state_hash(model)
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", 518, "train"))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed),
    )
    accumulator = _SharpnessAccumulator()
    batch_indices: list[int] = []
    input_manifest: list[dict[str, Any]] = []
    residual_definition_max_error = 0.0
    routed_reconstruction_max_error = 0.0
    actual_reconstruction_max_error = 0.0
    started = time.monotonic()
    for batch_number, sample in enumerate(loader, start=1):
        if batch_number > args.max_batches:
            break
        dataset_index = int(sample["dataset_index"].item())
        batch_indices.append(dataset_index)
        meta = dataset.dataset.meta[dataset_index]
        image_relative = str(meta["image_path"])
        mask_relative = meta.get("mask_path")
        data_root = Path(dataset.dataset.data_path)
        manifest_row = {
            "sample_index": dataset_index,
            "category": str(meta["class_name"]),
            "image_relative_path": image_relative,
            "image_sha256": _sha256(data_root / image_relative),
            "mask_relative_path": str(mask_relative) if mask_relative else None,
            "mask_sha256": _sha256(data_root / mask_relative) if mask_relative else None,
        }
        input_manifest.append(manifest_row)
        image = sample["image"].to(device, non_blocking=True)
        mask = sample["mask"].to(device, non_blocking=True)
        local_valid = sample["local_mask_valid"].to(device, non_blocking=True)
        class_names = list(sample["class_name"])
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, "VisA", class_names, visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]), update_load_bias=False,
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
            base_loss = F.binary_cross_entropy_with_logits(z0.float(), targets, reduction="none")
            candidate_logits = z0.float().unsqueeze(-1) + 0.05 * residual
            factor_loss = F.binary_cross_entropy_with_logits(
                candidate_logits, targets.unsqueeze(-1).expand_as(candidate_logits), reduction="none"
            )
            gain = (base_loss.unsqueeze(-1) - factor_loss) / base_loss.unsqueeze(-1).clamp_min(0.1)
            accumulator.add(gain, targets, valid)
            definition = h6_batch["factor_patch_logits"].float() - h6_batch["noop_reference_logit"].float().unsqueeze(-1)
            residual_definition_max_error = max(residual_definition_max_error, float((residual - definition).abs().max().item()))
            routed = (h6_batch["dense_probabilities"].float() * residual).sum(dim=-1)
            reconstructed = h6_batch["act_probability"].float() * routed
            routed_reconstruction_max_error = max(routed_reconstruction_max_error, float((h6_batch["h6_logits"].float() - reconstructed).abs().max().item()))
            actual_formula = z0.float() + reconstructed * model.h6.rho_values().view(-1, 1, 1)
            actual_payload = z0.float() + h6_batch["rho_scaled_actual_correction"].float()
            actual_reconstruction_max_error = max(actual_reconstruction_max_error, float((actual_formula - actual_payload).abs().max().item()))
        if batch_number % args.progress_every == 0:
            print(json.dumps({"batches": batch_number, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
    if len(batch_indices) != 300:
        raise RuntimeError(f"replay ended after {len(batch_indices)} batches")

    sharpness = accumulator.result()
    historical_delta = {
        region: {
            "historical_selected_count": historical_margin["regions"][region]["count"],
            "new_selected_count": sharpness["support"][region]["selected_count"],
            "count_delta": sharpness["support"][region]["selected_count"] - historical_margin["regions"][region]["count"],
            "historical_fraction": historical_margin["regions"][region]["fraction"],
            "new_fraction": sharpness["support"][region]["selected_fraction"],
        }
        for region in ("overall", "normal", "anomaly")
    }
    state_hash_after = _full_state_hash(model)
    all_grads_none_after = all(parameter.grad is None for parameter in model.parameters())
    invariant_checks = {
        "all_grads_none_before": all_grads_none_before,
        "all_grads_none_after": all_grads_none_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "residual_definition_exact": residual_definition_max_error == 0.0,
        "routed_correction_reconstruction_exact": routed_reconstruction_max_error == 0.0,
        "actual_gated_reconstruction_exact": actual_reconstruction_max_error == 0.0,
        "exactly_300_batches": len(batch_indices) == 300,
    }
    invariants_ok = all(invariant_checks.values())
    baseline_accepted = _baseline_accepted(sharpness, invariant_checks)
    tau05_usable = _tau_usable(sharpness, "0.05") if baseline_accepted else None
    alternate_usable = (
        {tau: _tau_usable(sharpness, tau) for tau in ("0.03", "0.02")}
        if baseline_accepted else None
    )
    decision = (
        "ROUTER_NEW_BASELINE_STRUCTURAL_DRIFT" if not baseline_accepted
        else "ROUTER_TAU_CANONICAL_USABLE" if tau05_usable
        else "ROUTER_TAU_RECALIBRATION_REQUIRED" if any(alternate_usable.values())
        else "ROUTER_TARGET_FORMULATION_UNRESOLVED"
    )
    output = {
        "audit_kind": "FORWARD_ONLY_FINGERPRINTED_ROUTER_MARGIN_BASELINE",
        "status": "PASS" if decision == "ROUTER_TAU_CANONICAL_USABLE" else "EXIT_FOR_DISCUSSION",
        "decision": decision,
        "contract": {
            "checks": {**provenance_checks, **checkpoint_checks, **invariant_checks},
            "checkpoint_origin": CHECKPOINT_ORIGIN,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_git_sha": checkpoint.get("git_sha"),
            "current_git_head": _git_head(),
            "openai_checkpoint_sha256": openai_sha,
            "dataset": "VisA/train",
            "dataset_root": str(getattr(dataset.dataset, "data_path", "unknown")),
            "dataset_length": len(dataset),
            "seed": args.seed,
            "batches": len(batch_indices),
            "batch_indices": batch_indices,
            "batch_size": 1,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "preprocessing": {
                "image_size": 518,
                "image_resize": "bicubic",
                "mask_resize": "nearest",
                "image_normalize_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_normalize_std": [0.26862954, 0.26130258, 0.27577711],
                "train_augmentations": "seeded noise/color jitter plus geometric rotate/translate/hflip/vflip",
                "local_mask_valid": "ones transformed with the geometric mask operations then thresholded > 0.5",
            },
            "dataloader": {
                "batch_size": 1, "shuffle": True, "num_workers": 4,
                "pin_memory": True, "generator": "make_dataloader_generator(seed)",
                "worker_init": "seed_worker",
            },
            "model_switches": {
                "progress_version": "P1-v8.4-A", "rho": 0.05,
                "rho_trainable": False, "hybrid_text": True,
                "soft_prompt": False, "update_load_bias": False,
            },
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "backward_executed": False,
            "model_state_hash_before": state_hash_before,
            "model_state_hash_after": state_hash_after,
            "rho": 0.05,
            "rho_trainable": False,
            "residual_definition_max_abs_error": residual_definition_max_error,
            "routed_reconstruction_max_abs_error": routed_reconstruction_max_error,
            "actual_gated_reconstruction_max_abs_error": actual_reconstruction_max_error,
        },
        "sharpness": sharpness,
        "historical_delta": historical_delta,
        "baseline_acceptance": {
            "accepted": baseline_accepted,
            "criteria": {
                "overall_selected_fraction_ge_0_90": sharpness["support"]["overall"]["selected_fraction"] >= 0.90,
                "normal_selected_fraction_ge_0_90": sharpness["support"]["normal"]["selected_fraction"] >= 0.90,
                "anomaly_selected_fraction_ge_0_50": sharpness["support"]["anomaly"]["selected_fraction"] >= 0.50,
                "at_least_two_anomaly_winners": sum(value > 0 for value in sharpness["winner_counts_by_region"]["anomaly"].values()) >= 2,
                "normal_best_gain_p50_positive": sharpness["raw_gain_diagnostics"]["normal"]["best_gain"]["p50"] > 0.0,
                "anomaly_best_gain_p50_positive": sharpness["raw_gain_diagnostics"]["anomaly"]["best_gain"]["p50"] > 0.0,
            },
        },
        "tau_usable_contract": {
            "reference": "median normalized entropy < 0.98 overall and anomaly",
            "evaluated": invariants_ok,
            "canonical_tau_0_05_usable": tau05_usable,
            "alternate_tau_usable": alternate_usable,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    manifest = {
        "dataset": "VisA/train", "seed": args.seed,
        "ordered_manifest_sha256": _manifest_digest(input_manifest),
        "samples": input_manifest,
    }
    _write_json_atomic(output_paths["manifest"], manifest)
    _write_json_atomic(output_paths["fingerprint"], output)
    _write_json_atomic(output_paths["conditions"], {
        "condition_counts": sharpness["condition_counts"],
        "valid_winner_counts_by_region": sharpness["valid_winner_counts_by_region"],
        "eligible_winner_counts_by_region": sharpness["winner_counts_by_region"],
        "raw_gain_diagnostics": sharpness["raw_gain_diagnostics"],
        "per_factor_gain_rel": sharpness["per_factor_gain_rel"],
    })
    _write_json_atomic(output_paths["sharpness"], {
        "sharpness": sharpness, "tau_usable_contract": output["tau_usable_contract"],
    })
    print(json.dumps({
        "status": output["status"], "decision": decision, "batches": len(batch_indices),
        "support": sharpness["support"], "runtime_seconds": round(output["runtime_seconds"], 3),
        "output_dir": str(args.output_dir),
    }), flush=True)
    if not invariants_ok:
        raise RuntimeError(
            f"forward-only invariant failure: {[k for k, v in invariant_checks.items() if not v]}"
        )


if __name__ == "__main__":
    main()
