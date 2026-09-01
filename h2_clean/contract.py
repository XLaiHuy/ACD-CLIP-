"""Small, dependency-light contracts for the clean H2 continuation.

The historical H2 checkpoint remains model-only and is loaded by the legacy
evaluator.  This module is used only by the new, opt-in training/evaluation
paths.  In particular, the anchor and CIR code deliberately have an exact
zero/disabled branch so a clean run cannot silently change the H2 baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


PROTOCOL_VERSION = "H2_CLEAN_REPRO_ANCHOR_CIR_V1"
ANCHOR_FORMULA = "sum_i||theta_i-theta_ref_i||^2/(sum_i||theta_ref_i||^2+eps)"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes, and CPU tensor bytes in sorted order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state value for {name!r} is not a tensor")
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_sha(repo: str | os.PathLike[str] = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed all owned RNGs; deterministic algorithms are an explicit choice."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class EpochWorkerInit:
    """DataLoader worker seeding with a reproducible, explicit epoch stream."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch_one_based: int) -> None:
        self.epoch = int(epoch_one_based)

    def __call__(self, worker_id: int) -> None:
        worker_seed = (self.base_seed + 1_000_003 * self.epoch + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if generator is not None:
        state["dataloader_generator_state"] = generator.get_state()
    return state


def restore_rng_state(state: Mapping[str, Any], generator: torch.Generator | None = None) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    if generator is not None and "dataloader_generator_state" in state:
        generator.set_state(state["dataloader_generator_state"])


def environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import torchvision

        manifest["torchvision"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        manifest["torchvision_error"] = repr(exc)
    if torch.cuda.is_available():
        manifest["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return manifest


class SafeImageAdapterAnchor:
    """One global normalized distance over image-adapter parameters.

    References are detached CPU clones and never participate in autograd.  A
    single global denominator prevents small or zero-valued tensors from
    dominating the objective.
    """

    def __init__(
        self,
        reference: Mapping[str, torch.Tensor],
        *,
        reference_sha256: str,
        reference_checkpoint_sha256: str | None = None,
        reference_epoch: int | None = None,
        reference_config_sha256: str | None = None,
        eps: float = 1.0e-12,
    ) -> None:
        if float(eps) <= 0:
            raise ValueError("anchor eps must be positive")
        self.reference = {
            name: value.detach().float().cpu().clone().requires_grad_(False)
            for name, value in reference.items()
        }
        self.reference_sha256 = str(reference_sha256)
        self.reference_checkpoint_sha256 = reference_checkpoint_sha256
        self.reference_epoch = None if reference_epoch is None else int(reference_epoch)
        self.reference_config_sha256 = reference_config_sha256
        self.eps = float(eps)

    @classmethod
    def from_module(cls, module: torch.nn.Module, **metadata: Any) -> "SafeImageAdapterAnchor":
        reference = {name: value.detach().float().cpu().clone() for name, value in module.named_parameters()}
        return cls(reference, reference_sha256=state_dict_sha256(reference), **metadata)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], device: torch.device) -> "SafeImageAdapterAnchor":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reference = payload.get("image_anchor_reference") if isinstance(payload, Mapping) else None
        if reference is None and isinstance(payload, Mapping):
            reference = payload.get("image_parameter_reference")
        if reference is None and isinstance(payload, Mapping):
            reference = payload.get("image_adapter")
        if not isinstance(reference, Mapping):
            raise ValueError(f"{path} does not contain an image_adapter reference")
        tensors = {name: value.detach().float().cpu().clone() for name, value in reference.items()}
        return cls(
            tensors,
            reference_sha256=state_dict_sha256(tensors),
            reference_checkpoint_sha256=sha256_file(path),
            reference_epoch=payload.get("epoch"),
            reference_config_sha256=payload.get("config_sha256"),
        )

    def loss(self, module: torch.nn.Module) -> torch.Tensor:
        live = dict(module.named_parameters())
        if set(live) != set(self.reference):
            missing = sorted(set(live) - set(self.reference))
            extra = sorted(set(self.reference) - set(live))
            raise ValueError(f"anchor parameter identity mismatch; missing={missing[:3]} extra={extra[:3]}")
        numerator: torch.Tensor | None = None
        denominator = torch.zeros((), device=next(module.parameters()).device, dtype=torch.float32)
        for name, parameter in live.items():
            reference = self.reference[name].to(device=parameter.device, dtype=torch.float32)
            if tuple(parameter.shape) != tuple(reference.shape):
                raise ValueError(f"anchor shape mismatch for {name}: {tuple(parameter.shape)} != {tuple(reference.shape)}")
            difference = parameter.float() - reference
            term = difference.square().sum()
            numerator = term if numerator is None else numerator + term
            denominator = denominator + reference.square().sum()
        if numerator is None:
            return denominator * 0.0
        return numerator / (denominator + self.eps)

    def metadata(self, coefficient: float) -> dict[str, Any]:
        return {
            "enabled": bool(float(coefficient) > 0.0),
            "lambda_image_anchor": float(coefficient),
            "formula": ANCHOR_FORMULA,
            "eps": self.eps,
            "scope": "image_adapter_parameters_only",
            "train_only": True,
            "reference_sha256": self.reference_sha256,
            "reference_checkpoint_sha256": self.reference_checkpoint_sha256,
            "reference_epoch": self.reference_epoch,
            "reference_config_sha256": self.reference_config_sha256,
        }


def _midpoint_median(values: torch.Tensor, dim: int) -> torch.Tensor:
    ordered, _ = torch.sort(values, dim=dim)
    count = ordered.shape[dim]
    if count % 2:
        return ordered.select(dim, count // 2)
    return (ordered.select(dim, count // 2 - 1) + ordered.select(dim, count // 2)) * 0.5


def select_gt_free_peers(
    stage_features: torch.Tensor,
    stage_margins: torch.Tensor,
    *,
    peer_count: int = 8,
    spatial_radius: int = 3,
) -> dict[str, torch.Tensor]:
    """Select a shared deterministic peer set from detached features/margins."""
    if stage_features.ndim != 4 or stage_margins.ndim != 3:
        raise ValueError("expected features [G,B,P,D] and margins [G,B,P]")
    groups, batch, patches, _ = stage_features.shape
    if tuple(stage_margins.shape) != (groups, batch, patches):
        raise ValueError("feature and margin geometry mismatch")
    if peer_count < 1 or peer_count >= patches:
        raise ValueError("peer_count must be in [1, P)")
    side = int(round(patches ** 0.5))
    if side * side != patches:
        raise ValueError(f"peer selection requires square patch geometry, got P={patches}")
    with torch.no_grad():
        features = stage_features.detach().float()
        margins = stage_margins.detach().float()
        if not torch.isfinite(features).all() or not torch.isfinite(margins).all():
            raise ValueError("non-finite peer inputs")
        shared = F.normalize(features.mean(dim=0), dim=-1)
        pooled = margins.mean(dim=0)
        center = _midpoint_median(pooled, dim=-1)
        normal_like = pooled <= center.unsqueeze(-1)
        similarity = torch.einsum("bpd,bqd->bpq", shared, shared)
        coords = torch.arange(patches, device=features.device)
        yy = torch.div(coords, side, rounding_mode="floor")
        xx = torch.remainder(coords, side)
        distance = torch.maximum(
            (yy[:, None] - yy[None, :]).abs(),
            (xx[:, None] - xx[None, :]).abs(),
        )
        allowed = normal_like[:, None, :] & (distance > int(spatial_radius))
        scores = similarity.masked_fill(~allowed, float("-inf"))
        indices = torch.topk(scores, k=int(peer_count), dim=-1, largest=True, sorted=True).indices
        candidate_count = allowed.sum(dim=-1)
        valid = candidate_count >= int(peer_count)
        indices = torch.where(valid.unsqueeze(-1), indices, torch.zeros_like(indices))
    return {
        "peer_indices": indices.detach().long(),
        "candidate_count": candidate_count.detach().long(),
        "valid": valid.detach(),
    }


def gather_peer_values(values: torch.Tensor, peer_indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or peer_indices.ndim != 3:
        raise ValueError("values must be [G,B,P] and indices must be [B,P,K]")
    groups, batch, patches = values.shape
    if tuple(peer_indices.shape[:2]) != (batch, patches):
        raise ValueError("peer index geometry mismatch")
    index = peer_indices.clamp(0, patches - 1).unsqueeze(0).expand(groups, -1, -1, -1)
    source = values.unsqueeze(-1).expand(-1, -1, -1, peer_indices.shape[-1])
    return source.gather(2, index)


def robust_peer_delta(
    observed_margin: torch.Tensor,
    peer_margins: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    mad_constant: float = 1.4826,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if peer_margins.shape[:-1] != observed_margin.shape:
        raise ValueError("observed and peer margin geometry mismatch")
    if peer_margins.shape[-1] < 1:
        raise ValueError("peer axis is empty")
    observed = observed_margin.detach().float()
    peers = peer_margins.detach().float()
    center = _midpoint_median(peers, dim=-1)
    mad = _midpoint_median((peers - center.unsqueeze(-1)).abs(), dim=-1)
    z = (observed - center) / (float(mad_constant) * mad + float(eps))
    delta = torch.tanh(z).detach()
    if not torch.isfinite(delta).all():
        raise ValueError("non-finite peer delta")
    return delta, {
        "center": center.detach(),
        "mad": mad.detach(),
        "z": z.detach(),
        "delta_abs_mean": delta.abs().mean().detach(),
        "delta_saturation_fraction": (delta.abs() > 0.95).float().mean().detach(),
    }


def cir_adjust_native_logits(
    native_logits: torch.Tensor,
    stage_features: torch.Tensor,
    alpha: float,
    *,
    peer_count: int = 8,
    spatial_radius: int = 3,
    eps: float = 1.0e-6,
    mad_constant: float = 1.4826,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply train-only, detached peer evidence in native segmentation space.

    The native H2 logits remain the differentiable path.  Relational evidence
    is computed under ``no_grad`` and only shifts normal/abnormal logits in
    opposite directions.  Alpha zero returns the original tensor unchanged.
    """
    if float(alpha) == 0.0:
        return native_logits, {"enabled": False, "alpha": 0.0}
    if native_logits.ndim != 4 or stage_features.ndim != 4:
        raise ValueError("native logits must be [G,B,2,P] and features [G,B,P,D]")
    groups, batch, classes, patches = native_logits.shape
    if classes != 2 or tuple(stage_features.shape[:3]) != (groups, batch, patches):
        raise ValueError("native CIR geometry mismatch")
    stage_margins = native_logits.detach()[:, :, 1, :] - native_logits.detach()[:, :, 0, :]
    peer_info = select_gt_free_peers(
        stage_features.detach(),
        stage_margins.detach(),
        peer_count=int(peer_count),
        spatial_radius=int(spatial_radius),
    )
    peer_values = gather_peer_values(stage_margins.detach(), peer_info["peer_indices"])
    delta, delta_stats = robust_peer_delta(
        stage_margins.detach(),
        peer_values,
        eps=eps,
        mad_constant=mad_constant,
    )
    # A sparse normal-like pool can leave some query patches without the
    # requested number of peers. The selector returns deterministic filler
    # indices for tensor-shape safety; invalid queries receive no relational
    # shift rather than borrowing the filler patch's margin.
    delta = torch.where(
        peer_info["valid"].unsqueeze(0),
        delta,
        torch.zeros_like(delta),
    )
    delta_stats["invalid_delta_zeroed"] = (~peer_info["valid"]).detach()
    delta_stats["delta_abs_mean"] = delta.abs().mean().detach()
    delta_stats["delta_saturation_fraction"] = (delta.abs() > 0.95).float().mean().detach()
    shift = float(alpha) * delta
    adjusted = torch.stack(
        [native_logits[:, :, 0, :] - shift, native_logits[:, :, 1, :] + shift],
        dim=2,
    )
    return adjusted, {
        "enabled": True,
        "alpha": float(alpha),
        "peer_indices": peer_info["peer_indices"],
        "candidate_count": peer_info["candidate_count"],
        "valid": peer_info["valid"],
        "delta": delta,
        **delta_stats,
    }


def build_full_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    repo: str | os.PathLike[str],
    clip_sha256: str | None,
    dataset_manifest_sha256: str | None,
    dataloader_generator: torch.Generator | None,
    anchor: SafeImageAdapterAnchor | None,
    anchor_lambda: float,
    seed: int,
    precision: str,
    tf32_enabled: bool,
) -> dict[str, Any]:
    image_state = {name: value.detach().cpu().clone() for name, value in model.image_adapter.state_dict().items()}
    image_parameter_reference = {
        name: value.detach().cpu().clone()
        for name, value in model.image_adapter.named_parameters()
    }
    text_state = {name: value.detach().cpu().clone() for name, value in model.text_adapter.state_dict().items()}
    payload: dict[str, Any] = {
        "checkpoint_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": {"image_adapter": image_state, "text_adapter": text_state},
        # Keep these aliases so the historical evaluator can still load new
        # checkpoints after selecting its explicit mode.
        "image_adapter": image_state,
        "image_parameter_reference": image_parameter_reference,
        "text_adapter": text_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "resolved_scientific_config": dict(config),
        "config_sha256": canonical_json_hash(dict(config)),
        "git_sha": current_git_sha(repo),
        "clip_sha256": clip_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed": int(seed),
        "precision": str(precision),
        "amp_enabled": bool(getattr(scaler, "is_enabled", lambda: False)()),
        "tf32_enabled": bool(tf32_enabled),
    }
    payload.update(capture_rng_state(dataloader_generator))
    if hasattr(model, "soft_prompt"):
        payload["soft_prompt"] = {
            name: value.detach().cpu().clone() for name, value in model.soft_prompt.state_dict().items()
        }
        payload["model_state"]["soft_prompt"] = payload["soft_prompt"]
    if anchor is not None:
        payload["image_anchor"] = anchor.metadata(anchor_lambda)
        payload["image_anchor_reference"] = {
            name: value.clone() for name, value in anchor.reference.items()
        }
    return payload


def restore_full_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    dataloader_generator: torch.Generator | None,
) -> tuple[int, int]:
    if int(payload.get("checkpoint_version", 0)) < 2:
        raise ValueError("exact continuation requires checkpoint_version >= 2; model-only H2 checkpoints are replay-only")
    state = payload.get("model_state", payload)
    model.image_adapter.load_state_dict(state["image_adapter"])
    model.text_adapter.load_state_dict(state["text_adapter"])
    if "soft_prompt" in state and hasattr(model, "soft_prompt"):
        model.soft_prompt.load_state_dict(state["soft_prompt"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload.get("scaler_state", {}))
    restore_rng_state(payload, dataloader_generator)
    return int(payload["epoch"]), int(payload["global_step"])


def checkpoint_required_keys() -> tuple[str, ...]:
    return (
        "model_state",
        "image_parameter_reference",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "dataloader_generator_state",
        "epoch",
        "global_step",
        "resolved_scientific_config",
        "config_sha256",
        "git_sha",
        "clip_sha256",
        "dataset_manifest_sha256",
    )
