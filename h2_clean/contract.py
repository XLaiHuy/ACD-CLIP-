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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


CHECKPOINT_VERSION = 3
PROTOCOL_VERSION = "H2_CLEAN_REPRO_ANCHOR_CIR_V2_REDTEAM"
H2_BASE_COMMIT = "e03966997d4cecfd985943a4053a93e1e40197ec"
ANCHOR_FORMULA = "sum_i||theta_i-theta_ref_i||^2/(sum_i||theta_ref_i||^2+eps)"
ANCHOR_FAMILY_BUDGET_DEFAULT = 0.10
ANCHOR_TASK_FLOOR_MULTIPLIER = 1.0e-6
ANCHOR_TASK_FLOOR_MIN = 1.0e-12
ANCHOR_GRAD_EPS = 1.0e-12
ANCHOR_FAMILY_NAMES = (
    "lora_adapters",
    "m_i_w",
    "seg_proj",
    "det_proj",
    "seg_layer_norms",
    "det_layer_norms",
    "vision_text_q",
    "vision_text_k",
    "dfg_ss2d_branches",
    "dfg_raw_gamma",
    "direction_logits",
    "remaining_image_adapter_params",
)


RESUME_BRANCH_KEYS = ("use_safe_anchor", "anchor_lambda", "anchor_reference_sha256", "anchor_gradient_budget", "use_cir_training", "cir_alpha", "cir_peer_count", "cir_spatial_radius")
SCIENTIFIC_CONFIG_KEYS = ("model_name", "img_size", "dataset", "epoch", "n_groups", "image_adapt_weight", "text_adapt_weight", "lora_rank", "lora_alpha", "conv_lora_rank", "conv_lora_alpha", "conv_kernel_size_list", "batch_size", "image_lr", "text_lr", "use_soft_prompt", "use_hybrid_soft_prompt", "hybrid_alpha_max", "soft_prompt_freeze_epochs", "soft_prompt_ctx_len", "soft_prompt_lr", "soft_prompt_init", "soft_prompt_init_phrase", "lambda_kg", "lambda_k", "lr_gamma", "dfg_mode", "dfg_attn_dim", "dfg_attn_tau", "use_ss2d_dfg", "dfg_gamma_max", "dfg_ss2d_fusion", "dfg_beta", "dfg_beta_schedule", "dfg_beta_target", "dfg_weight_residual_fp32", "grad_clip_norm", "amp", "grad_checkpointing", "seed", "deterministic_algorithms", "tf32_enabled", "use_safe_anchor", "anchor_lambda", "anchor_reference_sha256", "anchor_gradient_budget", "anchor_family_budget", "use_cir_training", "cir_alpha", "cir_peer_count", "cir_spatial_radius", "cir_transport_direction", "cir_score_mode", "cir_reference_commit", "clip_sha256", "dataset_manifest_sha256", "base_h2_commit", "implementation_git_sha", "working_tree_diff_sha256")
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


def scientific_config_from_mapping(
        raw: Mapping[str, Any],
        *,
        clip_sha256: str | None,
        dataset_manifest_sha256: str | None,
        implementation_git_sha: str | None,
        working_tree_diff_sha256: str | None = None,
        anchor_reference_sha256: str | None = None,
        base_h2_commit: str = H2_BASE_COMMIT,
        tf32_enabled: bool = False,
        parent: bool = False,
) -> dict[str, Any]:
    """Build the scientific identity while excluding operational paths."""
    source = dict(raw)
    config: dict[str, Any] = {}
    for key in SCIENTIFIC_CONFIG_KEYS:
        if key == "epoch":
            value = source.get("protocol_horizon", source.get("epoch"))
        elif key == "tf32_enabled":
            value = bool(tf32_enabled)
        elif key == "anchor_reference_sha256":
            value = anchor_reference_sha256
        elif key == "anchor_gradient_budget":
            value = bool(source.get(key, False))
        elif key == "anchor_family_budget":
            value = float(source.get(key, ANCHOR_FAMILY_BUDGET_DEFAULT))
        elif key == "clip_sha256":
            value = clip_sha256
        elif key == "dataset_manifest_sha256":
            value = dataset_manifest_sha256
        elif key == "base_h2_commit":
            value = base_h2_commit
        elif key == "implementation_git_sha":
            value = implementation_git_sha
        elif key == "working_tree_diff_sha256":
            value = working_tree_diff_sha256
        elif key == "dfg_weight_residual_fp32":
            value = source.get(key, True)
        elif key == "cir_transport_direction":
            value = source.get(key, "abnormal_minus_normal_plus")
        elif key == "cir_score_mode":
            value = source.get(key, "exact_score_space")
        elif key == "cir_reference_commit":
            value = source.get(key, "9cc0ad4cc6b34e34a8c15e74df881866516b3181")
        else:
            value = source.get(key)
        config[key] = value
    if parent:
        config = parent_scientific_config(config)
    return config


def parent_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable source-parent identity for an intervention arm."""
    parent = dict(config)
    parent.update({
        "use_safe_anchor": False,
        "anchor_lambda": 0.0,
        "anchor_reference_sha256": None,
        "anchor_gradient_budget": False,
        "use_cir_training": False,
        "cir_alpha": 0.0,
        "cir_peer_count": 8,
        "cir_spatial_radius": 3,
    })
    return parent


def operational_config_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Keep save/resume/runtime controls separate from scientific identity."""
    source = dict(raw)
    keys = (
        "save_path", "resume", "max_batches", "num_workers", "cuda_device",
        "non_finite_loss_abort_threshold", "anchor_reference_path",
        "anchor_grad_audit_interval", "trace_batch_identity", "anchor_family_audit", "protocol_horizon",
    )
    return {key: source.get(key) for key in keys}


def _branch_only_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = sorted(set(left) | set(right))
    return all(
        key in RESUME_BRANCH_KEYS or left.get(key) == right.get(key)
        for key in keys
    )


def validate_resume_identity(
        payload: Mapping[str, Any],
        *,
        expected_scientific_config: Mapping[str, Any],
        expected_parent_config: Mapping[str, Any] | None = None,
        expected_epoch: int | None = None,
        expected_total_epoch: int | None = None,
        expected_seed: int | None = None,
        expected_clip_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_git_sha: str | None = None,
        expected_worktree_diff_sha256: str | None = None,
) -> None:
    """Reject stale or cross-protocol full-state checkpoints before training."""
    required = checkpoint_required_keys()
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"resume checkpoint missing scientific identity fields: {missing}")
    if int(payload.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"resume checkpoint_version mismatch: expected {CHECKPOINT_VERSION}, "
            f"got {payload.get('checkpoint_version')}"
        )
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"resume protocol_version mismatch: expected {PROTOCOL_VERSION}, "
            f"got {payload.get('protocol_version')}"
        )
    actual = payload.get("resolved_scientific_config")
    if not isinstance(actual, Mapping):
        raise ValueError("resume checkpoint has no resolved_scientific_config mapping")
    expected = dict(expected_scientific_config)
    expected_parent = dict(expected_parent_config or parent_scientific_config(expected))
    actual_parent = payload.get("parent_scientific_config")
    if not isinstance(actual_parent, Mapping) or dict(actual_parent) != expected_parent:
        raise ValueError("resume scientific parent identity mismatch")
    if dict(actual) != expected:
        source_epoch = int(payload.get("epoch", -1))
        if source_epoch != 1 or not _branch_only_difference(actual, expected):
            raise ValueError("resume scientific configuration mismatch")
    if payload.get("config_sha256") != canonical_json_hash(dict(actual)):
        raise ValueError("resume config_sha256 does not match resolved scientific config")
    if expected_epoch is not None and int(payload.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(
            f"resume epoch mismatch: expected {expected_epoch}, got {payload.get('epoch')}"
        )
    if expected_total_epoch is not None and not (0 < int(payload.get("epoch", 0)) < int(expected_total_epoch)):
        raise ValueError(
            f"resume epoch {payload.get('epoch')} is not a valid parent of total_epoch={expected_total_epoch}"
        )
    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("resume seed mismatch")
    if expected_clip_sha256 is not None and payload.get("clip_sha256") != expected_clip_sha256:
        raise ValueError("resume CLIP SHA mismatch")
    if expected_manifest_sha256 is not None and payload.get("dataset_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("resume dataset manifest SHA mismatch")
    if expected_git_sha is not None and payload.get("git_sha") != expected_git_sha:
        raise ValueError("resume scientific Git revision mismatch")
    if expected_worktree_diff_sha256 is not None and payload.get("working_tree_diff_sha256") != expected_worktree_diff_sha256:
        raise ValueError("resume scientific worktree revision mismatch")
    expected_amp = bool(expected.get("amp", False))
    expected_precision = "amp" if expected_amp else "fp32"
    if bool(payload.get("amp_enabled")) != expected_amp or str(payload.get("precision")) != expected_precision:
        raise ValueError("resume AMP/precision identity mismatch")
    if bool(payload.get("tf32_enabled")) != bool(expected.get("tf32_enabled", False)):
        raise ValueError("resume TF32 identity mismatch")
    if "seed" in expected and int(payload.get("seed", -1)) != int(expected["seed"]):
        raise ValueError("resume scientific seed mismatch")
    if expected_clip_sha256 is None and "clip_sha256" in expected and payload.get("clip_sha256") != expected["clip_sha256"]:
        raise ValueError("resume scientific CLIP SHA mismatch")
    if expected_manifest_sha256 is None and "dataset_manifest_sha256" in expected and payload.get("dataset_manifest_sha256") != expected["dataset_manifest_sha256"]:
        raise ValueError("resume scientific manifest SHA mismatch")

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
        for name in sorted(live):
            parameter = live[name]
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


def anchor_parameter_family(name: str) -> str:
    """Map every image-adapter parameter to exactly one semantic family."""
    root = name.split(".", 1)[0]
    if root == "dfg_ss2d_branches":
        if name.endswith(".direction_logits"):
            return "direction_logits"
        return "dfg_ss2d_branches"
    if root in (
        "lora_adapters",
        "m_i_w",
        "seg_proj",
        "det_proj",
        "seg_layer_norms",
        "det_layer_norms",
        "vision_text_q",
        "vision_text_k",
        "dfg_raw_gamma",
    ):
        return root
    if root == "direction_logits":
        return "direction_logits"
    return "remaining_image_adapter_params"


def partition_image_adapter_parameters(
        module: torch.nn.Module,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Partition all image-adapter parameters without silent exclusions."""
    families = {family: [] for family in ANCHOR_FAMILY_NAMES}
    seen: dict[str, str] = {}
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if name in seen:
            raise ValueError(f"duplicate image-adapter parameter: {name}")
        family = anchor_parameter_family(name)
        if family not in families:
            raise ValueError(f"unknown Anchor family {family!r} for {name!r}")
        families[family].append((name, parameter))
        seen[name] = family
    if len(seen) != sum(len(entries) for entries in families.values()):
        raise ValueError("Anchor family partition is not complete and disjoint")
    return families


def _gradient_statistics(
        task_gradients: Iterable[torch.Tensor | None],
        raw_anchor_gradients: Iterable[torch.Tensor | None],
) -> dict[str, float | None]:
    task_list = list(task_gradients)
    raw_list = list(raw_anchor_gradients)
    values = [
        gradient
        for gradient in (*task_list, *raw_list)
        if gradient is not None
    ]
    device = values[0].device if values else torch.device("cpu")
    task_sq = torch.zeros((), device=device, dtype=torch.float32)
    raw_sq = torch.zeros((), device=device, dtype=torch.float32)
    dot = torch.zeros((), device=device, dtype=torch.float32)
    for task_gradient, raw_gradient in zip(task_list, raw_list):
        task_value = None if task_gradient is None else task_gradient.detach().float()
        raw_value = None if raw_gradient is None else raw_gradient.detach().float()
        if task_value is not None:
            task_sq = task_sq + task_value.square().sum()
        if raw_value is not None:
            raw_sq = raw_sq + raw_value.square().sum()
        if task_value is not None and raw_value is not None:
            dot = dot + (task_value * raw_value).sum()
    task_norm = float(task_sq.sqrt().item())
    raw_norm = float(raw_sq.sqrt().item())
    cosine = None
    if task_norm > 0.0 and raw_norm > 0.0:
        cosine = float((dot / (task_sq.sqrt() * raw_sq.sqrt())).item())
    return {
        "task_grad_norm": task_norm,
        "anchor_grad_raw_norm": raw_norm,
        "cosine_task_anchor_raw": cosine,
    }


def _resolve_task_gradients(
        image_adapter: torch.nn.Module,
        all_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        task_gradients: Mapping[str, torch.Tensor | None] | None,
        raw_anchor_gradients: Mapping[str, torch.Tensor | None],
        anchor_lambda: float,
) -> tuple[dict[int, torch.Tensor], float, int, dict[int, str]]:
    families = partition_image_adapter_parameters(image_adapter)
    image_by_id = {
        id(parameter): name
        for entries in families.values()
        for name, parameter in entries
    }
    explicit = dict(task_gradients or {})
    named = list(all_named_parameters)
    task_by_id: dict[int, torch.Tensor] = {}
    global_sq: torch.Tensor | None = None
    for _, parameter in sorted(named, key=lambda item: item[0]):
        parameter_id = id(parameter)
        if parameter_id in image_by_id:
            name = image_by_id[parameter_id]
            if name in explicit:
                task_value = explicit[name]
            else:
                total_value = parameter.grad
                raw_value = raw_anchor_gradients.get(name)
                if total_value is None and raw_value is None:
                    task_value = None
                elif total_value is None:
                    task_value = -float(anchor_lambda) * raw_value.detach().float()
                elif raw_value is None:
                    task_value = total_value.detach().float()
                else:
                    task_value = (
                        total_value.detach().float()
                        - float(anchor_lambda) * raw_value.detach().float()
                    )
        else:
            task_value = None if parameter.grad is None else parameter.grad.detach().float()
        if task_value is None:
            continue
        task_value = task_value.detach().float()
        if not torch.isfinite(task_value).all():
            raise FloatingPointError("non-finite task gradient in Anchor family budget")
        task_by_id[parameter_id] = task_value
        term = task_value.square().sum()
        global_sq = term if global_sq is None else global_sq + term
    total_trainable_parameters = sum(
        parameter.numel()
        for _, parameter in named
        if parameter.requires_grad
    )
    global_task_norm = 0.0 if global_sq is None else float(global_sq.sqrt().item())
    return task_by_id, global_task_norm, max(total_trainable_parameters, 1), image_by_id


def _family_status(
        task_norm: float,
        lambda_anchor_norm: float,
        task_floor: float,
        rho: float,
) -> str:
    if task_norm <= task_floor:
        return "TASK_NEAR_ZERO"
    raw_ratio = lambda_anchor_norm / max(task_norm, ANCHOR_GRAD_EPS)
    if raw_ratio > 1.0:
        return "ANCHOR_DOMINANT"
    if raw_ratio > rho:
        return "ANCHOR_MODERATE"
    if raw_ratio <= 0.01:
        return "ANCHOR_NEGLIGIBLE"
    return "TASK_ACTIVE"


def _family_budget_core(
        image_adapter: torch.nn.Module,
        all_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        task_gradients: Mapping[str, torch.Tensor | None] | None,
        raw_anchor_gradients: Mapping[str, torch.Tensor | None] | None,
        anchor_lambda: float,
        rho: float,
        total_trainable_parameters: int | None,
        mutate: bool,
) -> dict[str, Any]:
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError(f"Anchor family budget rho must be in [0, 1], got {rho}")
    if float(anchor_lambda) < 0.0:
        raise ValueError("Anchor lambda must be non-negative")
    named = list(all_named_parameters)
    raw = dict(raw_anchor_gradients or {})
    task_by_id, global_task_norm, inferred_parameter_count, image_by_id = _resolve_task_gradients(
        image_adapter,
        named,
        task_gradients,
        raw,
        float(anchor_lambda),
    )
    total_parameters = (
        inferred_parameter_count
        if total_trainable_parameters is None
        else max(int(total_trainable_parameters), 1)
    )
    task_floor = max(
        ANCHOR_TASK_FLOOR_MIN,
        ANCHOR_TASK_FLOOR_MULTIPLIER
        * global_task_norm
        / (float(total_parameters) ** 0.5),
    )
    families = partition_image_adapter_parameters(image_adapter)
    family_rows: dict[str, dict[str, Any]] = {}
    global_effective_sq = 0.0
    max_active_effective_ratio = 0.0
    for family in ANCHOR_FAMILY_NAMES:
        entries = families[family]
        task_values = [task_by_id.get(id(parameter)) for _, parameter in entries]
        raw_values = [raw.get(name) for name, _ in entries]
        stats = _gradient_statistics(task_values, raw_values)
        task_norm = float(stats["task_grad_norm"])
        raw_norm = float(stats["anchor_grad_raw_norm"])
        lambda_raw_norm = abs(float(anchor_lambda)) * raw_norm
        if task_norm <= task_floor:
            scale = 0.0
        elif lambda_raw_norm <= 0.0:
            scale = 1.0
        else:
            scale = min(
                1.0,
                float(rho) * task_norm
                / (lambda_raw_norm + ANCHOR_GRAD_EPS),
            )
        effective_norm = abs(float(anchor_lambda)) * scale * raw_norm
        raw_ratio = (
            None
            if task_norm <= task_floor
            else raw_norm / max(task_norm, ANCHOR_GRAD_EPS)
        )
        lambda_times_raw_over_task = (
            None
            if raw_ratio is None
            else abs(float(anchor_lambda)) * raw_ratio
        )
        effective_ratio = (
            0.0
            if task_norm <= task_floor
            else effective_norm / max(task_norm, ANCHOR_GRAD_EPS)
        )
        status = _family_status(task_norm, lambda_raw_norm, task_floor, float(rho))
        for (name, parameter), task_value, raw_value in zip(
                entries, task_values, raw_values
        ):
            if raw_value is not None and not torch.isfinite(raw_value).all():
                raise FloatingPointError(
                    f"non-finite raw Anchor gradient in family {family}: {name}"
                )
            if mutate:
                if task_value is None and raw_value is None:
                    parameter.grad = None
                    continue
                task_tensor = (
                    torch.zeros_like(raw_value.detach().float())
                    if task_value is None
                    else task_value.detach().float()
                )
                raw_tensor = (
                    torch.zeros_like(task_tensor)
                    if raw_value is None
                    else raw_value.detach().float()
                )
                new_gradient = (
                    task_tensor
                    + float(anchor_lambda) * float(scale) * raw_tensor
                )
                if not torch.isfinite(new_gradient).all():
                    raise FloatingPointError(
                        f"non-finite effective Anchor gradient in family {family}: {name}"
                    )
                dtype = parameter.grad.dtype if parameter.grad is not None else parameter.dtype
                parameter.grad = new_gradient.to(device=parameter.device, dtype=dtype)
        global_effective_sq += effective_norm * effective_norm
        if task_norm > task_floor:
            max_active_effective_ratio = max(max_active_effective_ratio, effective_ratio)
        family_rows[family] = {
            **stats,
            "parameter_count": int(sum(parameter.numel() for _, parameter in entries)),
            "task_floor": task_floor,
            "lambda_anchor_grad_norm": lambda_raw_norm,
            "effective_anchor_grad_norm": effective_norm,
            "raw_gradient_ratio": raw_ratio,
            "lambda_times_raw_over_task": lambda_times_raw_over_task,
            "effective_ratio": effective_ratio,
            "scale": float(scale),
            "status": status,
            "finite": True,
        }
    global_effective_norm = global_effective_sq ** 0.5
    return {
        "rho": float(rho),
        "anchor_lambda": float(anchor_lambda),
        "task_floor": task_floor,
        "global_task_grad_norm": global_task_norm,
        "global_effective_anchor_grad_norm": global_effective_norm,
        "global_effective_ratio": (
            0.0
            if global_task_norm <= 0.0
            else global_effective_norm / max(global_task_norm, ANCHOR_GRAD_EPS)
        ),
        "max_effective_active_family_ratio": max_active_effective_ratio,
        "total_trainable_parameters": total_parameters,
        "family_partition_complete": True,
        "families": family_rows,
    }


def collect_family_gradient_metrics(
        image_adapter: torch.nn.Module,
        all_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        task_gradients: Mapping[str, torch.Tensor | None] | None = None,
        raw_anchor_gradients: Mapping[str, torch.Tensor | None] | None = None,
        anchor_lambda: float = 0.0,
        rho: float = ANCHOR_FAMILY_BUDGET_DEFAULT,
        total_trainable_parameters: int | None = None,
) -> dict[str, Any]:
    """Collect family metrics without mutating gradients."""
    return _family_budget_core(
        image_adapter,
        all_named_parameters,
        task_gradients=task_gradients,
        raw_anchor_gradients=raw_anchor_gradients,
        anchor_lambda=anchor_lambda,
        rho=rho,
        total_trainable_parameters=total_trainable_parameters,
        mutate=False,
    )


def apply_family_safe_anchor_budget(
        image_adapter: torch.nn.Module,
        all_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        task_gradients: Mapping[str, torch.Tensor | None],
        raw_anchor_gradients: Mapping[str, torch.Tensor | None],
        anchor_lambda: float,
        rho: float = ANCHOR_FAMILY_BUDGET_DEFAULT,
        total_trainable_parameters: int | None = None,
) -> dict[str, Any]:
    """Replace image-adapter grads with task plus a capped Anchor component."""
    return _family_budget_core(
        image_adapter,
        all_named_parameters,
        task_gradients=task_gradients,
        raw_anchor_gradients=raw_anchor_gradients,
        anchor_lambda=anchor_lambda,
        rho=rho,
        total_trainable_parameters=total_trainable_parameters,
        mutate=True,
    )


def aggregate_anchor_family_metrics(
        metrics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-step family telemetry once per epoch."""
    rows = list(metrics)
    if not rows:
        return {}
    def mean(key: str, values: Iterable[Mapping[str, Any]]) -> float | None:
        numeric = [
            float(row[key])
            for row in values
            if row.get(key) is not None
        ]
        return None if not numeric else float(np.mean(numeric))
    families: dict[str, Any] = {}
    for family in ANCHOR_FAMILY_NAMES:
        family_rows = [row["families"][family] for row in rows]
        aggregate = {
            "parameter_count": family_rows[0]["parameter_count"],
            "task_floor": mean("task_floor", family_rows),
            "task_grad_norm": mean("task_grad_norm", family_rows),
            "anchor_grad_raw_norm": mean("anchor_grad_raw_norm", family_rows),
            "lambda_anchor_grad_norm": mean("lambda_anchor_grad_norm", family_rows),
            "effective_anchor_grad_norm": mean("effective_anchor_grad_norm", family_rows),
            "raw_gradient_ratio": mean("raw_gradient_ratio", family_rows),
            "lambda_times_raw_over_task": mean("lambda_times_raw_over_task", family_rows),
            "effective_ratio": mean("effective_ratio", family_rows),
            "cosine_task_anchor_raw": mean("cosine_task_anchor_raw", family_rows),
            "scale": mean("scale", family_rows),
            "finite": all(bool(row["finite"]) for row in family_rows),
        }
        status_counts: dict[str, int] = {}
        for row in family_rows:
            status = str(row["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        aggregate["status_counts"] = {
            key: status_counts[key] for key in sorted(status_counts)
        }
        families[family] = aggregate
    return {
        "steps": len(rows),
        "rho": float(rows[0]["rho"]),
        "anchor_lambda": float(rows[0]["anchor_lambda"]),
        "task_floor": mean("task_floor", rows),
        "global_task_grad_norm": mean("global_task_grad_norm", rows),
        "global_effective_anchor_grad_norm": mean(
            "global_effective_anchor_grad_norm", rows
        ),
        "global_effective_ratio": mean("global_effective_ratio", rows),
        "max_effective_active_family_ratio": max(
            float(row["max_effective_active_family_ratio"]) for row in rows
        ),
        "total_trainable_parameters": rows[0]["total_trainable_parameters"],
        "family_partition_complete": all(
            bool(row["family_partition_complete"]) for row in rows
        ),
        "families": families,
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


def CIR_LOGIT_SHIFT_EXPERIMENTAL(
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
    raise RuntimeError("disabled experimental logit shift; use exact frozen CIR-V2")
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
    parent_config: Mapping[str, Any] | None = None,
    operational_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    image_state = {name: value.detach().cpu().clone() for name, value in model.image_adapter.state_dict().items()}
    image_parameter_reference = {
        name: value.detach().cpu().clone()
        for name, value in model.image_adapter.named_parameters()
    }
    text_state = {name: value.detach().cpu().clone() for name, value in model.text_adapter.state_dict().items()}
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
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
        "parent_scientific_config": dict(parent_config or parent_scientific_config(config)),
        "resolved_operational_config": dict(operational_config or {}),
        "config_sha256": canonical_json_hash(dict(config)),
        "git_sha": current_git_sha(repo),
        "base_h2_commit": config.get("base_h2_commit", H2_BASE_COMMIT),
        "implementation_git_sha": config.get("implementation_git_sha", current_git_sha(repo)),
        "working_tree_diff_sha256": config.get("working_tree_diff_sha256"),
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
    expected_scientific_config: Mapping[str, Any] | None = None,
    expected_parent_config: Mapping[str, Any] | None = None,
    expected_epoch: int | None = None,
    expected_total_epoch: int | None = None,
    expected_seed: int | None = None,
    expected_clip_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_git_sha: str | None = None,
    expected_worktree_diff_sha256: str | None = None,
) -> tuple[int, int]:
    if expected_scientific_config is not None:
        validate_resume_identity(
            payload,
            expected_scientific_config=expected_scientific_config,
            expected_parent_config=expected_parent_config,
            expected_epoch=expected_epoch,
            expected_total_epoch=expected_total_epoch,
            expected_seed=expected_seed,
            expected_clip_sha256=expected_clip_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_git_sha=expected_git_sha,
            expected_worktree_diff_sha256=expected_worktree_diff_sha256,
        )
    elif int(payload.get("checkpoint_version", 0)) != CHECKPOINT_VERSION or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("exact continuation requires a clean full-state checkpoint from this protocol")
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
        "checkpoint_version",
        "protocol_version",
        "parent_scientific_config",
        "resolved_operational_config",
        "base_h2_commit",
        "implementation_git_sha",
        "working_tree_diff_sha256",
        "seed",
        "precision",
        "amp_enabled",
        "tf32_enabled",
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
