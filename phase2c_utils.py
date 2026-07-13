"""Shared deterministic protocol, validation selection, and gradient diagnostics for Phase2C."""
import csv
import copy
import hashlib
import io
import json
import math
import random
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


NA = "NA"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class EpochDeterministicSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def alpha_for_epoch(epoch, alpha_max, freeze_epochs=3):
    if epoch <= freeze_epochs:
        return 0.0
    warm = epoch - freeze_epochs
    return alpha_max * (0.25 if warm == 1 else 0.50 if warm == 2 else 1.0)


# PCGrad conditions that are children of A_prime.
_PCGRAD_CONDITIONS = {"P", "P_LoRA_only"}


def phase2c_config(condition, save_path, alpha_max):
    """Return the full reproducible configuration dict for the given Phase2C condition.

    Conditions
    ----------
    A_prime : baseline, hybrid_alpha_max=0.20, no PCGrad
    B       : hybrid_alpha_max=0.15, no PCGrad
    C       : delayed activation, hybrid_alpha_max=0.20, no PCGrad
    P       : A_prime + PCGrad on all four shared groups
    P_LoRA_only : A_prime + PCGrad on shared_image_lora only
    """
    expected_alpha = {"A_prime": 0.20, "B": 0.15, "C": 0.20, "P": 0.20, "P_LoRA_only": 0.20}
    if condition not in expected_alpha:
        raise ValueError(f"condition must be one of {sorted(expected_alpha)}")
    expected = expected_alpha[condition]
    if not math.isclose(alpha_max, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{condition} requires hybrid_alpha_max={expected}")
    activation_delay_epochs = 2 if condition == "C" else 0
    soft_prompt_freeze_epochs = 3 + activation_delay_epochs
    pcgrad_enabled = condition in _PCGRAD_CONDITIONS
    if condition == "P":
        pcgrad_groups = ["shared_image_lora", "m_i_w", "hard_text_adapter", "soft_prompt"]
        pcgrad_variant = "deterministic_symmetric_two_task"
        parent_condition = "A_prime"
    elif condition == "P_LoRA_only":
        pcgrad_groups = ["shared_image_lora"]
        pcgrad_variant = "symmetric_module_scoped_lora_only"
        parent_condition = "A_prime"
    else:
        pcgrad_groups = []
        pcgrad_variant = None
        parent_condition = None
    config = {
        "condition": condition,
        "save_path": str(save_path),
        "dataset": "VisA",
        "epochs": 15,
        "seed": 42,
        "n_groups": 3,
        "dfg_mode": "attn",
        "dfg_attn_dim": 256,
        "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True,
        "dfg_gamma_max": 0.2,
        "dfg_ss2d_fusion": "weight_residual",
        "dfg_beta": 0.10,
        "dfg_beta_schedule": "warmup010",
        "dfg_beta_target": 0.10,
        "activation_delay_epochs": activation_delay_epochs,
        "text_adapt_weight": 0.2,
        "use_hybrid_soft_prompt": True,
        "hybrid_alpha_max": alpha_max,
        "alpha_schedule": [
            alpha_for_epoch(epoch, alpha_max, soft_prompt_freeze_epochs) for epoch in range(1, 16)
        ],
        "soft_prompt_ctx_len": 4,
        "soft_prompt_lr": 0.00005,
        "soft_prompt_freeze_epochs": soft_prompt_freeze_epochs,
        "lambda_kg": 0.01,
        "lambda_k": 0.002,
        "image_lr": 0.001,
        "text_lr": 0.0005,
        "lr_gamma": 0.9,
        "grad_clip_norm": 1.0,
        "non_finite_loss_abort_threshold": 20,
        "batch_size": 6,
        "num_workers": 6,
        "img_size": 518,
        "amp": True,
        "grad_checkpointing": True,
        "score_rule": "cls_only",
        "parent_condition": parent_condition,
        "pcgrad_enabled": pcgrad_enabled,
        "pcgrad_groups": pcgrad_groups,
        "pcgrad_variant": pcgrad_variant,
        "pcgrad_epsilon": 1e-12 if pcgrad_enabled else None,
        "precision": "bf16" if pcgrad_enabled else None,
    }
    return config


# Fields excluded from normalized cross-condition comparison.
# condition, save_path, hybrid_alpha_max, and alpha_schedule vary by design.
# PCGrad metadata and parent linkage are condition-specific and must not
# cause false inequality in scientific-field comparisons.
_NORMALIZED_IGNORED = {
    "condition",
    "save_path",
    "hybrid_alpha_max",
    "alpha_schedule",
    "parent_condition",
    "pcgrad_enabled",
    "pcgrad_groups",
    "pcgrad_variant",
    "pcgrad_epsilon",
    "precision",
}


def normalized_config(config):
    """Return config with condition-specific and PCGrad metadata fields removed.

    Use this function to compare scientific fields across conditions.
    """
    return {key: value for key, value in config.items() if key not in _NORMALIZED_IGNORED}


def image_anchor(rows):
    early = sorted(
        (float(row["image_ap"]) for row in rows if 1 <= int(row["epoch"]) <= 3),
        reverse=True,
    )
    if len(early) < 2:
        raise ValueError("Image anchor requires image AP for at least two of epochs 1-3")
    return sum(early[:2]) / 2.0


def select_checkpoint(rows, tolerance=1.0):
    anchor = image_anchor(rows)
    eligible = [row for row in rows if float(row["image_ap"]) >= anchor - tolerance]
    if not eligible:
        raise ValueError("No checkpoint satisfies image anchor constraint")
    winner = max(
        eligible,
        key=lambda row: (float(row["pixel_ap"]), float(row["image_ap"]), -int(row["epoch"])),
    )
    return {
        "image_anchor": anchor,
        "constraint": f"image_ap >= {anchor - tolerance:.6f}",
        "selected_epoch": int(winner["epoch"]),
        "selected_checkpoint": winner.get("checkpoint", f"checkpoints/adapter_{int(winner['epoch'])}.pth"),
        "selected_metrics": {
            key: float(winner[key]) for key in ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
        },
        "tie_break": ["pixel_ap_desc", "image_ap_desc", "epoch_asc"],
    }


def write_selection(rows, path):
    selection = select_checkpoint(rows)
    Path(path).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection


def append_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _update_hash(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(repr(key).encode())
            _update_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _update_hash(digest, item)
    else:
        digest.update(repr(value).encode())


def state_checksum(value):
    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def _rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_equal(left, right):
    return (
        left["python"] == right["python"]
        and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["cpu"], right["cpu"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))
    )


class DiagnosticStateGuard(AbstractContextManager):
    def __init__(self, model, optimizer=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __enter__(self):
        self.rng = _rng_snapshot()
        self.modes = {name: module.training for name, module in self.model.named_modules()}
        self.params = {name: value.detach().cpu().clone() for name, value in self.model.named_parameters()}
        self.buffers = {name: value.detach().cpu().clone() for name, value in self.model.named_buffers()}
        self.optimizer_checksum = state_checksum(self.optimizer.state_dict()) if self.optimizer else None
        self.scheduler_checksum = state_checksum(self.scheduler.state_dict()) if self.scheduler else None
        self.model.eval()
        return self

    def __exit__(self, exc_type, exc, traceback):
        current_params = {name: value.detach().cpu() for name, value in self.model.named_parameters()}
        params_ok = self.params.keys() == current_params.keys() and all(
            torch.equal(value, current_params[name]) for name, value in self.params.items()
        )
        current_buffers = dict(self.model.named_buffers())
        buffers_ok = self.buffers.keys() == current_buffers.keys()
        for name, value in self.buffers.items():
            if name in current_buffers:
                buffers_ok = buffers_ok and torch.equal(value, current_buffers[name].detach().cpu())
                current_buffers[name].copy_(value.to(current_buffers[name].device))
        modules = dict(self.model.named_modules())
        for name, mode in self.modes.items():
            modules[name].train(mode)
        before_restore_rng = _rng_snapshot()
        _restore_rng(self.rng)
        optimizer_ok = self.optimizer is None or self.optimizer_checksum == state_checksum(self.optimizer.state_dict())
        scheduler_ok = self.scheduler is None or self.scheduler_checksum == state_checksum(self.scheduler.state_dict())
        if exc_type is None:
            if not params_ok:
                raise AssertionError("Gradient diagnostics changed model parameters")
            if not buffers_ok:
                raise AssertionError("Gradient diagnostics changed mutable buffers")
            if not optimizer_ok:
                raise AssertionError("Gradient diagnostics changed optimizer state")
            if not scheduler_ok:
                raise AssertionError("Gradient diagnostics changed scheduler state")
            if not all(modules[name].training == mode for name, mode in self.modes.items()):
                raise AssertionError("Gradient diagnostics failed to restore module modes")
            if not _rng_equal(self.rng, _rng_snapshot()):
                raise AssertionError("Gradient diagnostics failed to restore RNG state")
        return False


def parameter_groups(model):
    named = dict(model.named_parameters())
    specs = {
        "shared_image_lora": lambda name: name.startswith("image_adapter.lora_adapters."),
        "m_i_w": lambda name: name.startswith("image_adapter.m_i_w."),
        "hard_text_adapter": lambda name: name.startswith("text_adapter."),
        "soft_prompt": lambda name: name.startswith("soft_prompt."),
    }
    return {
        group: [(name, parameter) for name, parameter in named.items() if predicate(name)]
        for group, predicate in specs.items()
    }


def _flatten_gradients(loss, named_parameters):
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    if not parameters or not loss.requires_grad:
        return None
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False)
    pieces = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
    if not pieces:
        return None
    flat = torch.cat(pieces)
    if not torch.isfinite(flat).all() or flat.norm().item() == 0:
        return None
    return flat

def _cpu_gradient_tensors(loss, named_parameters):
    """Materialize diagnostic gradients on CPU without a GPU-wide concatenation."""
    parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if not parameters or not loss.requires_grad:
        return None
    gradients = torch.autograd.grad(
        loss, [parameter for _, parameter in parameters], allow_unused=True, retain_graph=False
    )
    tensors = {}
    for (name, _), gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        tensor = gradient.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not torch.isfinite(tensor).all():
            return None
        tensors[name] = tensor
    if not tensors or not any(tensor.norm().item() != 0 for tensor in tensors.values()):
        return None
    return tensors


def gradient_tensor_pair_stats(cls_tensors, seg_tensors):
    """Compute norms and cosine from per-parameter CPU tensors."""
    def squared_norm(tensors):
        return sum(torch.sum(tensor * tensor).item() for tensor in tensors.values())

    cls_norm_sq = squared_norm(cls_tensors) if cls_tensors is not None else None
    seg_norm_sq = squared_norm(seg_tensors) if seg_tensors is not None else None
    shared = () if cls_tensors is None or seg_tensors is None else cls_tensors.keys() & seg_tensors.keys()
    dot = sum(torch.sum(cls_tensors[name] * seg_tensors[name]).item() for name in shared)
    return {
        "cls_grad_norm": NA if cls_norm_sq is None else float(cls_norm_sq ** 0.5),
        "seg_grad_norm": NA if seg_norm_sq is None else float(seg_norm_sq ** 0.5),
        "cosine": NA if cls_norm_sq is None or seg_norm_sq is None else float(
            dot / (cls_norm_sq * seg_norm_sq) ** 0.5
        ),
    }


def gradient_pair_stats(cls_gradient, seg_gradient):
    return {
        "cls_grad_norm": NA if cls_gradient is None else float(cls_gradient.norm().item()),
        "seg_grad_norm": NA if seg_gradient is None else float(seg_gradient.norm().item()),
        "cosine": NA if cls_gradient is None or seg_gradient is None else float(
            torch.nn.functional.cosine_similarity(cls_gradient, seg_gradient, dim=0).item()
        ),
    }


def run_gradient_diagnostics(model, optimizer, scheduler, batches, loss_builder, epoch):
    rows = []
    groups = parameter_groups(model)
    with DiagnosticStateGuard(model, optimizer, scheduler):
        for batch_index, batch in enumerate(batches):
            for group_name, named_parameters in groups.items():
                cls_loss, _ = loss_builder(batch)
                cls_gradient = _cpu_gradient_tensors(cls_loss, named_parameters)
                _, seg_loss = loss_builder(batch)
                seg_gradient = _cpu_gradient_tensors(seg_loss, named_parameters)
                rows.append({
                    "epoch": epoch,
                    "batch": batch_index,
                    "parameter_group": group_name,
                    **gradient_tensor_pair_stats(cls_gradient, seg_gradient),
                })
    return rows


def select_diagnostic_batch_ids(dataset, batch_size, seed=42, batch_count=3):
    ids = [record["sample_id"] for record in dataset.meta]
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    needed = batch_size * batch_count
    if len(order) < needed:
        raise ValueError(f"Need at least {needed} training samples for diagnostic batches")
    return [[ids[index] for index in order[start:start + batch_size]] for start in range(0, needed, batch_size)]


def persist_diagnostic_batch_ids(dataset, batch_size, path, seed=42, batch_count=3):
    batches = select_diagnostic_batch_ids(dataset, batch_size, seed, batch_count)
    payload = {"seed": seed, "batch_size": batch_size, "batches": batches}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return batches
