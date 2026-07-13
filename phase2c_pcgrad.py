"""Deterministic, module-scoped two-task PCGrad for Phase2C conditions P and P_LoRA_only.

apply_pcgrad() is now parameterized via groups_to_project so that any subset
of the four defined scoped groups can receive PCGrad projection while all
other trainable parameters receive their normal total-loss gradients.
"""
import torch

# Canonical ordered tuple of all supported PCGrad groups.
# Ordering is stable and reproducible across calls.
ALL_GROUPS = ("shared_image_lora", "m_i_w", "hard_text_adapter", "soft_prompt")

EPSILON = 1e-12


def scoped_parameter_groups(model):
    """Return a dict mapping each canonical group name to its (name, param) list.

    The dict always contains exactly the four canonical group keys regardless
    of which parameters are present or trainable.  An empty list indicates
    the model has no parameters matching that group's predicate.
    """
    named = dict(model.named_parameters())
    predicates = {
        "shared_image_lora": lambda name: name.startswith("image_adapter.lora_adapters."),
        "m_i_w": lambda name: name.startswith("image_adapter.m_i_w."),
        "hard_text_adapter": lambda name: name.startswith("text_adapter."),
        "soft_prompt": lambda name: name.startswith("soft_prompt."),
    }
    return {
        group: [(name, parameter) for name, parameter in named.items() if predicate(name)]
        for group, predicate in predicates.items()
    }


def _zero_or_fp32(gradient, parameter, scale_factor=1.0):
    """Return FP32 zeros when gradient is None, otherwise detach, cast to FP32, and unscale."""
    if gradient is None:
        return torch.zeros_like(parameter, dtype=torch.float32)
    unscaled = gradient.detach().to(dtype=torch.float32)
    if scale_factor != 1.0:
        unscaled = unscaled / scale_factor
    return unscaled


def project_group(named_parameters, cls_gradients, seg_gradients, other_gradients, eps=EPSILON, scale_factor=1.0):
    """Project original task gradients symmetrically; retain unprojected other gradients.

    When dot(g_cls, g_seg) >= 0 there is no conflict and the result is
    exactly g_cls + g_seg + g_other (within floating-point tolerance).

    All intermediate tensors are FP32.
    """
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        return [], {
            "projection_applied": False,
            "number_of_parameters": 0,
            "number_of_valid_gradient_tensors": 0,
        }
    cls = [_zero_or_fp32(gradient, parameter, scale_factor) for gradient, parameter in zip(cls_gradients, parameters)]
    seg = [_zero_or_fp32(gradient, parameter, scale_factor) for gradient, parameter in zip(seg_gradients, parameters)]
    other = [_zero_or_fp32(gradient, parameter, scale_factor) for gradient, parameter in zip(other_gradients, parameters)]
    flatten = lambda values: torch.cat([value.reshape(-1) for value in values])
    cls_flat, seg_flat = flatten(cls), flatten(seg)
    dot = torch.dot(cls_flat, seg_flat)
    cls_sq, seg_sq = torch.dot(cls_flat, cls_flat), torch.dot(seg_flat, seg_flat)
    projected = bool(dot.item() < 0)
    if projected:
        cls_post = [left - dot / (seg_sq + eps) * right for left, right in zip(cls, seg)]
        seg_post = [right - dot / (cls_sq + eps) * left for left, right in zip(cls, seg)]
    else:
        cls_post, seg_post = cls, seg
    post_cls, post_seg = flatten(cls_post), flatten(seg_post)
    cosine = lambda left, right: torch.dot(left, right) / (left.norm() * right.norm() + eps)
    result = [left + right + extra for left, right, extra in zip(cls_post, seg_post, other)]
    return result, {
        "pre_projection_cls_grad_norm": float(cls_flat.norm()),
        "pre_projection_seg_grad_norm": float(seg_flat.norm()),
        "pre_projection_cosine": float(cosine(cls_flat, seg_flat)),
        "post_projection_cls_grad_norm": float(post_cls.norm()),
        "post_projection_seg_grad_norm": float(post_seg.norm()),
        "post_projection_cosine": float(cosine(post_cls, post_seg)),
        "final_combined_grad_norm": float(flatten(result).norm()),
        "projection_applied": projected,
        "number_of_parameters": len(parameters),
        "number_of_valid_gradient_tensors": sum(
            a is not None or b is not None
            for a, b in zip(cls_gradients, seg_gradients)
        ),
    }


def apply_pcgrad(
    total_loss,
    cls_loss_weighted,
    seg_loss_weighted,
    model,
    groups_to_project,
    eps=EPSILON,
    scale_factor=1.0,
):
    """Populate .grad using PCGrad for selected groups; other parameters use standard gradients.

    Parameters
    ----------
    total_loss : Tensor
        The complete scalar training loss (cls + seg + all regularization terms).
    cls_loss_weighted : Tensor
        The weighted classification task loss.  Under the current protocol
        both task coefficients are 1.0, so this equals cls_loss.  The
        argument is named _weighted to remain correct if coefficients change.
    seg_loss_weighted : Tensor
        The weighted segmentation task loss.  Same note as above.
    model : nn.Module
        The model whose trainable parameters will receive gradients.
    groups_to_project : sequence of str
        Subset of scoped group names that receive PCGrad projection.
        Must be a non-empty sequence of names that all exist in
        scoped_parameter_groups(model).  No duplicates allowed.
        Each named group must contain at least one trainable parameter.
        No trainable parameter may appear in more than one entry.
    eps : float
        Small value for numerical stability in projection math.
    scale_factor : float
        Manual loss scaling factor to prevent FP16 underflow.

    The caller must call optimizer.zero_grad(set_to_none=True) before this
    function and must NOT call total_loss.backward() afterwards.

    Returns
    -------
    dict mapping each projected group name to its per-group stats dict.
    """
    groups = scoped_parameter_groups(model)
    available_groups = set(groups.keys())

    # ── Validation ────────────────────────────────────────────────────────────
    if not groups_to_project:
        raise ValueError(
            "groups_to_project must not be empty when apply_pcgrad is called"
        )

    groups_seq = list(groups_to_project)
    unknown = set(groups_seq) - available_groups
    if unknown:
        raise ValueError(
            f"Unknown PCGrad group names: {sorted(unknown)}. "
            f"Available: {sorted(available_groups)}"
        )

    if len(groups_seq) != len(set(groups_seq)):
        seen, duplicates = set(), []
        for name in groups_seq:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(f"Duplicate group names in groups_to_project: {duplicates}")

    # Verify each group has trainable parameters and that no parameter appears
    # in more than one requested group.
    seen_param_ids: dict[int, str] = {}
    for group_name in groups_seq:
        trainable = [(name, p) for name, p in groups[group_name] if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"PCGrad group '{group_name}' contains no trainable parameters. "
                "Check that model parameters are enabled (requires_grad=True) "
                "and that the group predicate matches the actual parameter names."
            )
        for param_name, param in trainable:
            pid = id(param)
            if pid in seen_param_ids:
                raise ValueError(
                    f"Trainable parameter '{param_name}' (id={pid}) is assigned "
                    f"to both '{seen_param_ids[pid]}' and '{group_name}'. "
                    "Each trainable parameter must belong to at most one PCGrad group."
                )
            seen_param_ids[pid] = group_name

    # ── Build scoped and unscoped parameter lists ──────────────────────────────
    scoped = [
        parameter
        for group_name in groups_seq
        for _, parameter in groups[group_name]
        if parameter.requires_grad
    ]
    scoped_ids = {id(p) for p in scoped}
    all_parameters = [p for p in model.parameters() if p.requires_grad]

    # ── Gradient computation ───────────────────────────────────────────────────
    # Total gradients for all parameters (used for unscoped parameters).
    # If scale_factor is not 1.0, we scale the loss before backward.
    total_grads = torch.autograd.grad(
        total_loss * scale_factor, all_parameters, allow_unused=True, retain_graph=True
    )
    # Separate task gradients for scoped parameters only.
    cls_grads = torch.autograd.grad(
        cls_loss_weighted * scale_factor, scoped, allow_unused=True, retain_graph=True
    )
    seg_grads = torch.autograd.grad(
        seg_loss_weighted * scale_factor, scoped, allow_unused=True, retain_graph=False
    )
    # Calculate other-loss gradients (regularization terms) via linear subtraction.
    # Since loss = cls_loss_weighted + seg_loss_weighted + other_loss,
    # we have other_grad = total_grad - cls_grad - seg_grad.
    total_grad_map = {id(p): g for p, g in zip(all_parameters, total_grads)}
    other_grads_list = []
    for i, p in enumerate(scoped):
        t_val = total_grad_map[id(p)]
        if t_val is None:
            other_grads_list.append(None)
        else:
            c_val = cls_grads[i]
            s_val = seg_grads[i]
            if c_val is None and s_val is None:
                other_grads_list.append(t_val)
            elif c_val is None:
                other_grads_list.append(t_val - s_val)
            elif s_val is None:
                other_grads_list.append(t_val - c_val)
            else:
                other_grads_list.append(t_val - c_val - s_val)
    other_grads = tuple(other_grads_list)

    # ── Assign standard gradients to unscoped parameters ──────────────────────
    for parameter, gradient in zip(all_parameters, total_grads):
        if id(parameter) not in scoped_ids:
            if gradient is None:
                parameter.grad = None
            else:
                unscaled_grad = gradient.detach().to(dtype=torch.float32)
                if scale_factor != 1.0:
                    unscaled_grad = unscaled_grad / scale_factor
                parameter.grad = unscaled_grad.clone().to(
                    device=parameter.device, dtype=parameter.dtype
                )

    # ── Apply PCGrad projection to each requested group ────────────────────────
    offset = 0
    summaries: dict = {}
    for group_name in groups_seq:
        named = [
            (name, parameter)
            for name, parameter in groups[group_name]
            if parameter.requires_grad
        ]
        count = len(named)
        final, stats = project_group(
            named,
            cls_grads[offset: offset + count],
            seg_grads[offset: offset + count],
            other_grads[offset: offset + count],
            eps,
            scale_factor=scale_factor,
        )
        for (_, parameter), gradient in zip(named, final):
            parameter.grad = (
                None
                if gradient is None
                else gradient.detach().clone().to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
        summaries[group_name] = stats
        offset += count

    return summaries
