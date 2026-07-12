"""Deterministic, module-scoped two-task PCGrad for Phase2C condition P."""
import torch

GROUPS = ("shared_image_lora", "m_i_w", "hard_text_adapter", "soft_prompt")
EPSILON = 1e-12


def scoped_parameter_groups(model):
    named = dict(model.named_parameters())
    predicates = {
        "shared_image_lora": lambda name: name.startswith("image_adapter.lora_adapters."),
        "m_i_w": lambda name: name.startswith("image_adapter.m_i_w."),
        "hard_text_adapter": lambda name: name.startswith("text_adapter."),
        "soft_prompt": lambda name: name.startswith("soft_prompt."),
    }
    return {group: [(name, parameter) for name, parameter in named.items() if predicate(name)] for group, predicate in predicates.items()}


def _zero_or_fp32(gradient, parameter):
    return torch.zeros_like(parameter, dtype=torch.float32) if gradient is None else gradient.detach().to(dtype=torch.float32)


def project_group(named_parameters, cls_gradients, seg_gradients, other_gradients, eps=EPSILON):
    """Project original task gradients symmetrically; retain unprojected other gradients."""
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        return [], {"projection_applied": False, "number_of_parameters": 0, "number_of_valid_gradient_tensors": 0}
    cls = [_zero_or_fp32(gradient, parameter) for gradient, parameter in zip(cls_gradients, parameters)]
    seg = [_zero_or_fp32(gradient, parameter) for gradient, parameter in zip(seg_gradients, parameters)]
    other = [_zero_or_fp32(gradient, parameter) for gradient, parameter in zip(other_gradients, parameters)]
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
        "pre_projection_cls_grad_norm": float(cls_flat.norm()), "pre_projection_seg_grad_norm": float(seg_flat.norm()),
        "pre_projection_cosine": float(cosine(cls_flat, seg_flat)), "post_projection_cls_grad_norm": float(post_cls.norm()),
        "post_projection_seg_grad_norm": float(post_seg.norm()), "post_projection_cosine": float(cosine(post_cls, post_seg)),
        "final_combined_grad_norm": float(flatten(result).norm()), "projection_applied": projected,
        "number_of_parameters": len(parameters), "number_of_valid_gradient_tensors": sum(a is not None or b is not None for a, b in zip(cls_gradients, seg_gradients)),
    }


def apply_pcgrad(total_loss, cls_loss, seg_loss, model, eps=EPSILON):
    """Populate .grad without duplicate accumulation; unscoped parameters use total loss."""
    groups = scoped_parameter_groups(model)
    scoped = [parameter for group in GROUPS for _, parameter in groups[group] if parameter.requires_grad]
    scoped_ids = {id(parameter) for parameter in scoped}
    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total = torch.autograd.grad(total_loss, all_parameters, allow_unused=True, retain_graph=True)
    cls = torch.autograd.grad(cls_loss, scoped, allow_unused=True, retain_graph=True)
    seg = torch.autograd.grad(seg_loss, scoped, allow_unused=True, retain_graph=True)
    other = torch.autograd.grad(total_loss - cls_loss - seg_loss, scoped, allow_unused=True)
    for parameter, gradient in zip(all_parameters, total):
        if id(parameter) not in scoped_ids:
            parameter.grad = None if gradient is None else gradient.detach()
    offset, summaries = 0, {}
    for group in GROUPS:
        named = [(name, parameter) for name, parameter in groups[group] if parameter.requires_grad]
        count = len(named)
        final, stats = project_group(named, cls[offset:offset + count], seg[offset:offset + count], other[offset:offset + count], eps)
        for (_, parameter), gradient in zip(named, final):
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)
        summaries[group] = stats
        offset += count
    return summaries
