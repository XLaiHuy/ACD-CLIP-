"""Non-mutating fixed-batch PCGrad diagnostics for Phase2C conditions P and P_LoRA_only.

run_pcgrad_diagnostics() now accepts groups_to_project so that only the groups
that receive PCGrad projection are logged in pcgrad_diagnostics.csv.

The generic gradient_diagnostics.csv (written by run_gradient_diagnostics in
phase2c_utils.py) continues to observe all four shared groups regardless of
which groups receive PCGrad, so conflict migration to other groups is visible.
"""
import torch

from phase2c_pcgrad import project_group, scoped_parameter_groups
from phase2c_utils import DiagnosticStateGuard


PCGRAD_DIAGNOSTIC_FIELDS = [
    "epoch",
    "batch",
    "parameter_group",
    "pre_projection_cls_grad_norm",
    "pre_projection_seg_grad_norm",
    "pre_projection_cosine",
    "post_projection_cls_grad_norm",
    "post_projection_seg_grad_norm",
    "post_projection_cosine",
    "final_combined_grad_norm",
    "projection_applied",
    "number_of_parameters",
    "number_of_valid_gradient_tensors",
]


def run_pcgrad_diagnostics(model, optimizer, scheduler, batches, loss_builder, epoch, groups_to_project, eps):
    """Compute PCGrad projection diagnostics for the requested groups only.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer
    scheduler : LRScheduler
    batches : list of batch dicts
        Fixed diagnostic batches (same IDs as used during training).
    loss_builder : callable
        Called as ``cls_loss, seg_loss = loss_builder(batch)``.
        Must not mutate model parameters.
    epoch : int
    groups_to_project : sequence of str
        Only groups listed here are logged.  Must match the groups actually
        receiving PCGrad projection in the training step.
    eps : float
        Epsilon passed to project_group for numerical stability.

    Returns
    -------
    list of row dicts matching PCGRAD_DIAGNOSTIC_FIELDS.

    Notes
    -----
    Uses DiagnosticStateGuard to guarantee that model parameters,
    optimizer/scheduler state, and RNG state are unchanged after the call.
    The generic gradient_diagnostics.csv is written separately by
    run_gradient_diagnostics() in phase2c_utils.py and always covers all
    four canonical shared groups.
    """
    rows = []
    groups = scoped_parameter_groups(model)

    with DiagnosticStateGuard(model, optimizer, scheduler):
        for batch_index, batch in enumerate(batches):
            for group_name in groups_to_project:
                named = [
                    (name, parameter)
                    for name, parameter in groups[group_name]
                    if parameter.requires_grad
                ]
                parameters = [parameter for _, parameter in named]
                if parameters:
                    cls_loss, _ = loss_builder(batch)
                    cls = torch.autograd.grad(cls_loss, parameters, allow_unused=True)
                    _, seg_loss = loss_builder(batch)
                    seg = torch.autograd.grad(seg_loss, parameters, allow_unused=True)
                else:
                    cls = seg = ()
                _, stats = project_group(named, cls, seg, [None] * len(named), eps)
                rows.append({
                    "epoch": epoch,
                    "batch": batch_index,
                    "parameter_group": group_name,
                    **stats,
                })
    return rows
