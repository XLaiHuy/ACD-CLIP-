"""Non-mutating fixed-batch PCGrad diagnostics for condition P."""
import torch

from phase2c_pcgrad import GROUPS, project_group, scoped_parameter_groups
from phase2c_utils import DiagnosticStateGuard


PCGRAD_DIAGNOSTIC_FIELDS = [
    "epoch", "batch", "parameter_group", "pre_projection_cls_grad_norm",
    "pre_projection_seg_grad_norm", "pre_projection_cosine",
    "post_projection_cls_grad_norm", "post_projection_seg_grad_norm",
    "post_projection_cosine", "final_combined_grad_norm", "projection_applied",
    "number_of_parameters", "number_of_valid_gradient_tensors",
]


def run_pcgrad_diagnostics(model, optimizer, scheduler, batches, loss_builder, epoch, eps):
    rows, groups = [], scoped_parameter_groups(model)
    with DiagnosticStateGuard(model, optimizer, scheduler):
        for batch_index, batch in enumerate(batches):
            for group in GROUPS:
                named = [(name, parameter) for name, parameter in groups[group] if parameter.requires_grad]
                parameters = [parameter for _, parameter in named]
                if parameters:
                    cls_loss, _ = loss_builder(batch)
                    cls = torch.autograd.grad(cls_loss, parameters, allow_unused=True)
                    _, seg_loss = loss_builder(batch)
                    seg = torch.autograd.grad(seg_loss, parameters, allow_unused=True)
                else:
                    cls = seg = ()
                _, stats = project_group(named, cls, seg, [None] * len(named), eps)
                rows.append({"epoch": epoch, "batch": batch_index, "parameter_group": group, **stats})
    return rows
