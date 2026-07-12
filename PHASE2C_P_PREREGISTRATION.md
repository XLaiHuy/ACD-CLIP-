# Phase2C Condition P preregistration: module-scoped PCGrad

## Hypothesis

The remaining A-prime/C trade-off is partly caused by conflicting
classification and segmentation gradients on parameters shared by the two
tasks.  Deterministic two-task PCGrad on those shared modules may reduce
destructive interference without changing the model, losses, schedule, or
selection policy.

## Parent and invariants

P is exactly BF16 A-prime (`alpha_max=0.20`, no activation delay) with seed
42.  The architecture, VisA seed-42 train/validation manifests, 15 epochs,
BF16 autocast, batch size 6, six workers, optimizer type and hyperparameters,
learning rates, scheduler, loss weights, A-prime alpha/beta schedules,
augmentations, deterministic sampler/order, checkpoint selection, validation,
`cls_only` image score, and fixed diagnostic batches are unchanged.

P does not add a delay, optimizer/LR restart, freezing, GradNorm, adaptive
loss weighting, consistency loss, distillation, medical evaluation, or
multi-seed execution.

## Scoped parameters

PCGrad is applied only to these named shared parameter groups:

- `shared_image_lora`
- `m_i_w`
- `hard_text_adapter`
- `soft_prompt`

The task-specific classification head, task-specific segmentation head,
segmentation-only DFG/SS2D parameters, and every other task-exclusive
parameter keep ordinary autograd gradients.  Frozen parameters and parameters
not used by a task are not projected.

## Deterministic projection and regularization rule

For each configured group, concatenate the original unmodified FP32 flattened
classification and segmentation gradients, `g_cls` and `g_seg`, in the fixed
parameter order defined by the model.  Let `dot = g_cls · g_seg` and use
`eps = 1e-12`.

- If `dot >= 0`, set `g_final = g_cls + g_seg` exactly (up to normal floating
  point tolerance); no projection is applied.
- If `dot < 0`, deterministically and symmetrically project both original
  gradients:

  ```text
  g_cls_projected = g_cls - dot / (||g_seg||^2 + eps) * g_seg
  g_seg_projected = g_seg - dot / (||g_cls||^2 + eps) * g_cls
  g_final = g_cls_projected + g_seg_projected
  ```

No random task ordering is used.  Projection math is FP32, then each slice is
restored to its parameter's native gradient dtype/device.

All loss terms other than `cls_loss` and `seg_loss` retain their ordinary
autograd behavior.  Specifically, the implementation computes gradients of
`other_loss = total_loss - cls_loss - seg_loss` and adds those unprojected
gradients to `g_final` for scoped parameters; this includes KG/K
regularization (and therefore preserves soft-prompt regularization).  For
unscoped parameters, gradients of the original complete `total_loss` are used.

## Logging

For every epoch, fixed diagnostic batch, and configured PCGrad group, write a
row to `pcgrad_diagnostics.csv` with: epoch, batch, parameter group,
pre-projection CLS/SEG norms and cosine, post-projection CLS/SEG norms and
cosine, final combined-gradient norm, projection flag, parameter count, and
valid-gradient-tensor count.  Run metadata records the parent condition,
enabled flag, groups, variant, epsilon, BF16 precision, and training commit
SHA.

## Checkpoint rule and decision criteria

The existing per-run rule is unchanged: eligible checkpoints have image AP at
least one point below the best image AP in that run; select highest Pixel AP,
then image AP, then earliest epoch.  Primary success is a selected P checkpoint
with Pixel AP above A-prime e13 while satisfying that image-AP guardrail.
Secondary evidence is an expanded four-metric Pareto frontier and reduced
negative-cosine diagnostics without materially worse image metrics.  Failure
includes no primary improvement, invalid/unstable gradients, or merely moving
conflict to excluded/task-specific modules.

P is a seed-42 exploratory experiment until replicated with the fixed-split
training-seed protocol.
