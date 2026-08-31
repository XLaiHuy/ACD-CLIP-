# Selected solution contract

Solution ID: `SELECTIVE_PHASE2B_ANCHOR`

## Intended change

Add a normalized parameter-distance penalty over `model.image_adapter`
parameters:

`loss = cls_loss + seg_loss + lambda_kg * kg_loss + lambda_k * k_loss + lambda_image_anchor * anchor_loss`

where `anchor_loss` is the mean, per-parameter squared distance from a frozen
matched Phase2B E14 image-adapter state, normalized by that reference tensor's
squared norm. The first bounded coefficient is preregistered as
`lambda_image_anchor = 1e-3`; it is not selected from Medical or MVTec data.

## Frozen reference

- Parent checkpoint: matched Phase2B E14 checkpoint from the corrective run.
- Scope: `image_adapter` parameters only.
- Not anchored: text adapter, soft prompt, RMT peer/transport state, CLIP
  backbone, optimizer state, or inference/deployment operators.

## Invariants

Adam groups, betas, epsilon, weight decay, base LRs, StepLR step size/gamma and
post-epoch timing, FP32/AMP/TF32 settings, effective batch six, seed, VisA
source, CLIP asset, image size, loss terms, prompt freeze/unfreeze schedule,
DFG schedule, checkpoint schedule, evaluator, and Gaussian deployment remain
unchanged. The anchor is optional with default coefficient zero so the identity
path remains testable.

## Validation requirements

The implementation must prove optimizer-group identity, scheduler/resume
identity, FP32, checkpoint identity, frozen-reference no-grad behavior, and
finite real-VisA smoke loss/gradient. A short source-only gate may reject or
mark the candidate inconclusive; it cannot authorize the full 20-epoch run by
itself.
